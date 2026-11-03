"""Yerel web arayüzü (FastAPI). Yalnızca 127.0.0.1. CLI ile aynı pipeline fonksiyonlarını iş kuyruğundan çağırır.

Çalıştırma: python -m clipper.web [--port 8770] [--open]
"""
import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import webbrowser
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import censor
from . import review as review_mod
from .common import ROOT, load_config, read_json, slug, video_meta, work_dir, write_json
from .jobs import FINAL, JobQueue
from .pipeline import VIDEO_EXT, save_meta

SAMPLES = ROOT / "samples"
OUT = ROOT / "out"
WORK = ROOT / "work"
SHARES = WORK / "_shares.json"
CONFIG = ROOT / "config.yaml"
PLATFORMS = ("youtube", "instagram", "tiktok")
MAX_UPLOAD = 3 * 1024 ** 3
log = logging.getLogger("web")

# Ayarlar sayfasındaki alanlar: yol → tür/seçenekler
SETTINGS = {
    "render.layout": ["crop", "blur_bg"],
    "upscale.mode": ["auto", "always", "off"],
    "laughter.threshold": float,
    "select.max_bits": int,
    "candidates.max_len": int,
    "subs.size": int,
    "subs.note_size": int,
    "claude.mode": ["cli", "api"],
    "claude.model": str,
    "censor.enabled": bool,
    "censor.mode": ["full", "first_letter"],
    "censor.include_mild": bool,
}

_probe_cache = {}


def probe(path):
    """Süre ve çözünürlük (mtime'a göre önbellekli)."""
    key = (str(path), path.stat().st_mtime)
    if key not in _probe_cache:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height:format=duration", "-of", "json", str(path)],
                             capture_output=True, text=True)
        try:
            d = json.loads(out.stdout)
            s = d["streams"][0]
            _probe_cache[key] = {"duration": float(d["format"]["duration"]), "width": s["width"], "height": s["height"]}
        except (ValueError, KeyError, IndexError):
            _probe_cache[key] = {"duration": None, "width": None, "height": None}
    return _probe_cache[key]


def sample_path(name):
    """samples/ içindeki dosya; klasör dışına çıkmayı engeller."""
    p = SAMPLES / name
    if Path(name).name != name or not p.is_file():
        raise HTTPException(404, "Video bulunamadı")
    return p


def output_path(rel):
    p = (OUT / rel).resolve()
    if OUT.resolve() not in p.parents or not p.is_file():
        raise HTTPException(404, "Çıktı bulunamadı")
    return p


def upload_name(comedian, show, ext):
    """Komedyen + gösteri → "komedyen__gosteri.mp4"."""
    part = lambda s: slug(s.strip().lower(), 60)
    if not comedian.strip() or not show.strip():
        raise HTTPException(400, "Komedyen ve gösteri adı gerekli")
    ext = ext.lower() if ext.lower() in VIDEO_EXT else ".mp4"
    return f"{part(comedian)}__{part(show)}{ext}"


def video_status(name, wd, job):
    if job and job["status"] in ("queued", "running"):
        return "sırada" if job["status"] == "queued" else "işleniyor"
    if job and job["status"] in ("error", "limit"):
        return "hata"
    if any((OUT / wd.name).glob("*.mp4")):
        return "render edildi"
    if (wd / "translations_checked.json").exists() and (wd / "reframe.json").exists():
        return "onay bekliyor"
    return "yeni"


def library(queue):
    items = []
    for p in sorted(SAMPLES.iterdir()) if SAMPLES.exists() else []:
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        wd = work_dir(p)
        comedian, show = video_meta(p)
        job = queue.latest_for(p.name)
        info = probe(p)
        sel = wd / "selected.json"
        items.append({
            "name": p.name, "comedian": comedian, "show": show, **info,
            "low_res": bool(info["height"] and info["height"] < 720),
            "status": video_status(p.name, wd, job), "job": job,
            "bits": len(read_json(sel)) if sel.exists() else None,
            "outputs": len(list((OUT / wd.name).glob("*.mp4"))),
        })
    return items


def outputs():
    shares = read_json(SHARES) if SHARES.exists() else {}
    items = []
    for d in sorted(p for p in OUT.iterdir() if p.is_dir()) if OUT.exists() else []:
        sel_p = WORK / d.name / "selected.json"
        bits = {str(b["n"]): b for b in read_json(sel_p)} if sel_p.exists() else {}
        for f in sorted(d.glob("*.mp4"), key=lambda f: (int(f.name.split("_")[0]) if f.name.split("_")[0].isdigit() else 0, f.name)):
            rel = f"{d.name}/{f.name}"
            b = bits.get(f.name.split("_")[0], {})
            txt = f.with_suffix(".txt")
            items.append({"path": rel, "video": d.name, "hook": b.get("hook_tr") or f.stem,
                          "score": b.get("score"), "duration": probe(f)["duration"],
                          "text": txt.read_text(encoding="utf-8") if txt.exists() else "",
                          "shares": shares.get(rel, {})})
    return items


def read_settings():
    cfg = load_config()
    out = {}
    for path in SETTINGS:
        a, b = path.split(".")
        out[path] = cfg[a][b]
    return out


def write_settings(values):
    """config.yaml'ı yorumları koruyarak günceller (ruamel round-trip)."""
    from ruamel.yaml import YAML
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # uzun satırlar (ör. initial_prompt) kırılmasın
    data = yaml.load(CONFIG.read_text(encoding="utf-8"))
    for path, v in values.items():
        if path not in SETTINGS:
            raise HTTPException(400, f"Bilinmeyen ayar: {path}")
        kind = SETTINGS[path]
        if isinstance(kind, list):
            if v not in kind:
                raise HTTPException(400, f"{path}: {kind} değerlerinden biri olmalı")
            v = DoubleQuotedScalarString(v)  # çıplak off, PyYAML'da False okunur
        elif kind is bool:
            if str(v).lower() not in ("true", "false"):
                raise HTTPException(400, f"{path}: true/false olmalı")
            v = str(v).lower() == "true"
        else:
            try:
                v = kind(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{path}: geçersiz değer {v!r}")
        a, b = path.split(".")
        data[a][b] = v
    with open(CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


class StartBody(BaseModel):
    comedian: str = ""
    auto_retry: bool = False


class RenderBody(BaseModel):
    layout: str | None = None
    upscale: str | None = None


class ReviewBody(BaseModel):
    approved: list[int] = []
    hooks: dict[str, str] = {}
    lines: dict[str, list[list[str]]] = {}
    captions: dict[str, str] = {}
    censor: dict[str, dict[str, list[str]]] | None = None  # {n: {add: [], remove: []}}
    render: bool = False
    layout: str | None = None
    upscale: str | None = None


class CensorBody(BaseModel):
    n: int
    texts: list[str]
    add: list[str] = []
    remove: list[str] = []


class MetaBody(BaseModel):
    comedian: str
    show: str = ""


class RetryBody(BaseModel):
    auto_retry: bool | None = None


class RevealBody(BaseModel):
    path: str


class ShareBody(BaseModel):
    path: str
    platform: str
    done: bool


def create_app(queue=None, start_worker=True, open_url=None):
    queue = queue or JobQueue()

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            queue.start()
        if open_url:
            webbrowser.open(open_url)
        yield

    app = FastAPI(title="Standup Clipper", lifespan=lifespan)
    app.state.queue = queue

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).with_name("web.html")).read_text(encoding="utf-8")

    @app.get("/api/videos")
    def videos():
        return library(queue)

    @app.exception_handler(Exception)
    async def crash(request, e):
        """Beklenmeyen hata sunucuyu düşürmez: loglanır, arayüze Türkçe mesaj döner."""
        log.exception("İstek hatası: %s %s", request.method, request.url.path)
        return JSONResponse({"detail": f"Sunucu hatası: {e}"}, 500)

    @app.put("/api/videos/upload")
    async def upload(request: Request, comedian: str, show: str, ext: str, offset: int, total: int):
        """Parça parça yükleme: parçalar sırayla .part dosyasına eklenir, son parçada yerine taşınır."""
        name = upload_name(comedian, show, ext)
        if not 0 < total <= MAX_UPLOAD:
            raise HTTPException(413, "Dosya boş ya da 3 GB'tan büyük")
        dst = SAMPLES / name
        tmp = dst.with_suffix(dst.suffix + ".part")
        if dst.exists():
            raise HTTPException(409, f"{name} zaten var")
        data = await request.body()
        if offset == 0:
            SAMPLES.mkdir(exist_ok=True)
            tmp.write_bytes(b"")
        elif not tmp.exists() or tmp.stat().st_size != offset:
            raise HTTPException(409, "Yükleme sırası bozuldu, baştan deneyin")
        with open(tmp, "ab") as f:
            f.write(data)
        size = offset + len(data)
        if size > total:
            tmp.unlink()
            raise HTTPException(400, "Dosya beklenenden büyük geldi")
        if size < total:
            return {"received": size}
        tmp.replace(dst)
        save_meta(dst, comedian.strip())  # yazıldığı gibi (ör. "ISMO"), dosya adından türetilmez
        log.info("Video eklendi: %s (%d MB)", name, total >> 20)
        return {"name": name}

    @app.delete("/api/videos/{name}")
    def delete_video(name: str):
        """samples/ içindeki video ve work/ ara dosyaları silinir; out/ içindeki klipler korunur."""
        p = sample_path(name)
        job = queue.latest_for(name)
        if job and job["status"] in ("queued", "running"):
            raise HTTPException(409, "Video işlenirken silinemez")
        try:
            p.unlink()
            wd = WORK / slug(p.stem)  # work_dir() klasörü yeniden oluşturur, burada kullanılmaz
            if wd.exists():
                shutil.rmtree(wd)
        except PermissionError:
            raise HTTPException(409, "Dosya kullanımda (oynatıcıyı kapatıp tekrar deneyin)")
        log.info("Video silindi: %s", name)
        return {"ok": True}

    @app.get("/api/videos/{name}/file")
    def video_file(name: str):
        return FileResponse(sample_path(name))  # Range destekli: önizleme oynatıcı aralığa atlayabilir

    @app.post("/api/videos/{name}/start")
    def start(name: str, body: StartBody):
        sample_path(name)
        return queue.add(name, "process", body.model_dump())

    @app.post("/api/videos/{name}/render")
    def render(name: str, body: RenderBody):
        sample_path(name)
        return queue.add(name, "render", body.model_dump())

    @app.post("/api/videos/{name}/captions")
    def captions(name: str):
        """Açıklamaları yeniden yaz: yalnız .txt'ler güncellenir, video render edilmez."""
        wd = work_dir(sample_path(name))
        if not (wd / review_mod.TR_FILE).exists():
            raise HTTPException(409, "Bu video henüz işlenmedi")
        return queue.add(name, "captions")

    @app.post("/api/videos/{name}/meta")
    def meta(name: str, body: MetaBody):
        if not body.comedian.strip():
            raise HTTPException(400, "Komedyen adı gerekli")
        save_meta(sample_path(name), body.comedian.strip(), body.show.strip())
        return dict(zip(("comedian", "show"), video_meta(sample_path(name))))

    @app.get("/api/videos/{name}/review")
    def review_data(name: str):
        wd = work_dir(sample_path(name))
        if not (wd / review_mod.TR_FILE).exists():
            raise HTTPException(409, "Bu video henüz işlenmedi")
        subs = load_config()["subs"]
        return {**review_mod.load(wd), "max_chars": subs["max_chars"], "note_secs": subs["note_secs"]}

    @app.post("/api/videos/{name}/censor")
    def censor_preview(name: str, body: CensorBody):
        """Onay sayfası: metinlerin videoya basılacak (sansürlü) hâli + sansürlenen kelimeler."""
        wd = work_dir(sample_path(name))
        tr = read_json(wd / review_mod.TR_FILE).get(str(body.n), {}) if (wd / review_mod.TR_FILE).exists() else {}
        extra, allow = censor.overrides(tr, {"censor": {str(body.n): body.model_dump()}}, body.n)
        c = load_config()["censor"]
        return {"texts": [censor.apply(t, c, extra, allow) for t in body.texts],
                "words": censor.found(body.texts, c, extra, allow)}

    @app.post("/api/videos/{name}/review")
    def review_save(name: str, body: ReviewBody):
        wd = work_dir(sample_path(name))
        try:
            review_mod.save(wd, body.model_dump(include={"approved", "hooks", "lines", "captions", "censor"}, exclude_none=True))
        except ValueError as e:
            raise HTTPException(400, str(e))
        job = queue.add(name, "render", {"layout": body.layout, "upscale": body.upscale}) if body.render else None
        return {"ok": True, "job": job}

    @app.get("/api/jobs")
    def jobs():
        return list(reversed(queue.list()))

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = queue.get(job_id)
        if not j:
            raise HTTPException(404, "İş bulunamadı")
        p = queue.log_path(job_id)
        return {**j, "log": p.read_text(encoding="utf-8") if p.exists() else ""}

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: str, body: RetryBody):
        if not queue.get(job_id):
            raise HTTPException(404, "İş bulunamadı")
        return queue.retry(job_id, body.auto_retry)

    @app.get("/api/jobs/{job_id}/stream")
    async def stream(job_id: str, request: Request):
        """SSE: 'log' olayları (yeni satırlar) ve 'job' olayları (durum değişince). İş bitince kapanır."""
        if not queue.get(job_id):
            raise HTTPException(404, "İş bulunamadı")

        async def events():
            pos, last = 0, None
            while not await request.is_disconnected():
                p, chunk = queue.log_path(job_id), ""
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    for line in chunk.splitlines():
                        yield f"event: log\ndata: {json.dumps(line, ensure_ascii=False)}\n\n"
                j = queue.get(job_id)
                state = json.dumps(j, ensure_ascii=False)
                if state != last:
                    last = state
                    yield f"event: job\ndata: {state}\n\n"
                if j["status"] in FINAL and not chunk:  # bitti ve son log satırları da gönderildi
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/outputs")
    def outputs_list():
        return outputs()

    @app.get("/api/outputs/file")
    def output_file(path: str):
        return FileResponse(output_path(path))

    @app.post("/api/outputs/reveal")
    def reveal(body: RevealBody):
        p = output_path(body.path)
        subprocess.Popen(["explorer", "/select,", str(p)])
        return {"ok": True}

    @app.post("/api/outputs/share")
    def share(body: ShareBody):
        output_path(body.path)
        if body.platform not in PLATFORMS:
            raise HTTPException(400, "Bilinmeyen platform")
        shares = read_json(SHARES) if SHARES.exists() else {}
        entry = shares.setdefault(body.path, {})
        if body.done:
            entry[body.platform] = date.today().isoformat()
        else:
            entry.pop(body.platform, None)
        write_json(SHARES, shares)
        return entry

    @app.get("/api/settings")
    def settings():
        return {"values": read_settings(), "fields": {k: (v if isinstance(v, list) else v.__name__) for k, v in SETTINGS.items()}}

    @app.post("/api/settings")
    def settings_save(values: dict):
        write_settings(values)
        return read_settings()

    return app


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--open", action="store_true", help="Tarayıcıda aç")
    a = ap.parse_args()
    WORK.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(WORK / "server.log", encoding="utf-8")])
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # Windows: tarayıcı video bağlantısını kesince asyncio zararsız ConnectionResetError basar; loga yazılmaz
    logging.getLogger("asyncio").addFilter(lambda r: not (r.exc_info and isinstance(r.exc_info[1], ConnectionResetError)))
    app = create_app(open_url=f"http://127.0.0.1:{a.port}/" if a.open else None)
    try:
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_config=None)  # yalnızca yerel
    except BaseException:
        logging.exception("Sunucu durdu")
        raise


if __name__ == "__main__":
    main()

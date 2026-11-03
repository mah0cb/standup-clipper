"""9) Render: kes → (çekim bazlı crop | bulanık arka plan) → [Real-ESRGAN] → 1080x1920 → altyazı EN SON
→ out/{video}/{n}_{hook}.mp4 + .txt"""
import shutil
import subprocess
import time

from . import captions, reframe, select_llm, subtitles
from .common import ROOT, log, read_json, slug, video_info, work_dir

ESRGAN = ROOT / "models" / "realesrgan"
SCALE = "scale=1080:1920:flags=lanczos,setsar=1"


BLUR_BG = ("split[a][b];[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,gblur=sigma=12,"
           "scale=1080:1920[bg];[b]scale=1080:-2:flags=lanczos[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1")


def layout_filter(layout, box, dur=None):
    """Tek giriş/tek çıkışlı filtre zinciri (altyazı hariç). crop düzeninde "blur" işaretli çekimler (ara çekim,
    çok uzak komedyen) bulanık arka planlı sürümle örtülür."""
    if layout == "blur_bg":
        # Arka plan: küçült → bulanıklaştır → büyüt (doğrudan 1080x1920'de bulanıklaştırmaktan çok daha hızlı)
        return BLUR_BG
    crop = f"{crop_filter(box)},{SCALE}"
    cuts = reframe.blur_ranges(box, dur) if dur else []
    if not cuts:
        return crop
    on = "+".join(f"gte(t,{a})*lt(t,{b})" for a, b in cuts)  # tam çekim sınırında (bitiş hariç)
    return f"split[cm][bm];[cm]{crop}[cv];[bm]{BLUR_BG}[bv];[cv][bv]overlay=enable='{on}'"


def crop_filter(box):
    return f"crop={box['w']}:{box['h']}:x='{reframe.crop_x_expr(box['shots'])}':y={box['y']}"


def encode_args(cfg, encoder):
    r = cfg["render"]
    venc = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(r["cq"]), "-b:v", "0"]
            if encoder == "h264_nvenc" else ["-c:v", "libx264", "-preset", r["x264_preset"], "-crf", str(r["crf"])])
    return [*venc, "-pix_fmt", "yuv420p", "-af", r["audio_filter"], "-ar", "48000",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]


def ffmpeg(args, cfg):
    """Ayarlı kodlayıcıyla çalıştırır, başarısızsa libx264'e düşer."""
    enc = cfg["render"]["encoder"]
    for e in dict.fromkeys([enc, "libx264"]):
        p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args[:-1], *encode_args(cfg, e), args[-1]],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode == 0:
            return
        log.warning("%s başarısız: %s", e, p.stderr.strip()[-400:])
    raise RuntimeError(f"ffmpeg hata: {p.stderr.strip()[-1000:]}")


def should_upscale(mode, layout, crop_h):
    """auto: kırpılan alan 1920'den kısaysa. Yalnız crop düzeninde (blur_bg'de video zaten küçültülüyor)."""
    if mode == "off" or layout != "crop":
        return False
    return mode == "always" or crop_h < 1920


def upscale_frames(video, bit, box, fdir, cfg):
    """Kes + kırp → kareler → Real-ESRGAN (x4). Büyütülmüş karelerin klasörünü döndürür."""
    # ponytail: blur işaretli çekimler burada uygulanmaz (merkez kırpılır); gerekirse kareler üstünde overlay
    u = cfg["upscale"]
    src, dst = fdir / "in", fdir / "out"
    shutil.rmtree(fdir, ignore_errors=True)
    src.mkdir(parents=True)
    dst.mkdir()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{bit['start']:.3f}", "-to", f"{bit['end']:.3f}",
                    "-i", str(video), "-vf", crop_filter(box), "-fps_mode", "passthrough", "-q:v", "2",
                    str(src / "%06d.jpg")], check=True)
    subprocess.run([str(ESRGAN / "realesrgan-ncnn-vulkan.exe"), "-i", str(src), "-o", str(dst),
                    "-n", u["model"], "-s", "4", "-t", str(u["tile"]), "-f", "jpg", "-m", str(ESRGAN / "models")],
                   check=True, capture_output=True)
    return dst


def render_bit(video, bit, box, ass_rel, out, layout, mode, cfg):
    sub = f"subtitles='{ass_rel}':fontsdir=fonts"
    if should_upscale(mode, layout, box["h"]):
        fdir = work_dir(video) / "frames" / str(bit["n"])
        t = time.time()
        try:
            frames = upscale_frames(video, bit, box, fdir, cfg)
            log.info("Bit #%d Real-ESRGAN: %d kare, %.1f sn", bit["n"], len(list(frames.glob("*.jpg"))), time.time() - t)
            # Büyütülmüş kareler (orijinal fps) + orijinal ses; altyazı en son.
            # JPEG kareler tam renk aralığında (yuvj420p): standart yuv420p (tv aralığı) çıktıya çevrilir.
            ffmpeg(["-framerate", video_info(video)["fps"], "-i", str(frames / "%06d.jpg"),
                    "-ss", f"{bit['start']:.3f}", "-to", f"{bit['end']:.3f}", "-i", str(video),
                    "-map", "0:v", "-map", "1:a:0", "-shortest",
                    "-vf", f"scale=1080:1920:flags=lanczos:in_range=pc:out_range=tv,setsar=1,{sub}",
                    "-color_range", "tv", str(out)], cfg)
        finally:
            if not cfg["upscale"]["keep_frames"]:
                shutil.rmtree(fdir, ignore_errors=True)
        return time.time() - t
    ffmpeg(["-ss", f"{bit['start']:.3f}", "-to", f"{bit['end']:.3f}", "-i", str(video),
            "-vf", f"{layout_filter(layout, box, bit['end'] - bit['start'])},{sub}", str(out)], cfg)
    return 0.0


def run(video, cfg, force=True, layout=None, upscale=None, only=None, out_dir=None, suffix=""):
    layout = layout or cfg["render"]["layout"]
    mode = upscale or cfg["upscale"]["mode"]
    wd = work_dir(video)
    review = read_json(wd / "review.json") if (wd / "review.json").exists() else {}
    approved = set(only or review.get("approved", []))
    if not approved:
        log.warning("Onaylanmış klip yok. Önce `python main.py review` ile onay sayfasında klip seçin.")
        return []
    bits = [b for b in select_llm.run(video, cfg) if b["n"] in approved]  # yalnız onaylılar işlenir
    boxes = reframe.run(video, cfg)
    subs = subtitles.run(video, cfg, layout=layout)
    caps = captions.run(video, cfg)
    od = out_dir or ROOT / "out" / wd.name
    od.mkdir(parents=True, exist_ok=True)
    outs = []
    for b in bits:
        if "end" in boxes[str(b["n"])]:  # kadraj: son çekimde komedyen yok → klip önceki çekimin sonunda biter
            b = {**b, "end": b["start"] + boxes[str(b["n"])]["end"]}
        out = od / f"{b['n']}_{slug(b['hook_tr'], 40)}{suffix}.mp4"
        up = render_bit(video, b, boxes[str(b["n"])], subs[b["n"]].relative_to(ROOT).as_posix(), out, layout, mode, cfg)
        out.with_suffix(".txt").write_text(captions.txt_text(wd.name, b["n"], caps.get(str(b["n"]), ""), cfg),
                                           encoding="utf-8")
        log.info("Render edildi (%s%s): %s", layout, ", upscale %.0f sn" % up if up else "", out)
        outs.append(out)
    return outs

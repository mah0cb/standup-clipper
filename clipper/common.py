"""Ortak yardımcılar: config, çalışma klasörü, JSON cache."""
import functools
import json
import logging
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("clipper")


def load_config(path=ROOT / "config.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _title(s):
    # Komedyenler çoğunlukla yabancı: standart büyük harf (i → I); dosya adında "İ" yazılıysa korunur
    return " ".join(w[0].upper() + w[1:] for w in s.replace("_", " ").split())


def comedian_from_filename(path):
    """"zoltan_kaszas__london_fog.mp4" → "Zoltan Kaszas" (çift alt çizgiden önceki kısım)."""
    return _title(Path(path).stem.split("__")[0])


def show_from_filename(path):
    """"zoltan_kaszas__london_fog.mp4" → "London Fog"; çift alt çizgi yoksa ""."""
    parts = Path(path).stem.split("__")
    return _title(parts[1]) if len(parts) > 1 else ""


def save_meta(video, comedian="", show=None):
    meta_p = work_dir(video) / "meta.json"
    meta = read_json(meta_p) if meta_p.exists() else {}
    if comedian:
        meta["comedian"] = comedian
    if show is not None:
        meta["show"] = show
    write_json(meta_p, {**meta, "video": str(Path(video).resolve())})


def video_meta(video):
    """(komedyen, gösteri): meta.json'daki değerler, yoksa dosya adından."""
    p = work_dir(video) / "meta.json"
    meta = read_json(p) if p.exists() else {}
    show = meta["show"] if "show" in meta else show_from_filename(video)
    return meta.get("comedian") or comedian_from_filename(video), show


def slug(text, maxlen=60):
    # Dosya ve ffmpeg filtre yollarında sorun çıkarmayan ad (Türkçe harfler korunur)
    return re.sub(r"[^\w\-]+", "_", text).strip("_")[:maxlen] or "video"


def work_dir(video):
    d = ROOT / "work" / slug(Path(video).stem)
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cached(path, fn, force=False):
    """JSON çıktısı varsa onu döndürür, yoksa fn() çalıştırıp kaydeder."""
    path = Path(path)
    if path.exists() and not force:
        return read_json(path)
    data = fn()
    write_json(path, data)
    return data


def words_in(words, start, end):
    return [w for w in words if w["start"] >= start - 0.05 and w["end"] <= end + 0.05]


@functools.lru_cache
def video_info(video):
    """Video genişliği, yüksekliği ve fps'i ("24000/1001" biçiminde, ffmpeg'e olduğu gibi verilir)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, check=True).stdout.split()[0].split(",")
    return {"w": int(out[0]), "h": int(out[1]), "fps": out[2]}

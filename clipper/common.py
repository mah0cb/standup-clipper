"""Ortak yardımcılar: config, çalışma klasörü, JSON cache."""
import json
import logging
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("clipper")


def load_config(path=ROOT / "config.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


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

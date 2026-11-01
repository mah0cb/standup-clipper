"""Adım listesi ve ortak çalıştırıcılar (CLI ve web arayüzü aynı fonksiyonları kullanır)."""
import logging
import time
from pathlib import Path

from . import (audio, candidates, gapfill, laughter, proofread, reframe, render, review, select_llm, subtitles,
               transcribe, translate)
from .common import read_json, work_dir, write_json

STEPS = {
    "audio": audio, "transcribe": transcribe, "laughter": laughter, "candidates": candidates,
    "select": select_llm, "gapfill": gapfill, "translate": translate, "proofread": proofread,
    "subtitles": subtitles, "reframe": reframe, "render": render, "review": review,
}
RUN_STEPS = ["audio", "transcribe", "laughter", "candidates", "select", "gapfill", "translate", "proofread",
             "subtitles", "reframe"]
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".m4v", ".webm"}


def timed(name, video, cfg, force=False, **kw):
    t = time.time()
    STEPS[name].run(video, cfg, force=force, **kw)
    dt = round(time.time() - t, 1)
    p = work_dir(video) / "timings.json"
    tm = read_json(p) if p.exists() else {}
    tm[name] = dt
    write_json(p, tm)
    logging.info("✔ %-11s %6.1f sn", name, dt)
    return dt


def comedian_from_filename(path):
    """"zoltan_kaszas__london_fog.mp4" → "Zoltan Kaszas" (çift alt çizgiden önceki kısım)."""
    name = Path(path).stem.split("__")[0].replace("_", " ").strip()
    # Komedyenler çoğunlukla yabancı: standart büyük harf (i → I); dosya adında "İ" yazılıysa korunur
    return " ".join(w[0].upper() + w[1:] for w in name.split())


def save_meta(video, comedian=""):
    meta_p = work_dir(video) / "meta.json"
    meta = read_json(meta_p) if meta_p.exists() else {}
    if comedian:
        meta["comedian"] = comedian
    write_json(meta_p, {**meta, "video": str(Path(video).resolve())})


def process(video, cfg, comedian="", on_step=None):
    """1–8. adımlar (onay sayfası hariç). on_step(ad, süre|None): ilerleme bildirimi (web arayüzü)."""
    save_meta(video, comedian)
    for name in RUN_STEPS:
        if on_step:
            on_step(name, None)
        dt = timed(name, video, cfg)
        if on_step:
            on_step(name, dt)

"""Adım listesi ve ortak çalıştırıcılar (CLI ve web arayüzü aynı fonksiyonları kullanır)."""
import logging
import time

from . import (audio, candidates, captions, gapfill, laughter, proofread, reframe, render, review, select_llm,
               subtitles, transcribe, translate)
from .common import comedian_from_filename, read_json, save_meta, show_from_filename, work_dir, write_json  # noqa: F401

STEPS = {
    "audio": audio, "transcribe": transcribe, "laughter": laughter, "candidates": candidates,
    "select": select_llm, "gapfill": gapfill, "translate": translate, "proofread": proofread,
    "captions": captions, "subtitles": subtitles, "reframe": reframe, "render": render, "review": review,
}
RUN_STEPS = ["audio", "transcribe", "laughter", "candidates", "select", "gapfill", "translate", "proofread",
             "captions", "subtitles", "reframe"]
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



def process(video, cfg, comedian="", on_step=None):
    """1–8. adımlar (onay sayfası hariç). on_step(ad, süre|None): ilerleme bildirimi (web arayüzü)."""
    save_meta(video, comedian)
    for name in RUN_STEPS:
        if on_step:
            on_step(name, None)
        dt = timed(name, video, cfg)
        if on_step:
            on_step(name, dt)

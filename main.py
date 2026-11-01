"""standup-clipper komut satırı.

  python main.py run video.mp4 --comedian "İsim"   → 1–8. adımlar + onay sayfası
  python main.py render video.mp4                   → onaylanan klipleri render eder
  python main.py step <adim> video.mp4              → tek adımı yeniden çalıştırır
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from clipper import (audio, candidates, laughter, proofread, reframe, render, review, select_llm,
                     subtitles, transcribe, translate)
from clipper.common import ROOT, load_config, read_json, work_dir, write_json

STEPS = {
    "audio": audio, "transcribe": transcribe, "laughter": laughter, "candidates": candidates,
    "select": select_llm, "translate": translate, "proofread": proofread, "subtitles": subtitles, "reframe": reframe,
    "render": render, "review": review,
}
RUN_STEPS = ["audio", "transcribe", "laughter", "candidates", "select", "translate", "proofread", "subtitles", "reframe"]


def timed(name, video, cfg, force=False):
    t = time.time()
    STEPS[name].run(video, cfg, force=force)
    dt = round(time.time() - t, 1)
    p = work_dir(video) / "timings.json"
    tm = read_json(p) if p.exists() else {}
    tm[name] = dt
    write_json(p, tm)
    logging.info("✔ %-11s %6.1f sn", name, dt)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(ROOT / "clipper.log", encoding="utf-8")])
    ap = argparse.ArgumentParser(description="Stand-up videosundan Türkçe altyazılı dikey kısa videolar")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="1–8. adımlar + onay sayfası")
    r.add_argument("video")
    r.add_argument("--comedian", default="", help="Altta görünecek komedyen adı")
    r.add_argument("--no-review", action="store_true", help="Onay sayfasını açma")
    sub.add_parser("render", help="Onaylanan klipleri render et").add_argument("video")
    s = sub.add_parser("step", help="Tek adımı yeniden çalıştır")
    s.add_argument("name", choices=list(STEPS))
    s.add_argument("video")
    a = ap.parse_args()

    if not Path(a.video).exists():
        ap.error(f"Video bulunamadı: {a.video}")
    cfg = load_config()
    wd = work_dir(a.video)

    if a.cmd == "run":
        meta_p = wd / "meta.json"
        meta = read_json(meta_p) if meta_p.exists() else {}
        if a.comedian:
            meta["comedian"] = a.comedian
        write_json(meta_p, {**meta, "video": str(Path(a.video).resolve())})
        for name in RUN_STEPS:
            timed(name, a.video, cfg)
        if not a.no_review:
            review.run(a.video, cfg)
    elif a.cmd == "render":
        timed("render", a.video, cfg)
    else:
        timed(a.name, a.video, cfg, force=True)


if __name__ == "__main__":
    main()

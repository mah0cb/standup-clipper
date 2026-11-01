"""standup-clipper komut satırı.

  python main.py run video.mp4 --comedian "İsim"   → 1–8. adımlar + onay sayfası
  python main.py run-all samples\                  → klasördeki tüm videolar (onay sayfası açmadan)
  python main.py review video.mp4                   → onay sayfasını açar
  python main.py render video.mp4                   → onaylanan klipleri render eder
  python main.py step <adim> video.mp4              → tek adımı yeniden çalıştırır
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from clipper import (audio, candidates, gapfill, laughter, proofread, reframe, render, review, select_llm,
                     subtitles, transcribe, translate)
from clipper.common import ROOT, load_config, read_json, work_dir, write_json

STEPS = {
    "audio": audio, "transcribe": transcribe, "laughter": laughter, "candidates": candidates,
    "select": select_llm, "gapfill": gapfill, "translate": translate, "proofread": proofread, "subtitles": subtitles, "reframe": reframe,
    "render": render, "review": review,
}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".m4v", ".webm"}
RUN_STEPS = ["audio", "transcribe", "laughter", "candidates", "select", "gapfill", "translate", "proofread", "subtitles", "reframe"]


def timed(name, video, cfg, force=False, **kw):
    t = time.time()
    STEPS[name].run(video, cfg, force=force, **kw)
    dt = round(time.time() - t, 1)
    p = work_dir(video) / "timings.json"
    tm = read_json(p) if p.exists() else {}
    tm[name] = dt
    write_json(p, tm)
    logging.info("✔ %-11s %6.1f sn", name, dt)


def comedian_from_filename(path):
    """"zoltan_kaszas__london_fog.mp4" → "Zoltan Kaszas" (çift alt çizgiden önceki kısım)."""
    name = Path(path).stem.split("__")[0].replace("_", " ").strip()
    # Komedyenler çoğunlukla yabancı: standart büyük harf (i → I); dosya adında "İ" yazılıysa korunur
    return " ".join(w[0].upper() + w[1:] for w in name.split())


def process(video, cfg, comedian=""):
    """1–8. adımlar (onay sayfası hariç)."""
    meta_p = work_dir(video) / "meta.json"
    meta = read_json(meta_p) if meta_p.exists() else {}
    if comedian:
        meta["comedian"] = comedian
    write_json(meta_p, {**meta, "video": str(Path(video).resolve())})
    for name in RUN_STEPS:
        timed(name, video, cfg)


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
    rd = sub.add_parser("render", help="Onaylanan klipleri render et")
    rd.add_argument("video")
    rd.add_argument("--layout", choices=["crop", "blur_bg"], help="Düzen (varsayılan: config render.layout)")
    rd.add_argument("--upscale", choices=["auto", "always", "off"], help="Real-ESRGAN (varsayılan: config upscale.mode)")
    ra = sub.add_parser("run-all", help="Klasördeki tüm videoları sırayla işle (onay sayfası açmadan)")
    ra.add_argument("folder")
    sub.add_parser("review", help="Onay sayfasını aç").add_argument("video")
    s = sub.add_parser("step", help="Tek adımı yeniden çalıştır")
    s.add_argument("name", choices=list(STEPS))
    s.add_argument("video")
    a = ap.parse_args()

    cfg = load_config()
    if a.cmd == "run-all":
        videos = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in VIDEO_EXT)
        if not videos:
            ap.error(f"Klasörde video yok: {a.folder}")
        done, failed = [], []
        for i, v in enumerate(videos, 1):
            name = comedian_from_filename(v)
            logging.info("=== [%d/%d] %s (komedyen: %s)", i, len(videos), v.name, name)
            try:
                process(v, cfg, name)
                done.append(v)
            except Exception:  # bir video bozuksa diğerleri yine işlensin
                logging.exception("%s işlenemedi, sonrakine geçiliyor", v.name)
                failed.append(v)
        logging.info("Bitti: %d başarılı, %d hatalı. Onay için: python main.py review <video>", len(done), len(failed))
        for v in done:
            logging.info("  python main.py review %s", v)
        return

    if not Path(a.video).exists():
        ap.error(f"Video bulunamadı: {a.video}")

    if a.cmd == "run":
        process(a.video, cfg, a.comedian)
        if not a.no_review:
            review.run(a.video, cfg)
    elif a.cmd == "review":
        review.run(a.video, cfg)
    elif a.cmd == "render":
        timed("render", a.video, cfg, layout=a.layout, upscale=a.upscale)
    else:
        timed(a.name, a.video, cfg, force=True)


if __name__ == "__main__":
    main()

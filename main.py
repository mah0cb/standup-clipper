"""standup-clipper komut satırı.

  python main.py run video.mp4 --comedian "İsim"   → 1–8. adımlar + onay sayfası
  python main.py run-all samples\                  → klasördeki tüm videolar (onay sayfası açmadan)
  python main.py review video.mp4                   → onay sayfasını açar
  python main.py render video.mp4                   → onaylanan klipleri render eder
  python main.py captions video.mp4                 → açıklamaları yeniden yazar (render yok)
  python main.py step <adim> video.mp4              → tek adımı yeniden çalıştırır
"""
import argparse
import logging
import sys
from pathlib import Path

from clipper import review
from clipper.common import ROOT, load_config
from clipper.pipeline import STEPS, VIDEO_EXT, comedian_from_filename, process, timed


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
    sub.add_parser("captions", help="Açıklamaları yeniden yaz (.txt'ler güncellenir, render yok)").add_argument("video")
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
    elif a.cmd == "captions":
        timed("captions", a.video, cfg, force=True)
    else:
        timed(a.name, a.video, cfg, force=True)


if __name__ == "__main__":
    main()

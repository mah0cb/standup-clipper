"""9) ffmpeg ile kes + crop + altyazı yak + loudnorm → out/{video}/{n}_{hook}.mp4 + .txt"""
import subprocess

from . import reframe, select_llm, subtitles
from .common import ROOT, log, read_json, slug, work_dir


def ffmpeg_cmd(video, bit, box, ass_rel, out, cfg, encoder):
    r = cfg["render"]
    vf = (f"crop={box['w']}:{box['h']}:{box['x']}:{box['y']},scale=1080:1920:flags=lanczos,setsar=1,"
          f"subtitles='{ass_rel}':fontsdir=fonts")
    venc = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(r["cq"]), "-b:v", "0"]
            if encoder == "h264_nvenc" else ["-c:v", "libx264", "-preset", "medium", "-crf", str(r["cq"])])
    return ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{bit['start']:.2f}", "-to", f"{bit['end']:.2f}",
            "-i", str(video), "-vf", vf, *venc, "-pix_fmt", "yuv420p",
            "-af", r["audio_filter"], "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(out)]


def run(video, cfg, force=True):
    wd = work_dir(video)
    review = read_json(wd / "review.json") if (wd / "review.json").exists() else {}
    approved = set(review.get("approved", []))
    if not approved:
        log.warning("Onaylanmış klip yok. Önce `python main.py run` ile onay sayfasında klip seçin.")
        return []
    meta = read_json(wd / "meta.json") if (wd / "meta.json").exists() else {}
    bits = [b for b in select_llm.run(video, cfg) if b["n"] in approved]
    boxes = reframe.run(video, cfg)
    subs = subtitles.run(video, cfg)
    od = ROOT / "out" / wd.name
    od.mkdir(parents=True, exist_ok=True)
    outs = []
    for b in bits:
        out = od / f"{b['n']}_{slug(b['hook_tr'], 40)}.mp4"
        ass_rel = subs[b["n"]].relative_to(ROOT).as_posix()  # cwd=ROOT: Windows sürücü yolu kaçışından kaçınılır
        enc = cfg["render"]["encoder"]
        p = subprocess.run(ffmpeg_cmd(video, b, boxes[str(b["n"])], ass_rel, out, cfg, enc),
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0 and enc != "libx264":
            log.warning("%s başarısız (%s), libx264 ile tekrar", enc, p.stderr.strip()[-300:])
            p = subprocess.run(ffmpeg_cmd(video, b, boxes[str(b["n"])], ass_rel, out, cfg, "libx264"),
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg hata (bit #{b['n']}): {p.stderr.strip()[-1000:]}")
        comedian = meta.get("comedian", "")
        out.with_suffix(".txt").write_text(
            f"{b['description_tr']}\n\n{' '.join(b['hashtags'])}\n\n{comedian}\n", encoding="utf-8")
        log.info("Render edildi: %s", out)
        outs.append(out)
    return outs

"""1) Videodan 16 kHz mono wav çıkarır."""
import subprocess

from .common import work_dir


def run(video, cfg, force=False):
    out = work_dir(video) / "audio.wav"
    if force or not out.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                        "-vn", "-ac", "1", "-ar", "16000", str(out)], check=True)
    return out

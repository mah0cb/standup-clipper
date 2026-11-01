"""2) faster-whisper ile kelime zaman damgalı İngilizce transkript → words.json + segments.json."""
import os
import sys
from pathlib import Path

from . import audio
from .common import log, read_json, work_dir, write_json


def _add_cuda_dlls():
    # pip ile gelen cuBLAS/cuDNN DLL'lerini Windows'ta bulunur yap
    for d in (Path(sys.prefix) / "Lib" / "site-packages" / "nvidia").glob("*/bin"):
        os.add_dll_directory(str(d))
        os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


def _transcribe(wav, model_name, device, compute_type, w):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segs, _ = model.transcribe(str(wav), language=w["language"], word_timestamps=True, vad_filter=True,
                               initial_prompt=w.get("initial_prompt"))
    segments, words = [], []
    for s in segs:  # üreteç: asıl iş burada yapılır
        segments.append({"id": len(segments), "start": round(float(s.start), 2), "end": round(float(s.end), 2),
                         "text": s.text.strip()})
        for w in s.words or []:
            words.append({"i": len(words), "start": round(float(w.start), 2), "end": round(float(w.end), 2),
                          "word": w.word.strip(), "seg": segments[-1]["id"]})
    return segments, words


def run(video, cfg, force=False):
    wd = work_dir(video)
    wp, sp = wd / "words.json", wd / "segments.json"
    if wp.exists() and sp.exists() and not force:
        return read_json(sp), read_json(wp)
    wav = audio.run(video, cfg)
    w = cfg["whisper"]
    try:
        if w["device"] == "cuda":
            _add_cuda_dlls()
        segments, words = _transcribe(wav, w["model"], w["device"], w["compute_type"], w)
    except Exception as e:  # CUDA/cuDNN sorunu → CPU
        log.warning("GPU transkripsiyon başarısız (%s), CPU + %s + %s'a geçiliyor", e,
                    w["fallback_compute_type"], w["fallback_model"])
        segments, words = _transcribe(wav, w["fallback_model"], "cpu", w["fallback_compute_type"], w)
    write_json(sp, segments)
    write_json(wp, words)
    return segments, words

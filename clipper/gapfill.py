"""5b) Kaçan konuşmayı kurtarma: seçilen bitlerde kelimeler arası uzun boşlukları VAD kapalı yeniden
transkribe eder (Whisper bazen kahkaha altındaki konuşmayı atlar) → gapfill.json {n: [kelime]}."""
from . import audio, select_llm, transcribe
from .common import cached, log, work_dir


def find_gaps(words, start, end, min_gap):
    """Klip aralığında (başı/sonu dahil) min_gap'ten uzun, kelimesiz aralıklar."""
    times = [start] + [t for w in words for t in (w["start"], w["end"])] + [end]
    return [(a, b) for a, b in zip(times[0::2], times[1::2]) if b - a >= min_gap]


def keep_word(w, seg, g):
    """Sessizlikte uydurulan metni ele: düşük olasılık / konuşma yok olasılığı yüksek."""
    return w.probability >= g["min_word_prob"] and seg.no_speech_prob <= g["max_no_speech"]


def run(video, cfg, force=False):
    def make():
        _, words = transcribe.run(video, cfg)
        bits = select_llm.run(video, cfg)
        g, wcfg = cfg["gapfill"], cfg["whisper"]
        from faster_whisper import WhisperModel, decode_audio

        try:
            transcribe._add_cuda_dlls()
            model = WhisperModel(wcfg["model"], device=wcfg["device"], compute_type=wcfg["compute_type"])
        except Exception as e:  # CUDA sorunu → CPU
            log.warning("gapfill GPU açılamadı (%s), CPU'ya geçiliyor", e)
            model = WhisperModel(wcfg["fallback_model"], device="cpu", compute_type=wcfg["fallback_compute_type"])
        pcm = decode_audio(str(audio.run(video, cfg)), sampling_rate=16000)
        out = {}
        for b in bits:
            bw = words[b["start_word"]:b["end_word"] + 1]
            found = []
            for a, e in find_gaps(bw, b["start"], b["end"], g["min_gap"]):
                segs, _ = model.transcribe(pcm[int(a * 16000):int(e * 16000)], language=wcfg["language"],
                                           vad_filter=False, word_timestamps=True, condition_on_previous_text=False)
                for s in segs:
                    for w in s.words or []:
                        if keep_word(w, s, g) and a + w.end <= e + 0.05:
                            found.append({"start": round(a + float(w.start), 2), "end": round(a + float(w.end), 2),
                                          "word": w.word.strip(), "recovered": True})
            if found:
                log.info("Bit #%d: %d kaçan kelime kurtarıldı: %s", b["n"], len(found), " ".join(w["word"] for w in found))
            out[str(b["n"])] = found
        return out
    return cached(work_dir(video) / "gapfill.json", make, force)

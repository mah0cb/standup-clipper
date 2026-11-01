"""3) Seyirci kahkahası tespiti → laughter.json [{start, end, prob}].

Önce jrgillick/laughter-detection (ResNet, models/laughter-detection). Kurulum veya
yükleme hatası olursa MediaPipe üzerinden YAMNet'in kahkaha sınıflarına geçilir.
"""
import importlib.util

import numpy as np

from . import audio
from .common import ROOT, cached, log, work_dir

JR_DIR = ROOT / "models" / "laughter-detection"
YAMNET_CLASSES = ["Laughter", "Giggle", "Belly laugh", "Chuckle, chortle", "Snicker"]


def segments_from_probs(times, probs, step, thr, min_dur, merge_gap):
    """Olasılık serisini [{start,end,prob}] aralıklarına çevirir. Saf fonksiyon (test edilir)."""
    segs, cur = [], None
    for t, p in zip(times, probs):
        if p > thr:
            if cur and t - cur["end"] <= merge_gap:
                cur["end"], cur["ps"] = t + step, cur["ps"] + [p]
            else:
                cur = {"start": t, "end": t + step, "ps": [p]}
                segs.append(cur)
    return [{"start": round(s["start"], 2), "end": round(s["end"], 2), "prob": round(float(np.mean(s["ps"])), 3)}
            for s in segs if s["end"] - s["start"] >= min_dur]


def _jrgillick(wav, stride):
    import librosa
    import torch
    from scipy import signal

    spec = importlib.util.spec_from_file_location("jr_models", JR_DIR / "models.py")
    jr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jr)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = jr.ResNetBigger(dropout_rate=0.0, linear_layer_size=128, filter_sizes=[128, 64, 32, 32])
    m.set_device(dev)
    # Eski (torch 1.x) checkpoint: numpy skalerleri içerdiği için weights_only=False gerekli.
    # Dosya resmi repodan klonlanmıştır.
    ck = torch.load(JR_DIR / "checkpoints/in_use/resnet_with_augmentation/best.pth.tar",
                    map_location=dev, weights_only=False)
    m.load_state_dict(ck["state_dict"])
    m.eval()

    sr, hop, n = 8000, 186, 44  # orijinal resnet_with_augmentation ayarları
    y, _ = librosa.load(str(wav), sr=sr)
    # Orijinal featurize_melspec'in yeni librosa API'siyle karşılığı
    feat = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop).T, ref=np.max)
    X = np.lib.stride_tricks.sliding_window_view(feat, n, axis=0).transpose(0, 2, 1)[::stride, None]
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            probs.append(m(torch.from_numpy(X[i:i + 256].copy()).float().to(dev)).cpu().numpy().reshape(-1))
    probs = np.concatenate(probs)
    step = stride * hop / sr
    if len(probs) > 9:  # orijinaldeki alçak geçiren filtre, stride'a göre ölçekli
        B, A = signal.butter(2, min(0.01 * stride, 0.5))
        probs = signal.filtfilt(B, A, probs)
    return np.arange(len(probs)) * step, probs, step


def _yamnet(wav):
    import librosa
    from mediapipe.tasks.python import audio as mp_audio
    from mediapipe.tasks.python.components.containers import AudioData
    from mediapipe.tasks.python.core.base_options import BaseOptions

    y, sr = librosa.load(str(wav), sr=16000)
    opts = mp_audio.AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "yamnet.tflite")),
        max_results=-1, running_mode=mp_audio.RunningMode.AUDIO_CLIPS)
    times, probs = [], []
    with mp_audio.AudioClassifier.create_from_options(opts) as clf:
        for r in clf.classify(AudioData.create_from_array(y.astype(np.float32), sr)):
            cats = {c.category_name: c.score for c in r.classifications[0].categories}
            times.append(r.timestamp_ms / 1000)
            probs.append(max(cats.get(k, 0.0) for k in YAMNET_CLASSES))
    step = times[1] - times[0] if len(times) > 1 else 0.975
    return np.array(times), np.array(probs), step


def detect(wav, c):
    backend = c["backend"]
    if backend in ("auto", "jrgillick"):
        try:
            times, probs, step = _jrgillick(wav, c["stride"])
            log.info("Kahkaha modeli: jrgillick/laughter-detection (ResNet)")
        except Exception as e:
            if backend == "jrgillick":
                raise
            log.warning("jrgillick yüklenemedi (%s) → YAMNet'e geçiliyor", e)
            times, probs, step = _yamnet(wav)
    else:
        times, probs, step = _yamnet(wav)
        log.info("Kahkaha modeli: YAMNet")
    return segments_from_probs(times, probs, step, c["threshold"], c["min_dur"], c["merge_gap"])


def run(video, cfg, force=False):
    return cached(work_dir(video) / "laughter.json", lambda: detect(audio.run(video, cfg), cfg["laughter"]), force)

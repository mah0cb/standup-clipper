"""8) Çekim (kamera açısı) bazlı 9:16 crop: PySceneDetect ile kesmeler, her çekimde MediaPipe yüz merkezi → reframe.json."""
import statistics

import cv2
import numpy as np

from . import select_llm
from .common import ROOT, cached, log, work_dir


def smooth(xs, window):
    """Hareketli ortalama (kenarlarda pencere küçülür)."""
    if not xs:
        return []
    k = max(1, min(window, len(xs)))
    pad = np.pad(np.asarray(xs, float), (k // 2, k - 1 - k // 2), mode="edge")
    return list(np.convolve(pad, np.ones(k) / k, mode="valid"))


def crop_box(centers, W, H, window):
    """Yüz merkezlerinden (piksel, x) crop kutusu. Yüz yoksa merkez crop."""
    cw = min(W, int(H * 9 / 16)) // 2 * 2
    ch = min(H, int(cw * 16 / 9)) // 2 * 2
    # ponytail: çekim başına tek (medyan) x; çekim içinde komedyen çok gezerse kare kare takip eklenebilir
    cx = statistics.median(smooth(centers, window)) if centers else W / 2
    x = int(min(max(cx - cw / 2, 0), W - cw)) // 2 * 2
    return {"x": x, "y": (H - ch) // 2 // 2 * 2, "w": cw, "h": ch}


def pick_face(faces, H, max_y):
    """Seyirci kafalarını (karenin alt kısmı) ele, en güvenilir yüzün x merkezini döndür."""
    ok = [d for d in faces if (d.bounding_box.origin_y + d.bounding_box.height / 2) / H <= max_y]
    if not ok:
        return None
    b = max(ok, key=lambda d: d.categories[0].score).bounding_box
    return b.origin_x + b.width / 2


def face_centers(video, start, end, c):
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    # Önce yakın çekim modeli; yüz bulamazsa geniş sahne çekimleri için full range
    dets = [vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / f"{m}.tflite")),
        min_detection_confidence=c["min_confidence"])) for m in c["models"]]
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    centers, checked, i = [], 0, 0
    while cap.get(cv2.CAP_PROP_POS_MSEC) <= end * 1000 and cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - start
        if i % c["every_n"] == 0:
            _, frame = cap.retrieve()
            img = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            checked += 1
            for det in dets:
                x = pick_face(det.detect(img).detections, H, c["max_face_y"])
                if x is not None:
                    centers.append((t, x))
                    break
        i += 1
    cap.release()
    for det in dets:
        det.close()
    return centers, checked, W, H


def detect_cuts(video, start, end, c):
    """Klip içindeki kamera açısı değişimleri (klip başına göre sn)."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video), ContentDetector(threshold=c["scene_threshold"]), start_time=start, end_time=end)
    return [round(a.seconds - start, 3) for a, _ in scenes[1:]]


def shots_from(cuts, samples, dur, W, H, c):
    """Kesmelerden çekimler; her çekimin crop x'i o çekimdeki yüzlerden. Çok kısa çekimler öncekiyle birleşir.
    Yüz bulunamayan çekim önceki çekimin konumunu kullanır (ilkse merkez)."""
    bounds = [0.0]
    for t in cuts:
        if t - bounds[-1] >= c["min_shot"] and dur - t >= c["min_shot"]:
            bounds.append(t)
    bounds.append(dur)
    shots, prev = [], None
    for a, b in zip(bounds, bounds[1:]):
        xs = [x for t, x in samples if a <= t < b]
        x = crop_box(xs, W, H, c["smooth_window"])["x"] if xs else prev
        shots.append({"t": round(a, 3), "x": x, "faces": len(xs)})
        prev = x
    center = crop_box([], W, H, 1)["x"]
    for s in shots:  # baştaki yüzsüz çekimler: ilk bulunan konum, hiç yoksa merkez
        if s["x"] is None:
            s["x"] = next((o["x"] for o in shots if o["x"] is not None), center)
    return shots


def crop_x_expr(shots):
    """ffmpeg crop x ifadesi: çekim değiştikçe x değişir (t, klip başından itibaren sn)."""
    expr = str(shots[-1]["x"])
    for cur, nxt in zip(reversed(shots[:-1]), reversed(shots[1:])):
        expr = f"if(lt(t,{nxt['t']}),{cur['x']},{expr})"
    return expr


def run(video, cfg, force=False):
    def make():
        c, out = cfg["reframe"], {}
        for b in select_llm.run(video, cfg):
            samples, checked, W, H = face_centers(video, b["start"], b["end"], c)
            cuts = detect_cuts(video, b["start"], b["end"], c)
            shots = shots_from(cuts, samples, b["end"] - b["start"], W, H, c)
            box = crop_box([x for _, x in samples], W, H, c["smooth_window"])
            log.info("Bit #%d: %d/%d karede yüz, %d çekim → crop x %s", b["n"], len(samples), checked,
                     len(shots), [s["x"] for s in shots])
            out[str(b["n"])] = {**box, "shots": shots, "src_w": W, "src_h": H, "faces": len(samples), "checked": checked}
        return out
    return cached(work_dir(video) / "reframe.json", make, force)

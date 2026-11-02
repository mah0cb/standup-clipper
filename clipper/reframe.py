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


def pick_pose(poses, W, max_y):
    """Yüz bulunamazsa yedek: kişinin gövde merkezi (omuz + kalça). Burnu karenin alt kısmında olan
    (seyirci) elenir; birden çok kişi varsa omuzları en geniş olan (sahnedeki, kameraya en yakın) seçilir."""
    best = None
    for lm in poses:
        if lm[0].y > max_y:
            continue
        pts = [lm[i] for i in (11, 12, 23, 24) if lm[i].visibility > 0.5] or [lm[11], lm[12]]
        width = abs(lm[11].x - lm[12].x)
        if best is None or width > best[0]:
            best = (width, sum(p.x for p in pts) / len(pts) * W)
    return best and best[1]


def face_centers(video, start, end, c):
    """Örneklenen karelerde komedyenin x merkezi: [(t, x)], yüz bulunamayan karelerde pose (gövde) yedeği."""
    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    # Önce yakın çekim modeli; yüz bulamazsa geniş sahne çekimleri için full range
    dets = [vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "models" / f"{m}.tflite")),
        min_detection_confidence=c["min_confidence"])) for m in c["models"]]
    pose_p = ROOT / "models" / f"{c['pose_model']}.task" if c.get("pose_model") else None
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_p)), num_poses=3,
        min_pose_detection_confidence=c["min_confidence"])) if pose_p and pose_p.exists() else None
    if pose_p and not pose:
        log.warning("Pose modeli yok (%s); kişi tespiti yedeği kapalı", pose_p)
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    centers, checked, by_pose, i = [], 0, 0, 0
    while cap.get(cv2.CAP_PROP_POS_MSEC) <= end * 1000 and cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - start
        if i % c["every_n"] == 0:
            _, frame = cap.retrieve()
            img = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            checked += 1
            for det in dets:
                x = pick_face(det.detect(img).detections, H, c["max_face_y"])
                if x is not None:
                    break
            if x is None and pose:
                x = pick_pose(pose.detect(img).pose_landmarks, W, c["max_face_y"])
                by_pose += x is not None
            if x is not None:
                centers.append((t, x))
        i += 1
    cap.release()
    for det in dets + ([pose] if pose else []):
        det.close()
    return centers, checked, W, H, by_pose


def detect_cuts(video, start, end, c):
    """Klip içindeki kamera açısı değişimleri (klip başına göre sn)."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video), ContentDetector(threshold=c["scene_threshold"]), start_time=start, end_time=end)
    return [round(a.seconds - start, 3) for a, _ in scenes[1:]]


def simplify(keys, tol):
    """Ramer–Douglas–Peucker: [(t, x)] yolunu en fazla tol px sapmayla az noktaya indirir (ffmpeg ifadesi kısa kalsın)."""
    if len(keys) < 3:
        return keys
    (t0, x0), (t1, x1) = keys[0], keys[-1]
    d = [abs(x - (x0 + (x1 - x0) * (t - t0) / ((t1 - t0) or 1))) for t, x in keys[1:-1]]
    i = max(range(len(d)), key=d.__getitem__)
    if d[i] <= tol:
        return [keys[0], keys[-1]]
    return simplify(keys[:i + 2], tol)[:-1] + simplify(keys[i + 1:], tol)


def follow(samples, start, W, cw, c):
    """Çekim içi ölü bölgeli takip → [(t, crop x)]. Yüz kırpılan alanın orta bandındayken (center_band, %40)
    kamera durur; dışına çıkacaksa yüzü bandın içinde tutacak kadar kayar. Yüz bulunamayan aralıkta son
    bulunan konum korunur. Kenarda (kaynak karenin dışına taşamayacağı için) kural sağlanamayabilir."""
    lim = c["center_band"] / 2 * cw * 0.75  # 0.75: ara karelerde ve doğrusal geçişte banttan taşmasın diye pay
    clamp = lambda cx: int(min(max(cx - cw / 2, 0), W - cw))
    keys, cx, last_t = [], None, None
    for t, x in samples:
        if cx is None:
            keys.append((start, clamp(x)))  # çekim başından ilk bulunan konum
        elif t - last_t > c["hold_gap"]:
            keys.append((t - 0.05, clamp(cx)))  # yüzsüz aralık: son konumda bekle
        cx = x if cx is None else min(max(cx, x - lim), x + lim)
        keys.append((t, clamp(cx)))
        last_t = t
    return [(round(t, 3), x) for t, x in simplify(keys, c["path_tol"])]


def shots_from(cuts, samples, dur, W, H, c):
    """Kesmelerden çekimler; her çekimde crop x yolu (keys) o çekimdeki yüzlerden. Çok kısa çekimler öncekiyle
    birleşir. Yüz bulunamayan çekim önceki çekimin son konumunu kullanır (ilkse ilk bulunan, hiç yoksa merkez)."""
    bounds = [0.0]
    for t in cuts:
        if t - bounds[-1] >= c["min_shot"] and dur - t >= c["min_shot"]:
            bounds.append(t)
    bounds.append(dur)
    cw = crop_box([], W, H, 1)["w"]
    shots, prev = [], None
    for a, b in zip(bounds, bounds[1:]):
        pts = [(t, x) for t, x in samples if a <= t < b]
        keys = follow(pts, round(a, 3), W, cw, c) if pts else ([(round(a, 3), prev)] if prev is not None else None)
        shots.append({"t": round(a, 3), "keys": keys, "faces": len(pts)})
        prev = keys[-1][1] if keys else prev
    first = next((s["keys"][0][1] for s in shots if s["keys"]), crop_box([], W, H, 1)["x"])
    for s in shots:  # baştaki yüzsüz çekimler: ilk bulunan konum, hiç yoksa merkez
        s["keys"] = s["keys"] or [(s["t"], first)]
        s["x"] = s["keys"][0][1]
    return shots


def x_at(keys, t):
    """crop_x_expr'in Python karşılığı: anahtarlar arasında doğrusal, öncesi/sonrası sabit."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, x0), (t1, x1) in zip(keys, keys[1:]):
        if t < t1:
            return x0 + (x1 - x0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def in_band(samples, shots, cw, band):
    """Yüz merkezinin kırpılan alanın orta bandında (ör. %40) kaldığı örneklerin oranı."""
    if not samples:
        return None
    ok = 0
    for t, fx in samples:
        s = ([s for s in shots if s["t"] <= t] or shots[:1])[-1]
        ok += abs(fx - x_at(s["keys"], t) - cw / 2) <= band / 2 * cw
    return round(ok / len(samples), 3)


def shot_expr(s):
    """Bir çekimin crop x'i: anahtar noktalar arasında doğrusal geçiş, sondan sonra sabit."""
    keys = s.get("keys") or [(s["t"], s["x"])]  # eski reframe.json: çekim başına tek x
    expr = str(keys[-1][1])
    for (t0, x0), (t1, x1) in reversed(list(zip(keys, keys[1:]))):
        lin = x0 if x0 == x1 else f"{x0}+{x1 - x0}*(t-{t0})/{round(t1 - t0, 3)}"
        expr = f"if(lt(t,{t1}),{lin},{expr})"
    return expr


def crop_x_expr(shots):
    """ffmpeg crop x ifadesi (t, klip başından itibaren sn): çekim değişince x atlar, çekim içinde yumuşak takip."""
    expr = shot_expr(shots[-1])
    for cur, nxt in zip(reversed(shots[:-1]), reversed(shots[1:])):
        expr = f"if(lt(t,{nxt['t']}),{shot_expr(cur)},{expr})"
    return expr


def run(video, cfg, force=False):
    def make():
        c, out = cfg["reframe"], {}
        for b in select_llm.run(video, cfg):
            samples, checked, W, H, by_pose = face_centers(video, b["start"], b["end"], c)
            cuts = detect_cuts(video, b["start"], b["end"], c)
            shots = shots_from(cuts, samples, b["end"] - b["start"], W, H, c)
            box = crop_box([x for _, x in samples], W, H, c["smooth_window"])
            band = in_band(samples, shots, box["w"], c["center_band"])
            log.info("Bit #%d: %d/%d karede kişi (%d pose ile), %d çekim, %d takip noktası, yüz orta bantta %s",
                     b["n"], len(samples), checked, by_pose, len(shots), sum(len(s["keys"]) for s in shots),
                     "–" if band is None else f"%{band * 100:.1f}")
            out[str(b["n"])] = {**box, "shots": shots, "src_w": W, "src_h": H, "faces": len(samples), "checked": checked,
                                "in_band": band}
        return out
    return cached(work_dir(video) / "reframe.json", make, force)

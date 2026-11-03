"""8) Kadraj ("kameraman"): PySceneDetect çekimleri + komedyen tanıma (YuNet yüz + SFace kimlik) + MediaPipe pose
gövde merkezi. Her çekim yakın / geniş / ara olarak sınıflanır ve tipine göre kadraj yolu çıkarılır → reframe.json.
- ara: komedyen güvenilir bulunamıyor (seyirci, mekân) → sabit merkez (ya da cutaway_mode: blur_bg).
- geniş: gövde merkezinin hareket aralığı sığıyorsa tek sabit kadraj; sığmıyorsa sabit parçalar + yavaş, yumuşak geçiş.
- yakın: yüz kırpılan alanın orta bandında (center_band) tutulur; kısa hareketler yok sayılır, hız sınırlı."""
import math
import subprocess

import cv2
import numpy as np

from . import select_llm
from .common import ROOT, cached, log, read_json, work_dir

MODELS = ROOT / "models"
YUNET, SFACE = "face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"
CLOSE, WIDE, CUTAWAY = "yakın", "geniş", "ara"


def crop_box(centers, W, H, window=1):
    """9:16 crop kutusu; x = yüz merkezlerinin medyanı, yoksa merkez."""
    cw = min(W, int(H * 9 / 16)) // 2 * 2
    ch = min(H, int(cw * 16 / 9)) // 2 * 2
    cx = float(np.median(centers)) if len(centers) else W / 2
    x = int(min(max(cx - cw / 2, 0), W - cw)) // 2 * 2
    return {"x": x, "y": (H - ch) // 2 // 2 * 2, "w": cw, "h": ch}


# ---------- tespit ----------

class Detector:
    """YuNet + SFace (OpenCV) ve MediaPipe pose. detect(frame) → {"faces": [...], "poses": [...]}."""

    def __init__(self, c):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        self.yunet = cv2.FaceDetectorYN.create(str(MODELS / YUNET), "", (320, 320), c["face_score"])
        self.sface = cv2.FaceRecognizerSF.create(str(MODELS / SFACE), "")
        pose_p = MODELS / f"{c['pose_model']}.task" if c.get("pose_model") else None
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_p)), num_poses=3,
            min_pose_detection_confidence=c["min_confidence"])) if pose_p and pose_p.exists() else None
        if pose_p and not self.pose:
            log.warning("Pose modeli yok (%s); geniş çekimlerde gövde takibi kapalı", pose_p)

    def detect(self, frame):
        H, W = frame.shape[:2]
        self.yunet.setInputSize((W, H))
        _, rows = self.yunet.detect(frame)
        faces = []
        for r in rows if rows is not None else []:
            f = self.sface.feature(self.sface.alignCrop(frame, r)).ravel()
            faces.append({"box": [float(v) for v in r[:4]], "score": float(r[14]), "feat": f / (np.linalg.norm(f) or 1)})
        poses = []
        if self.pose:
            from mediapipe import Image, ImageFormat
            img = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            poses = [body_of(lm, W, H) for lm in self.pose.detect(img).pose_landmarks]
        return {"faces": faces, "poses": poses}

    def close(self):
        if self.pose:
            self.pose.close()


def body_of(lm, W, H):
    """Gövde merkezi = omuz + kalça ortalaması (el hareketlerinden etkilenmez); burun konumu sahne kontrolü için."""
    pts = [lm[i] for i in (11, 12, 23, 24) if lm[i].visibility > 0.5] or [lm[11], lm[12]]
    return {"x": sum(p.x for p in pts) / len(pts) * W, "nose_x": lm[0].x * W, "nose_y": lm[0].y * H,
            "width": abs(lm[11].x - lm[12].x) * W}


def analyze(video, start, end, c, det):
    """Örneklenen kareler (her every_n karede bir): [{"t", "faces", "poses"}], t klip başına göre."""
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    out, i = [], 0
    while cap.get(cv2.CAP_PROP_POS_MSEC) <= end * 1000 and cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - start
        if i % c["every_n"] == 0:
            _, frame = cap.retrieve()
            out.append({"t": round(t, 3), **det.detect(frame)})
        i += 1
    cap.release()
    return out, W, H


def reference(samples, H, c):
    """Komedyenin referans yüz vektörü: yakın çekim yüzlerinin (yükseklik ≥ close_face) en kalabalık benzerlik
    kümesinin ortalaması. Yakın yüz yoksa en büyük yüzler. samples: tüm klipler (gösteri geneli)."""
    faces = [f for s in samples for f in s["faces"]]
    big = [f for f in faces if f["box"][3] / H >= c["close_face"]] or sorted(faces, key=lambda f: -f["box"][3])[:30]
    if not big:
        return None
    F = np.stack([f["feat"] for f in big])
    sim = F @ F.T >= c["face_match"]
    members = F[sim[int(np.argmax(sim.sum(1)))]]
    ref = members.mean(0)
    return ref / np.linalg.norm(ref)


# ---------- komedyeni seçme ----------

def comedian_in(s, ref, W, H, c):
    """Örnekte komedyenin yüzü: referansa en benzer yüz (eşiğin altındakiler seyirci/tablo sayılır).
    → (yüz indeksi, benzerlik) ya da (None, en iyi benzerlik)."""
    if ref is None or not s["faces"]:
        return None, None
    # Küçük yüzlerin (uzak seyirci) vektörü güvenilmez: tanımaya yalnız min_face'ten büyükler girer
    sims = [float(f["feat"] @ ref) if f["box"][3] / H >= c["min_face"] else -1.0 for f in s["faces"]]
    i = int(np.argmax(sims))
    return (i if sims[i] >= c["face_match"] else None), sims[i]


def track(samples, ref, bounds, W, H, c):
    """Her örnekte komedyenin konumu: {"t", "fx" (yüz x | None), "bx" (gövde x), "fh" (yüz yüksekliği oranı),
    "src": "face"|"pose"} ya da None. Yüz tanınmadıysa sahne bölgesindeki (burun üst %stage_y) ve son bilinen
    konuma yakın pose kişisi kabul edilir; çekimde hiç konum yoksa omuzları en geniş olan."""
    out = []
    for a, b in zip(bounds, bounds[1:]):
        last = None
        for s in (s for s in samples if a <= s["t"] < b):
            i, sim = comedian_in(s, ref, W, H, c)
            s["who"], s["sim"] = i, sim
            s["pick"] = None
            p = None
            if i is not None:
                x, y, w, h = s["faces"][i]["box"]
                fx = x + w / 2
                body = [q for q in s["poses"] if x <= q["nose_x"] <= x + w]  # yüzün gövdesi
                p = {"t": s["t"], "fx": fx, "bx": body[0]["x"] if body else fx, "fh": h / H, "src": "face"}
            else:
                stage = [q for q in s["poses"] if q["nose_y"] <= c["stage_y"] * H]
                if last is not None:
                    stage = [q for q in stage if abs(q["x"] - last) <= c["pose_near"] * W]
                    q = min(stage, key=lambda q: abs(q["x"] - last), default=None)
                else:
                    q = max(stage, key=lambda q: q["width"], default=None)
                if q:
                    p = {"t": s["t"], "fx": None, "bx": q["x"], "fh": None, "src": "pose"}
            s["pick"] = p
            if p:
                out.append(p)
                last = p["bx"]
    return out


# ---------- çekimler ----------

def shot_bounds(cuts, dur):
    return [0.0, *[t for t in cuts if 0 < t < dur], dur]


def drop_outliers(pts, W, c):
    """Çekim içinde 3 sn'lik kayan medyandan max_jump'tan fazla sapan tekil konumlar (yanlış eşleşme) atılır."""
    if len(pts) < 3:
        return pts
    med = rolling_median([p["t"] for p in pts], [p["bx"] for p in pts], 3.0)
    return [p for p, m in zip(pts, med) if abs(p["bx"] - m) <= c["max_jump"] * W]


def classify(pts, n_checked, W, c):
    """yakın | geniş | ara. Ara: tespit oranı < min_detect_rate ya da ardışık konumların max_jump_ratio'dan
    fazlası kare genişliğinin max_jump'ından fazla zıplıyor."""
    if not n_checked or len(pts) / n_checked < c["min_detect_rate"]:
        return CUTAWAY
    xs = [p["bx"] for p in pts]
    jumps = sum(abs(x1 - x0) > c["max_jump"] * W for x0, x1 in zip(xs, xs[1:]))
    if len(xs) > 1 and jumps / (len(xs) - 1) > c["max_jump_ratio"]:
        return CUTAWAY
    fh = [p["fh"] for p in pts if p["fh"]]
    return CLOSE if fh and float(np.median(fh)) >= c["close_face"] else WIDE


def rolling_median(ts, xs, window):
    """Zamana göre ortalanmış kayan medyan: window/2'den kısa sapmalar (el, eğilme) silinir, gecikme yok."""
    ts, xs = np.asarray(ts, float), np.asarray(xs, float)
    lo, hi = np.searchsorted(ts, ts - window / 2), np.searchsorted(ts, ts + window / 2, side="right")
    return [float(np.median(xs[a:b])) for a, b in zip(lo, hi)]


def rate_limit(keys, vmax):
    """Hız sınırı (px/sn), ileri ve geri geçişin ortalaması: geçiş hareketin etrafında ortalanır, gecikmez."""
    def one(ks):
        out = [ks[0][1]]
        for (t0, _), (t1, x) in zip(ks, ks[1:]):
            d = vmax * abs(t1 - t0)
            out.append(min(max(x, out[-1] - d), out[-1] + d))
        return out
    f, b = one(keys), one(keys[::-1])[::-1]
    return [(t, (x0 + x1) / 2) for (t, _), x0, x1 in zip(keys, f, b)]


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


def vmax_of(c, H):
    return c["max_speed"] * H / 1080  # config 1080p kaynak ölçeğinde


def follow(samples, start, W, cw, c, H=1080):
    """Yakın çekim: ölü bölgeli takip → [(t, crop x)]. Yüz kırpılan alanın orta bandındayken (center_band, %40)
    kamera durur; dışına çıkacaksa bandın içinde tutacak kadar kayar. min_move'dan kısa hareketler (kayan medyan)
    yok sayılır, hız max_speed ile sınırlı. Yüzsüz aralıkta son konum korunur."""
    lim = c["center_band"] / 2 * cw * 0.75  # 0.75: ara karelerde ve doğrusal geçişte banttan taşmasın diye pay
    clamp = lambda cx: min(max(cx - cw / 2, 0), W - cw)
    ts = [t for t, _ in samples]
    keys, cx, last_t = [], None, None
    for t, x in zip(ts, rolling_median(ts, [x for _, x in samples], 2 * c["min_move"])):
        if cx is None:
            keys.append((start, clamp(x)))  # çekim başından ilk bulunan konum
        elif t - last_t > c["hold_gap"]:
            keys.append((t - 0.05, clamp(cx)))  # yüzsüz aralık: son konumda bekle
        cx = x if cx is None else min(max(cx, x - lim), x + lim)
        keys.append((t, clamp(cx)))
        last_t = t
    keys = rate_limit(keys, vmax_of(c, H))
    return [(round(t, 3), int(x)) for t, x in simplify(keys, c["path_tol"])]


def plan_wide(pts, a, b, W, cw, c, H):
    """Geniş çekim: gövde merkezinin (bx) aralığı kenarlarda wide_margin pay bırakarak sığıyorsa tek sabit kadraj.
    Sığmıyorsa sabit parçalara böl, aralarda ease in/out geçiş (≤ max_speed), yön değiştirmeden dir_hold bekle."""
    clamp = lambda cx: int(min(max(cx - cw / 2, 0), W - cw))
    ts = [p["t"] for p in pts]
    xs = rolling_median(ts, [p["bx"] for p in pts], 1.0)
    usable = cw * (1 - 2 * c["wide_margin"])
    lo, hi = np.percentile(xs, 5), np.percentile(xs, 95)
    if hi - lo <= usable:
        return [(a, clamp((lo + hi) / 2))]
    # Açgözlü bölme: parça içindeki aralık usable'a sığdıkça uzat
    holds = [[0, 0, xs[0], xs[0]]]  # [ilk i, son i, min, max]
    for i, x in enumerate(xs[1:], 1):
        h = holds[-1]
        if max(h[3], x) - min(h[2], x) <= usable:
            h[1], h[2], h[3] = i, min(h[2], x), max(h[3], x)
        else:
            holds.append([i, i, x, x])
    # min_move'dan kısa parçalar öncekine katılır (kısa gidip gelmeler yok sayılır)
    merged = []
    for h in holds:
        if merged and ts[h[1]] - ts[h[0]] < c["min_move"]:
            merged[-1][1] = h[1]
        else:
            merged.append(h)
    targets = []
    for h in merged:
        x = clamp((h[2] + h[3]) / 2)
        if targets and abs(x - targets[-1][2]) < 0.05 * cw:
            continue
        targets.append((ts[h[0]], ts[h[1]], x))
    vmax = vmax_of(c, H)
    keys, prev_end, prev_dir = [(a, targets[0][2])], a, 0
    for (_, e0, x0), (s1, _, x1) in zip(targets, targets[1:]):
        d = math.pi / 2 * abs(x1 - x0) / vmax  # cos ease'in tepe hızı ortalamanın π/2 katı
        start = max((e0 + s1) / 2 - d / 2, prev_end)
        sgn = 1 if x1 > x0 else -1
        if prev_dir and sgn != prev_dir:
            start = max(start, prev_end + c["dir_hold"])
        keys += [(round(start, 3), x0), (round(start + d, 3), x1, 1)]
        prev_end, prev_dir = start + d, sgn
    return keys


def shots_from(cuts, pts, checked, dur, W, H, c):
    """Çekimler: {"t", "type", "keys", "faces"}. checked: örneklenen kare zamanları (tespit oranı için).
    min_shot'tan kısa çekimler önceki çekimin kadrajını (ve tipini) kullanır."""
    cw = crop_box([], W, H)["w"]
    center = crop_box([], W, H)["x"]
    bounds = shot_bounds(cuts, dur)
    shots = []
    for a, b in zip(bounds, bounds[1:]):
        sp = drop_outliers([p for p in pts if a <= p["t"] < b], W, c)
        s = {"t": round(a, 3), "faces": len(sp)}
        if b - a < c["min_shot"]:
            s["short"] = True
        else:
            kind = classify(sp, sum(a <= t < b for t in checked), W, c)
            if kind == CUTAWAY:
                keys = [(s["t"], center)]
            elif kind == WIDE:
                keys = plan_wide(sp, s["t"], b, W, cw, c, H)
            else:
                face = [(p["t"], p["fx"] if p["fx"] is not None else p["bx"]) for p in sp]
                keys = follow(face, s["t"], W, cw, c, H)
            s.update(type=kind, keys=keys)
        shots.append(s)
    for i, s in enumerate(shots):  # kısa çekimler: önceki çekimin son kadrajı (ilk çekimse sonrakinin ilki)
        if s.pop("short", False):
            prev = shots[i - 1] if i else None
            nxt = next((n for n in shots[i + 1:] if "keys" in n), None)
            if prev:
                s.update(type=prev["type"], keys=[(s["t"], prev["keys"][-1][1])])
            elif nxt:
                s.update(type=nxt["type"], keys=[(s["t"], nxt["keys"][0][1])])
            else:
                s.update(type=CUTAWAY, keys=[(s["t"], center)])
        s["x"] = s["keys"][0][1]
    return shots


# ---------- yol, ölçüm, ffmpeg ----------

def x_at(keys, t):
    """crop_x_expr'in Python karşılığı: anahtarlar arasında doğrusal (3. eleman 1 ise cos ease), öncesi/sonrası sabit."""
    if t <= keys[0][0]:
        return keys[0][1]
    for k0, k1 in zip(keys, keys[1:]):
        (t0, x0), (t1, x1) = k0[:2], k1[:2]
        if t < t1:
            u = (t - t0) / (t1 - t0)
            if len(k1) > 2 and k1[2]:
                u = (1 - math.cos(math.pi * u)) / 2
            return x0 + (x1 - x0) * u
    return keys[-1][1]


def shot_at(shots, t):
    return ([s for s in shots if s["t"] <= t] or shots[:1])[-1]


def in_band(samples, shots, cw, band):
    """Yüz merkezinin kırpılan alanın orta bandında kaldığı örneklerin oranı (ara çekimler hariç).
    samples: [(t, yüz x)]."""
    ok = n = 0
    for t, fx in samples:
        s = shot_at(shots, t)
        if s.get("type") == CUTAWAY:
            continue
        n += 1
        ok += abs(fx - x_at(s["keys"], t) - cw / 2) <= band / 2 * cw
    return round(ok / n, 3) if n else None


def jitter(shots, dur, H, types=None, step=0.1, eps=3.0):
    """Titreme: saniye başına ortalama kadraj hareketi (px/sn, 1080p ölçeği) ve 3 sn başına yön değişimi.
    Çekim geçişleri (kesme) sayılmaz. types verilirse yalnız o tipteki çekimler."""
    scale = 1080 / H
    bounds = [s["t"] for s in shots] + [dur]
    moved = secs = flips = 0.0
    for s, b in zip(shots, bounds[1:]):
        if types and s.get("type") not in types:
            continue
        ts = np.append(np.arange(s["t"], b, step), b)  # çekim sonu dahil (o anki x aynı çekimin yolundan)
        if len(ts) < 2:
            continue
        xs = np.array([x_at(s["keys"], t) for t in ts]) * scale
        v = np.diff(xs) / step
        moved += float(np.abs(np.diff(xs)).sum())
        secs += ts[-1] - ts[0]
        sg = [np.sign(x) for x in v if abs(x) > eps]
        flips += sum(a != b for a, b in zip(sg, sg[1:]))
    return {"speed": round(moved / secs, 1) if secs else 0.0, "dir_per_3s": round(flips / secs * 3, 2) if secs else 0.0,
            "secs": round(secs, 1)}


def shot_expr(s):
    """Bir çekimin crop x'i: anahtarlar arasında doğrusal ya da cos ease, sondan sonra sabit."""
    keys = s.get("keys") or [(s["t"], s["x"])]  # eski reframe.json: çekim başına tek x
    expr = str(keys[-1][1])
    for k0, k1 in reversed(list(zip(keys, keys[1:]))):
        (t0, x0), (t1, x1) = k0[:2], k1[:2]
        u = f"(t-{t0})/{round(t1 - t0, 3)}"
        if x0 == x1:
            lin = str(x0)
        elif len(k1) > 2 and k1[2]:
            lin = f"{x0}+{x1 - x0}*(1-cos(PI*{u}))/2"
        else:
            lin = f"{x0}+{x1 - x0}*{u}"
        expr = f"if(lt(t,{t1}),{lin},{expr})"
    return expr


def crop_x_expr(shots):
    """ffmpeg crop x ifadesi (t, klip başından itibaren sn): çekim değişince x atlar, çekim içinde yol izlenir."""
    expr = shot_expr(shots[-1])
    for cur, nxt in zip(reversed(shots[:-1]), reversed(shots[1:])):
        expr = f"if(lt(t,{nxt['t']}),{shot_expr(cur)},{expr})"
    return expr


def cutaway_ranges(box, dur):
    """cutaway_mode: blur_bg iken bulanık arka planlı gösterilecek ara çekim aralıkları [(a, b)]."""
    if box.get("cutaway_mode") != "blur_bg":
        return []
    shots = box["shots"]
    ends = [s["t"] for s in shots[1:]] + [dur]
    return [(s["t"], e) for s, e in zip(shots, ends) if s.get("type") == CUTAWAY]


# ---------- adım ----------

def detect_cuts(video, start, end, c):
    """Klip içindeki kamera açısı değişimleri (klip başına göre sn)."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video), ContentDetector(threshold=c["scene_threshold"]), start_time=start, end_time=end)
    return [round(a.seconds - start, 3) for a, _ in scenes[1:]]


def build(b, samples, ref, cuts, W, H, c):
    """Bir klibin kadraj kaydı (reframe.json'daki değer)."""
    dur = b["end"] - b["start"]
    pts = track(samples, ref, shot_bounds(cuts, dur), W, H, c)
    shots = shots_from(cuts, pts, [s["t"] for s in samples], dur, W, H, c)
    box = crop_box([], W, H)
    faces = [(p["t"], p["fx"]) for p in pts if p["fx"] is not None]
    return {**box, "shots": shots, "src_w": W, "src_h": H, "faces": len(pts), "checked": len(samples),
            "cutaway_mode": c["cutaway_mode"], "in_band": in_band(faces, shots, box["w"], c["center_band"]),
            "jitter": jitter(shots, dur, H), "jitter_calm": jitter(shots, dur, H, (WIDE, CUTAWAY))}


def run(video, cfg, force=False):
    def make():
        c, out = cfg["reframe"], {}
        bits = select_llm.run(video, cfg)
        det = Detector(c)
        try:
            data = {}
            for b in bits:
                samples, W, H = analyze(video, b["start"], b["end"], c, det)
                data[b["n"]] = (samples, W, H, detect_cuts(video, b["start"], b["end"], c))
        finally:
            det.close()
        allS = [s for v in data.values() for s in v[0]]
        ref = reference(allS, data[bits[0]["n"]][2], c) if bits else None
        if bits and ref is None:
            log.warning("Komedyenin yüzü hiçbir klipte bulunamadı; yalnız pose ile takip edilecek")
        for b in bits:
            samples, W, H, cuts = data[b["n"]]
            r = build(b, samples, ref, cuts, W, H, c)
            kinds = [s["type"] for s in r["shots"]]
            j, jc = r["jitter"], r["jitter_calm"]
            log.info("Bit #%d: %d/%d karede komedyen, çekimler %s | titreme %.0f px/sn, %.2f yön/3sn "
                     "(geniş+ara %.2f yön/3sn) | yüz orta bantta %s", b["n"], r["faces"], r["checked"],
                     " ".join(f"{k}:{kinds.count(k)}" for k in (CLOSE, WIDE, CUTAWAY) if k in kinds),
                     j["speed"], j["dir_per_3s"], jc["dir_per_3s"], "–" if r["in_band"] is None else f"%{r['in_band'] * 100:.1f}")
            out[str(b["n"])] = r
        out["_ref"] = None if ref is None else [round(float(v), 5) for v in ref]
        return out
    res = cached(work_dir(video) / "reframe.json", make, force)
    return {k: v for k, v in res.items() if not k.startswith("_")}


# ---------- hata ayıklama videosu ----------

def debug(video, cfg, n):
    """Kaynak videonun üstüne kadraj dikdörtgeni, komedyen yüzü (yeşil), reddedilen yüzler (kırmızı, benzerlikle),
    pose ile kabul edilen gövde (mavi) ve çekim tipi → work/debug/{video}_{n}.mp4."""
    c = cfg["reframe"]
    wd = work_dir(video)
    b = next(x for x in select_llm.run(video, cfg) if x["n"] == n)
    rp = wd / "reframe.json"
    saved = read_json(rp) if rp.exists() else {}
    det = Detector(c)
    try:
        samples, W, H = analyze(video, b["start"], b["end"], c, det)
    finally:
        det.close()
    ref = np.array(saved["_ref"]) if saved.get("_ref") else reference(samples, H, c)
    cuts = detect_cuts(video, b["start"], b["end"], c)
    r = build(b, samples, ref, cuts, W, H, c)
    out = ROOT / "work" / "debug" / f"{wd.name}_{n}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    ow, oh = 960, int(960 * H / W) // 2 * 2
    k = ow / W
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{ow}x{oh}", "-r", str(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS)),
                           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                           str(out)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, b["start"] * 1000)
    si = 0
    P = lambda v: int(v * k)
    while cap.get(cv2.CAP_PROP_POS_MSEC) <= b["end"] * 1000:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - b["start"]
        while si + 1 < len(samples) and samples[si + 1]["t"] <= t:
            si += 1
        s, shot = samples[si], shot_at(r["shots"], t)
        img = cv2.resize(frame, (ow, oh))
        for i, f in enumerate(s["faces"]):
            x, y, w, h = f["box"]
            sim = float(f["feat"] @ ref) if ref is not None else 0.0
            col = (0, 220, 0) if i == s.get("who") else (0, 0, 230)
            cv2.rectangle(img, (P(x), P(y)), (P(x + w), P(y + h)), col, 2)
            cv2.putText(img, f"{sim:.2f}", (P(x), P(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        p = s.get("pick")
        if p and p["src"] == "pose":
            cv2.circle(img, (P(p["bx"]), oh // 2), 8, (255, 140, 0), -1)
        cx = x_at(shot["keys"], t)
        cv2.rectangle(img, (P(cx), P(r["y"])), (P(cx + r["w"]), P(r["y"] + r["h"])), (0, 220, 255), 3)
        label = f"{t:5.1f} sn  {shot['type']}" + ("  (blur_bg)" if shot["type"] == CUTAWAY and r["cutaway_mode"] == "blur_bg" else "")
        cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        ff.stdin.write(img.tobytes())
    cap.release()
    ff.stdin.close()
    ff.wait()
    log.info("Hata ayıklama videosu: %s (çekimler: %s)", out, ", ".join(f"{s['t']}:{s['type']}" for s in r["shots"]))
    return out

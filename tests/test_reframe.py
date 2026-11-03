import math
from types import SimpleNamespace as N

import numpy as np
import pytest

from clipper import reframe, render
from clipper.common import load_config

C = load_config()["reframe"]
W, H = 1920, 1080
CW = reframe.crop_box([], W, H)["w"]  # 606
x_at = reframe.x_at


def P(t, bx, fx=None, fh=None):
    return {"t": t, "bx": bx, "fx": fx, "fh": fh, "src": "face" if fx is not None else "pose"}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_crop_box_center_and_edges():
    assert reframe.crop_box([], W, H) == {"x": 656, "y": 2, "w": 606, "h": 1076}
    assert reframe.crop_box([100.0] * 5, W, H)["x"] == 0
    r = reframe.crop_box([1900.0] * 5, W, H)
    assert r["x"] + r["w"] <= W


def test_reference_and_track_reject_audience_and_painting():
    me, other = unit([1, 0.1, 0]), unit([0, 1, 0])
    face = lambda x, h, f: {"box": [x - h / 2, 200, h, h], "score": 0.9, "feat": f}
    # yakın çekimler: komedyen çoğunlukta, bir seyirci yakın planı
    close = [{"t": i * 0.2, "faces": [face(900, 300, me)], "poses": []} for i in range(8)]
    close.append({"t": 1.8, "faces": [face(900, 300, other)], "poses": []})
    ref = reframe.reference(close, H, C)
    assert float(ref @ me) > 0.99
    # geniş çekim: tablo (büyük, başka kimlik) + küçük seyirci yüzü (eşleşse bile min_face altında) + komedyen
    s = {"t": 0.0, "faces": [face(1700, 180, other), face(700, 30, me), face(300, 50, me)],
         "poses": [{"x": 310, "nose_x": 300, "nose_y": 220, "width": 60}]}
    pts = reframe.track([s], ref, [0, 1], W, H, C)
    assert s["who"] == 2 and pts[0]["fx"] == 300 and pts[0]["bx"] == 310  # gövde merkezi pose'tan


def test_track_pose_only_in_stage_and_near_last():
    ref = unit([1, 0, 0])
    pose = lambda x, ny=300, w=80: {"x": x, "nose_x": x, "nose_y": ny, "width": w}
    samples = [{"t": 0.0, "faces": [], "poses": [pose(900), pose(400, ny=900, w=200)]},  # alttaki seyirci elenir
               {"t": 0.2, "faces": [], "poses": [pose(930), pose(1600)]},               # yakındaki seçilir
               {"t": 0.4, "faces": [], "poses": [pose(1600)]}]                           # uzak zıplama reddedilir
    pts = reframe.track(samples, ref, [0, 1], W, H, C)
    assert [p["bx"] for p in pts] == [900, 930]


def test_classify_close_wide_cutaway():
    close = [P(i * 0.2, 900, 900, 0.3) for i in range(20)]
    wide = [P(i * 0.2, 900 + i, 900, 0.05) for i in range(20)]
    assert reframe.classify(close, 20, W, C) == reframe.CLOSE
    assert reframe.classify(wide, 20, W, C) == reframe.WIDE
    assert reframe.classify(wide[:8], 20, W, C) == reframe.CUTAWAY  # tespit oranı %40
    jumpy = [P(i * 0.2, 300 if i % 2 else 1500) for i in range(20)]
    assert reframe.classify(jumpy, 20, W, C) == reframe.CUTAWAY


def test_drop_outliers_removes_single_false_match():
    pts = [P(i * 0.2, 1550.0) for i in range(15)]
    pts[7] = P(1.4, 750.0)
    assert [p["bx"] for p in reframe.drop_outliers(pts, W, C)] == [1550.0] * 14


def test_wide_shot_fits_is_one_static_frame():
    # Joe List #5 0–11 sn: küçük komedyen sağa sola adım atıyor, elleri oynuyor → tek sabit kadraj
    pts = [P(i * 0.2, 830 + 120 * math.sin(i / 3) + (40 if i % 3 == 0 else 0)) for i in range(55)]
    keys = reframe.plan_wide(pts, 0.0, 11.0, W, CW, C, H)
    assert len(keys) == 1
    assert reframe.jitter([{"t": 0.0, "keys": keys}], 11.0, H)["speed"] == 0


def test_wide_shot_pans_slowly_with_ease_and_direction_hold():
    # Uzun yürüyüş: 400 → 1500 px (sığmaz), sonra hemen geri 900 → parçalı sabit + yavaş geçiş
    pts = [P(t, 400 + 1100 * min(t / 6, 1) - 600 * max(0, min((t - 8) / 2, 1))) for t in np.arange(0, 16, 0.2)]
    keys = reframe.plan_wide(pts, 0.0, 16.0, W, CW, C, H)
    assert len(keys) > 1 and any(len(k) > 2 for k in keys)
    xs = [x_at(keys, t) for t in np.arange(0, 16, 0.02)]
    v = np.abs(np.diff(xs)) / 0.02
    assert v.max() <= C["max_speed"] * 1.02  # 1080p'de 150 px/sn
    assert reframe.jitter([{"t": 0.0, "keys": keys}], 16.0, H)["dir_per_3s"] <= 1
    # yön değişiminden önce en az dir_hold bekleme
    moves = [(k0[0], k1[0], np.sign(k1[1] - k0[1])) for k0, k1 in zip(keys, keys[1:]) if k1[1] != k0[1]]
    for (_, a1, d0), (b0, _, d1) in zip(moves, moves[1:]):
        if d0 != d1:
            assert b0 - a1 >= C["dir_hold"] - 1e-6


def test_close_follow_keeps_walking_face_in_band_and_ignores_short_moves():
    # ISMO "Kısaltmalar": yüz 692 → 1160 → 850 px yürüyor
    path = [692 + 468 * min(t / 5, 1) - 300 * max(0, min((t - 10) / 4, 1)) for t in [i * 0.2 for i in range(110)]]
    samples = [(i * 0.2, x) for i, x in enumerate(path)]
    keys = reframe.follow(samples, 0.0, W, CW, C, H)
    assert reframe.in_band(samples, [{"t": 0.0, "keys": keys}], CW, 0.4) >= 0.95
    # 1 sn'lik eğilme (yüz bandın dışına) kadrajı oynatmaz
    lean = [(i * 0.2, 960 + (250 if 20 <= i < 25 else 0)) for i in range(50)]
    keys = reframe.follow(lean, 0.0, W, CW, C, H)
    assert len({x for _, x in keys}) == 1


def test_short_shot_uses_previous_framing_and_cutaway_is_center():
    pts = [P(t, 400.0, 400.0, 0.3) for t in np.arange(0, 5, 0.2)]
    checked = list(np.arange(0, 10, 0.2))
    shots = reframe.shots_from([5.0, 5.5], pts, checked, 10.0, W, H, C)
    assert [s["type"] for s in shots] == [reframe.CLOSE, reframe.CLOSE, reframe.CUTAWAY]
    assert shots[1]["keys"] == [(5.0, shots[0]["keys"][-1][1])]  # 0.5 sn'lik çekim önceki kadraj
    assert shots[2]["keys"] == [(5.5, reframe.crop_box([], W, H)["x"])]  # komedyensiz → sabit merkez


def test_in_band_excludes_cutaways_and_jitter_counts_direction_changes():
    shots = [{"t": 0.0, "type": reframe.CLOSE, "keys": [(0.0, 600)]}, {"t": 5.0, "type": reframe.CUTAWAY, "keys": [(5.0, 0)]}]
    assert reframe.in_band([(1.0, 903.0), (6.0, 1800.0)], shots, CW, 0.4) == 1.0
    zig = [{"t": 0.0, "keys": [(0.0, 0), (1.0, 100), (2.0, 0), (3.0, 100)]}]
    j = reframe.jitter(zig, 3.0, H)
    assert j["speed"] == pytest.approx(100, rel=0.05) and j["dir_per_3s"] == pytest.approx(2, abs=0.01)


def test_crop_x_expr_linear_and_ease():
    assert reframe.crop_x_expr([{"t": 0, "x": 100}]) == "100"
    assert reframe.crop_x_expr([{"t": 0, "x": 100}, {"t": 4.5, "x": 300}, {"t": 9, "x": 50}]) == \
        "if(lt(t,4.5),100,if(lt(t,9),300,50))"
    e = reframe.shot_expr({"t": 0, "keys": [(0, 100), (2, 100), (4, 300, 1)]})
    assert e == "if(lt(t,2),100,if(lt(t,4),100+200*(1-cos(PI*(t-2)/2))/2,300))"
    assert x_at([(0, 100), (2, 100), (4, 300, 1)], 3) == pytest.approx(200)
    assert x_at([(0, 100), (2, 100), (4, 300, 1)], 2.5) < 150  # ease in: yavaş başlar


def test_cutaway_blur_bg_filter():
    box = {**reframe.crop_box([], W, H), "cutaway_mode": "blur_bg",
           "shots": [{"t": 0.0, "type": reframe.WIDE, "keys": [(0.0, 300)]}, {"t": 5.0, "type": reframe.CUTAWAY, "keys": [(5.0, 656)]}]}
    f = render.layout_filter("crop", box, 9.0)
    assert "overlay=enable='between(t,5.0,9.0)'" in f and "gblur" in f
    assert "gblur" not in render.layout_filter("crop", {**box, "cutaway_mode": "center"}, 9.0)


def test_body_of_uses_shoulders_and_hips():
    lm = [N(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    lm[0] = N(x=0.9, y=0.2, visibility=0.9)  # baş/el oynasa da gövde merkezi değişmez
    lm[11], lm[12], lm[23], lm[24] = (N(x=x, y=0.5, visibility=0.9) for x in (0.4, 0.6, 0.45, 0.55))
    b = reframe.body_of(lm, 1000, 1000)
    assert b["x"] == pytest.approx(500) and b["nose_y"] == pytest.approx(200) and b["width"] == pytest.approx(200)

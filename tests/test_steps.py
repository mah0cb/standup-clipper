import json
import re
import threading
import urllib.request

import pytest

from clipper import proofread, reframe, review, select_llm, subtitles, translate
from clipper.common import load_config, read_json, write_json

WORDS = [{"i": i, "start": i * 0.5, "end": i * 0.5 + 0.4, "word": f"w{i}"} for i in range(6)]


def test_check_groups():
    translate.check_groups([{"word_from": 0, "word_to": 2}, {"word_from": 3, "word_to": 5}], 6)
    with pytest.raises(ValueError):
        translate.check_groups([{"word_from": 0, "word_to": 2}, {"word_from": 4, "word_to": 5}], 6)  # boşluk
    with pytest.raises(ValueError):
        translate.check_groups([{"word_from": 0, "word_to": 4}], 6)  # eksik


def test_align_uses_word_times():
    g = translate.align([{"word_from": 0, "word_to": 2, "lines": ["a"]}, {"word_from": 3, "word_to": 5, "lines": ["b"]}], WORDS)
    assert (g[0]["start"], g[0]["end"]) == (0.0, 1.4)
    assert (g[1]["start"], g[1]["end"]) == (1.5, 2.9)
    assert g[0]["en"] == "w0 w1 w2"


def test_schema_rejects_long_line():
    import jsonschema
    cfg = load_config()
    bad = {"groups": [{"word_from": 0, "word_to": 0, "lines": ["x" * 39]}], "replacements": []}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, translate.schema(cfg))


def test_karaoke_sum_equals_duration():
    k = subtitles.karaoke(["Bir iki üç", "dört beşşş"], 237)
    assert sum(map(int, re.findall(r"\\k(\d+)", k))) == 237
    assert "\\N" in k and "{\\k" in k


def test_build_ass():
    s = load_config()["subs"]
    bit = {"start": 10.0, "end": 40.0, "hook_tr": "Doktor {ne} dedi?"}
    tr = {"groups": [{"start": 10.5, "end": 12.0, "lines": ["Doktora gittim"]},
                     {"start": 12.2, "end": 14.0, "lines": ["Kilo ver dedi", "ben de"]}]}
    ass = subtitles.build_ass(bit, tr, "Ali Veli", s)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "Dialogue: 1,0:00:00.00,0:00:03.00,Hook" in ass
    assert "(ne)" in ass  # süslü parantez kaçırıldı
    assert "Credit,,0,0,0,,Ali Veli" in ass
    assert "Dialogue: 0,0:00:00.50,0:00:02.20,Main" in ass  # sonraki gruba kadar ekranda


def test_smooth_and_crop():
    assert reframe.smooth([0, 0, 9, 0, 0], 3)[2] == pytest.approx(3)
    box = reframe.crop_box([], 1920, 1080, 15)
    assert box == {"x": 656, "y": 2, "w": 606, "h": 1076}  # yüz yoksa merkez
    left = reframe.crop_box([100.0] * 10, 1920, 1080, 15)
    assert left["x"] == 0  # kenara dayanır
    right = reframe.crop_box([1900.0] * 10, 1920, 1080, 15)
    assert right["x"] + right["w"] <= 1920


def test_review_server_saves(tmp_path):
    write_json(tmp_path / "selected.json", [{"n": 1, "start": 0, "end": 30, "hook_tr": "eski", "score": 8, "reason": "r"},
                                            {"n": 2, "start": 40, "end": 70, "hook_tr": "iki", "score": 7, "reason": "r"}])
    write_json(tmp_path / review.TR_FILE, {"1": {"groups": [{"start": 0, "end": 1, "lines": ["a"]}], "notes": [], "replacements": []},
                                              "2": {"groups": [{"start": 40, "end": 41, "lines": ["b"]}], "notes": [], "replacements": []}})
    srv = review.make_server(tmp_path, load_config(), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        d = json.loads(urllib.request.urlopen(url + "/data").read())
        assert d["approved"] == [1, 2]
        assert b"<title>" in urllib.request.urlopen(url + "/").read()
        body = json.dumps({"approved": [2], "hooks": {"1": "yeni"}, "lines": {"1": [["düzeltildi", "satır"]]}}).encode()
        urllib.request.urlopen(urllib.request.Request(url + "/save", body, {"Content-Type": "application/json"}))
    finally:
        srv.shutdown()
    assert read_json(tmp_path / "review.json") == {"approved": [2]}
    assert read_json(tmp_path / "selected.json")[0]["hook_tr"] == "yeni"
    assert read_json(tmp_path / review.TR_FILE)["1"]["groups"][0]["lines"] == ["düzeltildi", "satır"]


def test_pick_face_ignores_audience():
    from types import SimpleNamespace as N
    face = lambda x, y, s: N(bounding_box=N(origin_x=x, origin_y=y, width=20, height=20), categories=[N(score=s)])
    faces = [face(100, 900, 0.95), face(500, 300, 0.7), face(800, 280, 0.9)]
    assert reframe.pick_face(faces, 1000, 0.6) == 810  # alttaki seyirci elendi, en güvenilir seçildi
    assert reframe.pick_face([face(100, 900, 0.9)], 1000, 0.6) is None


def test_check_text_rejects_parentheses():
    translate.check_text({"groups": [{"lines": ["Tamam, sorun yok"]}], "notes": [{"word": 0, "text": "CPR = kalp masajı"}]}, 3)
    with pytest.raises(ValueError):
        translate.check_text({"groups": [{"lines": ["Bana CPR (kalp masajı)"]}], "notes": []}, 3)
    with pytest.raises(ValueError):
        translate.check_text({"groups": [{"lines": ["a"]}], "notes": [{"word": 5, "text": "x"}]}, 3)


def test_note_in_ass():
    s = load_config()["subs"]
    tr = {"groups": [{"start": 11, "end": 12, "lines": ["Bana CPR lazım!"]}], "notes": [{"start": 11.5, "text": "CPR = kalp masajı"}]}
    ass = subtitles.build_ass({"start": 10.0, "end": 40.0, "hook_tr": "h"}, tr, "", s)
    assert r"Dialogue: 1,0:00:01.50,0:00:04.00,Note,,0,0,0,,{\fad(150,200)}CPR = kalp masajı" in ass


def test_max_bits():
    s = {"per_minutes": 5, "max_bits": 12}
    assert select_llm.max_bits(300, s) == 1
    assert select_llm.max_bits(3774, s) == 12
    assert select_llm.max_bits(1500, s) == 5


def test_clip_end_includes_laugh_but_not_next_speech():
    words = [{"start": 0, "end": 1, "word": "punch."}, {"start": 6, "end": 7, "word": "Next"}]
    assert select_llm.clip_end(words, 0, [{"start": 1.2, "end": 4.0, "prob": 1}], 0.5) == 4.5
    assert select_llm.clip_end(words, 0, [{"start": 1.2, "end": 9.0, "prob": 1}], 0.5) == 6  # sonraki konuşmada kesilir
    assert select_llm.clip_end(words, 0, [], 0.5) == 1.5


def test_select_check_rejects_mid_sentence_and_low_scores():
    words = [{"start": float(k), "end": k + 0.9, "word": f"w{k}" + ("." if k % 10 == 9 else "")} for k in range(60)]
    from clipper import candidates
    starts, ends = set(candidates.sentence_starts(words, 0.6)), set(candidates.sentence_ends(words, 0.6))
    c = {"min_len": 20, "max_len": 90, "tail": 0.5}
    check = select_llm.make_check([{"id": 0}], words, [], starts, ends, {"min_score": 6}, c, 1)
    bit = lambda sw, ew, sc=8: {"id": 0, "start_word": sw, "end_word": ew, "score": sc}
    d = {"bits": [bit(10, 39)]}
    check(d)
    assert d["bits"][0]["start"] == 10.0
    with pytest.raises(ValueError, match="cümle başı"):
        check({"bits": [bit(12, 39)]})
    d = {"bits": [bit(10, 39, sc=4)]}
    check(d)
    assert d["bits"] == []  # 6 altı puan elenir


def test_proofread_keeps_times(monkeypatch):
    tr = {"groups": [{"start": 1, "end": 2, "en": "hi", "lines": ["Selm"]}], "notes": [{"start": 1, "text": "not"}], "replacements": []}
    monkeypatch.setattr(proofread, "ask_json", lambda *a, **k: {
        "groups": [["Selam"]], "notes": [], "changes": [{"group": 0, "before": "Selm", "after": "Selam", "type": "yazım"}]})
    out = proofread.proofread_bit({"n": 1}, tr, load_config())
    assert out["groups"][0] == {"start": 1, "end": 2, "en": "hi", "lines": ["Selam"]}
    assert out["notes"] == [] and out["changes"][0]["type"] == "yazım"


def test_uncovered_and_coverage_message():
    words = [{"start": i, "end": i + 0.5, "word": f"w{i}"} for i in range(5)]
    groups = [{"word_from": 0, "word_to": 1}, {"word_from": 3, "word_to": 4}]
    assert translate.uncovered(groups, 5) == [2]
    with pytest.raises(ValueError, match="2:w2"):
        translate.check_coverage(groups, words)


def test_translate_retries_coverage_then_marks(monkeypatch):
    from clipper import claude_client
    words = [{"start": float(i), "end": i + 0.5, "word": f"w{i}"} for i in range(4)]
    gapped = {"groups": [{"word_from": 0, "word_to": 1, "lines": ["a"]}, {"word_from": 3, "word_to": 3, "lines": ["b"]}],
              "notes": [], "replacements": []}
    prompts = []

    def fake_call(p, c, system=None):
        prompts.append(p)
        return json.dumps(gapped), "test"

    monkeypatch.setattr(claude_client, "_call_cli", fake_call)
    cfg = load_config()
    cfg["claude"]["mode"] = "cli"
    out = translate.translate_bit({"n": 1, "start_word": 0, "end_word": 3}, words, cfg)
    assert len(prompts) == 3 and "2:w2" in prompts[1]  # eksik kelime adıyla yeniden istendi
    assert out["uncovered"] == [{"start": 2.0, "end": 2.5, "word": "w2"}]  # 2 denemeden sonra işaretlendi


def test_bit_words_merges_recovered():
    words = [{"start": 0.0, "end": 1, "word": "a"}, {"start": 5.0, "end": 6, "word": "c"}]
    got = translate.bit_words({"start_word": 0, "end_word": 1}, words, [{"start": 3.0, "end": 4, "word": "b", "recovered": True}])
    assert [w["word"] for w in got] == ["a", "b", "c"]


def test_find_gaps():
    from clipper import gapfill
    words = [{"start": 1.0, "end": 2.0}, {"start": 2.2, "end": 3.0}, {"start": 8.0, "end": 9.0}]
    assert gapfill.find_gaps(words, 0.0, 12.0, 2.0) == [(3.0, 8.0), (9.0, 12.0)]


def test_shots_from_cuts():
    c = {"min_shot": 1.0, "smooth_window": 1}
    samples = [(0.5, 500.0), (1.0, 520.0), (6.0, 1400.0), (6.5, 1420.0)]
    shots = reframe.shots_from([0.3, 5.0, 9.2], samples, 10.0, 1920, 1080, c)
    # 0.3 (çok kısa) ve 9.2 (sona çok yakın) birleşti → 2 çekim
    assert [s["t"] for s in shots] == [0.0, 5.0]
    assert shots[0]["x"] < shots[1]["x"]
    no_face = reframe.shots_from([5.0], [(6.0, 1400.0)], 10.0, 1920, 1080, c)
    assert no_face[0]["x"] == no_face[1]["x"]  # yüzsüz ilk çekim bulunan konumu kullanır


def test_crop_x_expr():
    assert reframe.crop_x_expr([{"t": 0, "x": 100}]) == "100"
    assert reframe.crop_x_expr([{"t": 0, "x": 100}, {"t": 4.5, "x": 300}, {"t": 9, "x": 50}]) == \
        "if(lt(t,4.5),100,if(lt(t,9),300,50))"


def test_should_upscale():
    from clipper import render
    assert render.should_upscale("auto", "crop", 1012)
    assert not render.should_upscale("auto", "crop", 1920)
    assert not render.should_upscale("auto", "blur_bg", 1012)
    assert render.should_upscale("always", "crop", 2160)
    assert not render.should_upscale("off", "crop", 500)


def test_main_position_blur_bg_below_video():
    s = load_config()["subs"]
    align, margin = subtitles.main_position("blur_bg", 1920, 1012, s)
    assert align == 8 and margin == (1920 - 570) // 2 + 570 + s["below_video_gap"]  # video 1080x570, ortada
    assert subtitles.main_position("crop", 1920, 1012, s) == (2, s["margin_v"])


def test_encode_args():
    from clipper import render
    cfg = load_config()
    nv = render.encode_args(cfg, "h264_nvenc")
    assert nv[nv.index("-cq") + 1] == "19" and "+faststart" in nv and "yuv420p" in nv and "192k" in nv
    x = render.encode_args(cfg, "libx264")
    assert x[x.index("-crf") + 1] == "18" and x[x.index("-preset") + 1] == "veryfast"

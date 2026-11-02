import json
import types

import pytest

from clipper import candidates, claude_client, laughter
from clipper.common import load_config


def test_extract_json_code_fence():
    assert claude_client.extract_json('tamam:\n```json\n{"a": [1]}\n```') == {"a": [1]}
    assert claude_client.extract_json('önce [1, 2] sonra') == [1, 2]
    with pytest.raises(ValueError):
        claude_client.extract_json("json yok")


def test_ask_json_retries_then_succeeds(monkeypatch):
    replies = iter(['{"n": "x"}', "çöp", '{"n": 4}'])
    prompts = []

    def fake_run(cmd, input, **kw):
        prompts.append(input)
        return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"result": next(replies)}))

    monkeypatch.setattr(claude_client.subprocess, "run", fake_run)
    cfg = load_config()
    cfg["claude"]["mode"] = "cli"
    data = claude_client.ask_json("soru", {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}, cfg)
    assert data == {"n": 4}
    assert len(prompts) == 3 and "GEÇERSİZ" in prompts[1]


def test_ask_json_gives_up(monkeypatch):
    monkeypatch.setattr(claude_client.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0, stderr="", stdout='{"result": "yok"}'))
    with pytest.raises(RuntimeError):
        claude_client.ask_json("soru", {"type": "object"}, load_config())


def test_segments_from_probs_merge_and_min():
    times = [i * 0.1 for i in range(30)]
    probs = [0.9] * 5 + [0.1] * 3 + [0.8] * 5 + [0.1] * 10 + [0.9] + [0.1] * 6
    segs = laughter.segments_from_probs(times, probs, 0.1, 0.5, 0.4, 0.4)
    assert len(segs) == 1  # ilk iki parça birleşti, tek karelik kahkaha atıldı
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 1.3


CFG = {"min_len": 20, "max_len": 58, "tail": 0.5, "top_n": 20, "consecutive_bonus": 1.0, "pause": 0.6}


def _words(n_sent, per=5, dur=1.0):
    """n_sent cümle, her biri `per` kelime, kelime başına `dur` sn; cümle sonunda nokta."""
    return [{"i": k, "start": k * dur, "end": k * dur + dur * 0.9, "word": f"w{k}" + ("." if k % per == per - 1 else "")}
            for k in range(n_sent * per)]


def test_sentence_starts_punct_and_pause():
    w = [{"start": 0, "end": 0.5, "word": "Hi."}, {"start": 0.6, "end": 1.0, "word": "So"},
         {"start": 1.1, "end": 1.4, "word": "yeah"}, {"start": 2.5, "end": 2.8, "word": "okay"}]
    assert candidates.sentence_starts(w, 0.6) == [0, 1, 3]  # nokta + 1.1 sn duraklama
    assert candidates.sentence_ends(w, 0.6) == [0, 2, 3]


def test_candidates_sentence_boundary_and_length():
    laughs = [{"start": 60.0, "end": 63.0, "prob": 0.8}]
    c = candidates.build(laughs, _words(20), CFG)
    assert len(c) == 1
    assert c[0]["start"] == 40.0  # cümle başı, 23.5 sn >= 20
    assert c[0]["end"] == 63.5
    assert c[0]["score"] == pytest.approx(2.4)


def test_candidates_merge_overlaps_with_bonus():
    laughs = [{"start": 60.0, "end": 62.0, "prob": 0.5}, {"start": 70.0, "end": 72.0, "prob": 0.5}]
    c = candidates.build(laughs, _words(20), CFG)
    assert len(c) == 1
    assert c[0]["laughs"] == 2 and c[0]["end"] == 72.5
    assert c[0]["end"] - c[0]["start"] <= 58
    assert c[0]["score"] == pytest.approx(2 * 2 * 0.5 + 1.0)


def test_candidates_never_start_mid_sentence():
    words = _words(30, per=7)
    starts = {words[i]["start"] for i in candidates.sentence_starts(words, 0.6)}
    laughs = [{"start": t, "end": t + 2, "prob": 0.7} for t in (50.0, 120.0, 180.0)]
    for cd in candidates.build(laughs, words, CFG):
        assert cd["start"] in starts


def test_comedian_from_filename():
    from clipper.pipeline import comedian_from_filename as cf
    assert cf("samples/zoltan_kaszas__london_fog.mp4") == "Zoltan Kaszas"
    assert cf("İlker_ayrık__gosteri.mkv") == "İlker Ayrık"
    assert cf("ismo.mp4") == "Ismo"


def test_video_meta_show_from_meta_or_filename(tmp_path, monkeypatch):
    from clipper import common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    assert common.video_meta("samples/zoltan_kaszas__london_fog.mp4") == ("Zoltan Kaszas", "London Fog")
    assert common.video_meta("samples/ismo_hello.mp4") == ("Ismo Hello", "")  # "__" yoksa adın tamamı komedyen
    common.save_meta("samples/ismo_hello.mp4", "ISMO", "Hello")
    assert common.video_meta("samples/ismo_hello.mp4") == ("ISMO", "Hello")
    common.save_meta("samples/ismo_hello.mp4", "ISMO")  # gösteri verilmezse korunur
    assert common.video_meta("samples/ismo_hello.mp4") == ("ISMO", "Hello")

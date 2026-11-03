import pytest

from clipper import censor, proofread, subtitles
from clipper.common import load_config

ON = {"enabled": True, "mode": "full", "include_mild": False}


@pytest.mark.parametrize("text", ["Seni eve götürdüm.", "ama", "amcam geldi", "Eski bir sikke", "Amaç bu, ambar, amele",
                                  "Pikap aldım", "Boksuz bok!", "Lanet olsun", "Sıkıcı bir gün", "Fin'e dönüp"])
def test_clean_words_stay(text):
    assert censor.apply(text, ON) == text


@pytest.mark.parametrize("text, want", [("Ananı siktim.", "Ananı ******."), ("Amına koyayım", "***** koyayım"),
                                        ("götüne sokarım", "****** sokarım"), ("Siktir git", "****** git"),
                                        ("Orospu çocuğu", "****** çocuğu"), ("piç gibi", "*** gibi"), ("AMK", "***"), ("Ben tam bir amcığım.", "Ben tam bir *******."),
                                        ("Hassiktir! Kanka", "*********! Kanka")])
def test_profanity_masked_whole_word(text, want):
    assert censor.apply(text, ON) == want


def test_modes_mild_overrides_and_disabled():
    assert censor.apply("siktim", {**ON, "mode": "first_letter"}) == "s*****"
    assert censor.apply("Boksuz bok", {**ON, "include_mild": True}) == "****** ***"
    assert censor.apply("Sen bir kerizsin", ON, extra={"kerizsin"}) == "Sen bir ********"
    assert censor.apply("siktim", ON, allow={"siktim"}) == "siktim"
    assert censor.apply("siktim", {**ON, "enabled": False}) == "siktim"
    assert censor.found(["Ananı siktim", "götüne, siktim"], ON) == ["siktim", "götüne"]


def test_karaoke_timing_unchanged_by_censor():
    lines = ["söyleyebileceğin en kötü şey:", "Ananı siktim."]
    raw, cen = subtitles.karaoke(lines, 290), subtitles.karaoke([censor.apply(l, ON) for l in lines], 290)
    assert "******" in cen and [p.split("}")[0] for p in raw.split("{")] == [p.split("}")[0] for p in cen.split("{")]


def test_build_ass_censors_hook_notes_and_lines():
    cfg = load_config()
    bit = {"n": 1, "start": 0, "end": 10, "hook_tr": "Siktir dedi"}
    tr = {"groups": [{"start": 0, "end": 2, "lines": ["Ananı siktim."]}], "notes": [{"start": 1, "text": "amk = küfür"}]}
    ass = subtitles.build_ass(bit, tr, "ISMO", cfg["subs"], cfg["safe_zone"], cen=lambda t: censor.apply(t, ON))
    assert "siktim" not in ass.lower() and "Siktir" not in ass and "*** = küfür" in ass and "******" in ass


def test_proofread_returns_profanity_present_in_lines(monkeypatch):
    tr = {"groups": [{"start": 1, "end": 2, "en": "I fucked your mom", "lines": ["Ananı siktim"]}], "notes": [], "replacements": []}
    monkeypatch.setattr(proofread, "ask_json", lambda *a, **k: {"groups": [["Ananı siktim"]], "notes": [], "changes": [],
                                                                "profanity": [["siktim", "uydurma"]]})
    out = proofread.proofread_bit({"n": 1}, tr, load_config())
    assert out["profanity"] == [["siktim"]] and out["groups"][0]["lines"] == ["Ananı siktim"]

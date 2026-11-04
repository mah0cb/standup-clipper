"""Takvim kuralları: saf mantık, sabit örnek havuzlar."""
from datetime import date, datetime, timedelta

import pytest

from clipper import schedule as S

ST = S.Settings(["12:30", "21:00"], date(2026, 1, 1), timedelta(minutes=60), set())
LATE = datetime(2026, 2, 1, 8, 0)  # hesabın ilk günleri çoktan geçti


def clip(name, who="a", score=8, topic=None, flagged=False, kisa=False, shares=None):
    return {"path": name, "who": who, "comedian": who.upper(), "hook": f"hook {name}", "text": f"açıklama {name}", "score": score,
            "topic": topic or f"#{name}", "flagged": flagged, "kufur": flagged, "yetiskin": False, "kisa": kisa,
            "duration": 30 if kisa else 60, "shares": shares or {}}


def pool(**counts):
    return [clip(f"{who}{i}", who) for who, n in counts.items() for i in range(n)]


def build(clips, now=LATE, st=ST, data=None, start=None):
    return S.rebuild(data or {"entries": []}, clips, st, start or now.date().isoformat(), now)


def order(data, clips):
    by = {c["path"]: c for c in clips}
    return [(e["at"], by[e["path"]]) for e in data["entries"]]


def by_day(data, clips):
    days = {}
    for at, c in order(data, clips):
        days.setdefault(at[:10], []).append(c)
    return list(days.values())


def test_two_comedians_swap_slots_every_day():
    clips = pool(a=6, b=6)
    days = by_day(build(clips), clips)
    pairs = ["".join(c["who"] for c in d) for d in days]
    assert len(pairs) == 6 and all(p in ("ab", "ba") for p in pairs)
    assert all(x != y for x, y in zip(pairs, pairs[1:]))  # A-B, B-A, A-B...


def test_three_comedians_rotate_without_any_repeat():
    clips = pool(a=4, b=4, c=4)
    seq = [c["who"] for _, c in order(build(clips), clips)]
    assert all(x != y for x, y in zip(seq, seq[1:]))
    days = ["".join(c["who"] for c in d) for d in by_day(build(clips), clips)]
    assert all(x[0] != y[0] and x[1] != y[1] for x, y in zip(days, days[1:]))  # her gün saatler değişir


def test_single_comedian_left_may_repeat():
    clips = pool(a=5, b=1)
    data = build(clips)
    assert len(data["entries"]) == 6 and [c["who"] for _, c in order(data, clips)].count("b") == 1


def test_topic_needs_two_posts_between():
    clips = [clip(f"a{i}", "a", topic="#uçak" if i < 2 else None) for i in range(5)] + \
            [clip(f"b{i}", "b", topic="#uçak" if i < 2 else None) for i in range(5)]
    topics = [c["topic"] for _, c in order(build(clips), clips)]
    idx = [i for i, t in enumerate(topics) if t == "#uçak"]
    assert len(idx) == 4 and all(y - x >= 3 for x, y in zip(idx, idx[1:]))


def test_flagged_only_evening_and_after_day_five():
    clips = pool(a=4, b=4) + [clip(f"f{i}", "ab"[i % 2], flagged=True) for i in range(4)]
    data = build(clips, now=datetime(2026, 1, 1, 8))
    flagged = [at for at, c in order(data, clips) if c["flagged"]]
    assert len(flagged) == 4 and all(at.endswith("21:00") and at >= "2026-01-06" for at in flagged)


def test_only_flagged_left_waits_for_evening_after_restricted_days():
    clips = [clip("f", flagged=True)]
    assert build(clips, now=datetime(2026, 1, 1, 8))["entries"][0]["at"] == "2026-01-06T21:00"


def test_pressure_prevents_empty_noons_at_the_end():
    # 4 temiz + 4 işaretli: işaretliler her akşam gitmezse sonda öğleler boş kalırdı
    clips = [clip(f"t{i}", "ab"[i % 2]) for i in range(4)] + [clip(f"f{i}", "ab"[i % 2], flagged=True) for i in range(4)]
    days = by_day(build(clips), clips)
    assert [len(d) for d in days] == [2, 2, 2, 2] and all(not d[0]["flagged"] and d[1]["flagged"] for d in days)


def test_first_three_days_prefer_short_high_score_clean():
    long_top = [clip(f"l{i}", "ab"[i % 2], score=10) for i in range(6)]
    short = [clip(f"s{i}", "ab"[i % 2], score=9 - i // 2, kisa=True) for i in range(6)]
    filler = [clip(f"f{i}", "ab"[i % 2], flagged=True, score=10) for i in range(2)]
    clips = long_top + short + filler
    data = build(clips, now=datetime(2026, 1, 1, 8))
    first = [c for at, c in order(data, clips) if at < "2026-01-04"]
    assert len(first) == 6 and all(c["kisa"] for c in first)
    assert first[0]["score"] == 9 and first[-1]["score"] == 7  # puan sırasıyla


def test_high_scores_spread_evenly():
    clips = [clip(f"a{i}", "a", score=10 if i < 3 else 6) for i in range(9)] + [clip(f"b{i}", "b", score=10 if i < 3 else 6) for i in range(9)]
    days = by_day(build(clips), clips)
    totals = [sum(c["score"] for c in d) for d in days]
    assert max(totals) - min(totals) <= 4  # 20'lik gün yok
    tops = [i for i, (_, c) in enumerate(order(build(clips), clips)) if c["score"] == 10]
    assert all(y - x >= 2 for x, y in zip(tops, tops[1:])) and tops[-1] - tops[0] >= 9  # takvime yayılmış


def test_rebuild_keeps_pins_and_history_skips_shared():
    clips = pool(a=4, b=4) + [clip("x", "a", shares={"tiktok": "2026-01-20"})]
    data = build(clips)
    assert "x" not in [e["path"] for e in data["entries"]]
    target = data["entries"][-1]["at"]
    S.move(data, ST, data["entries"][0]["path"], target, LATE)
    pinned = next(e for e in data["entries"] if e["pinned"])
    again = build(clips, data=data)
    assert {"path": pinned["path"], "at": target, "pinned": True} in again["entries"]
    assert sorted(e["path"] for e in again["entries"]) == sorted(c["path"] for c in clips if not c["shares"])


def test_shared_history_counts_as_previous_post():
    clips = pool(a=2, b=2) + [clip("x", "b", shares={"tiktok": "2026-01-31"})]
    assert build(clips)["entries"][0]["path"].startswith("a")


def test_missed_slot_shifts_forward_and_reports():
    clips = pool(a=3, b=3)
    data = build(clips)
    first = data["entries"][0]
    later = datetime(2026, 2, 1, 15, 0)  # 12:30 + 60 dk geçti, paylaşılmadı
    out = S.sync(data, clips, ST, later)
    assert all(e["at"] > "2026-02-01T15:00" for e in out["entries"])
    assert len(out["entries"]) == 6
    m = out["moved"][0]
    assert m["path"] == first["path"] and m["from"] == first["at"] and m["to"] > first["at"] and m["on"] == "2026-02-01"
    assert m["to"] == next(e["at"] for e in out["entries"] if e["path"] == first["path"])


def test_grace_period_keeps_slot():
    clips = pool(a=2, b=2)
    data = build(clips)
    assert S.sync(data, clips, ST, datetime(2026, 2, 1, 13, 0))["entries"] == data["entries"]


def test_shared_after_slot_is_not_missed():
    clips = pool(a=2, b=2)
    data = build(clips)
    next(c for c in clips if c["path"] == data["entries"][0]["path"])["shares"] = {"instagram": "2026-02-01"}
    out = S.sync(data, clips, ST, datetime(2026, 2, 1, 18, 0))
    assert out["entries"] == data["entries"] and not out["moved"]


def test_missed_keeps_pins_in_place_and_flagged_stays_in_evening():
    clips = pool(a=3, b=3) + [clip("f", "a", flagged=True)]
    data = build(clips)
    pinned_at = data["entries"][4]["at"]
    S.pin(data, data["entries"][4]["path"], True)
    pinned_path = data["entries"][4]["path"]
    out = S.sync(data, clips, ST, datetime(2026, 2, 1, 15, 0))
    assert {"path": pinned_path, "at": pinned_at, "pinned": True} in out["entries"]
    assert next(e for e in out["entries"] if e["path"] == "f")["at"].endswith("21:00")


def test_sync_without_changes_keeps_manual_order():
    clips = pool(a=3, b=3)
    data = build(clips)
    S.move(data, ST, data["entries"][0]["path"], data["entries"][1]["at"], LATE)  # kurala aykırı bile olsa dokunulmaz
    assert S.sync(data, clips, ST, LATE)["entries"] == data["entries"]


def test_new_clip_is_appended_and_deleted_clip_leaves():
    clips = pool(a=2, b=2)
    data = build(clips)
    gone = data["entries"][1]["path"]
    rest = [c for c in clips if c["path"] != gone] + [clip("yeni", "b")]
    out = S.sync(data, rest, ST, LATE)
    paths = [e["path"] for e in out["entries"]]
    assert gone not in paths and "yeni" in paths and len(paths) == 4


def test_move_swaps_and_validates():
    clips = pool(a=2, b=2)
    data = build(clips)
    (p0, at0), (p1, at1) = [(e["path"], e["at"]) for e in data["entries"][:2]]
    S.move(data, ST, p0, at1, LATE)
    got = {e["path"]: e for e in data["entries"]}
    assert got[p0]["at"] == at1 and got[p0]["pinned"] and got[p1]["at"] == at0 and not got[p1]["pinned"]
    for bad in ("2026-02-05T13:00", "yarın", "2026-01-01T12:30"):
        with pytest.raises(S.ScheduleError):
            S.move(data, ST, p0, bad, LATE)
    with pytest.raises(S.ScheduleError):
        S.move(data, ST, "yok", at0, LATE)


def test_view_stock_warning_flags_and_empty_slots():
    clips = pool(a=2, b=2) + [clip("f", "a", flagged=True)]
    data = build(clips, now=datetime(2026, 1, 1, 8))
    v = S.view(data, clips, ST, datetime(2026, 1, 1, 8))
    assert v["days_left"] == 3 and len(v["today"]) == 2  # 1, 2 Ocak + 6 Ocak
    assert any(s["clip"] is None for d in v["days"] for s in d["slots"])  # işaretli klibi bekleyen boş slotlar
    data["entries"][0]["path"] = "f"  # elle kural dışı yere konmuş gibi
    assert S.view(data, clips, ST, datetime(2026, 1, 1, 8))["today"][0]["warn"]


def test_ics_escapes_folds_and_has_alarm():
    long = "Bu çok uzun bir açıklama; virgül, emoji 😂 ve ters\\bölü içerir\n\n#etiket " * 4
    clips = [clip("a"), clip("b", "b"), clip("c", "b", shares={"tiktok": "2026-01-31"})]
    clips[0]["text"] = long
    text = S.to_ics(build(clips), clips, ST, LATE)
    raw = text.encode("utf-8")
    assert all(len(l) <= 75 for l in raw.split(b"\r\n"))  # bayt sınırı; decode harf bölünmediğini de kanıtlar
    assert all(l.decode("utf-8") is not None for l in raw.split(b"\r\n"))
    unfolded = text.replace("\r\n ", "")
    assert unfolded.count("BEGIN:VEVENT") == 2 and unfolded.count("TRIGGER:-PT15M") == 2  # paylaşılan yok
    assert "açıklama\\; virgül\\, emoji 😂 ve ters\\\\bölü içerir\\n\\n#etiket" in unfolded
    assert "DTSTART:20260201T123000" in unfolded and "SUMMARY:A: hook a" in unfolded


def test_adult_word_list():
    assert S.is_adult("sekse") and S.is_adult("Seks'e") and S.is_adult("SEX") and S.is_adult("cinsellik")
    assert not S.is_adult("seksen") and not S.is_adult("seksenlerde") and not S.is_adult("memeli") and not S.is_adult("kahve")


@pytest.mark.parametrize("slots", [[1230], ["25:00"], []])
def test_bad_slots_rejected(slots):
    with pytest.raises(S.ScheduleError):
        S.settings({"schedule": {"slots": slots, "account_start_date": "2026-11-02"}})

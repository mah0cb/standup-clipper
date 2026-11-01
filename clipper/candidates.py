"""4) Kahkahalardan geriye giderek aday klipler üretir → candidates.json."""
import re

from . import laughter, transcribe
from .common import cached, work_dir

END_RE = re.compile(r"[.?!][\"')\]]*$")


def sentence_starts(words, pause):
    """Cümle başı sayılan kelime indeksleri: noktalamadan sonra veya `pause` sn'den uzun duraklamadan sonra."""
    return [i for i, w in enumerate(words)
            if i == 0 or END_RE.search(words[i - 1]["word"]) or w["start"] - words[i - 1]["end"] > pause]


def sentence_ends(words, pause):
    starts = sentence_starts(words, pause)
    return [s - 1 for s in starts[1:]] + ([len(words) - 1] if words else [])


def _score(start, end, laughs, bonus):
    inside = [l for l in laughs if l["start"] >= start and l["end"] <= end + 0.01]
    return round(sum((l["end"] - l["start"]) * l["prob"] for l in inside) + bonus * max(0, len(inside) - 1), 3), len(inside)


def build(laughs, words, c):
    """Saf fonksiyon: kahkaha + kelime listesinden en iyi top_n adayı döndürür."""
    lo, hi = c["min_len"], c["max_len"]
    starts = [words[i]["start"] for i in sentence_starts(words, c["pause"])]
    raw = []
    for L in laughs:
        end = L["end"] + c["tail"]
        start = None
        # Kahkahadan önceki cümle başlarında geriye yürü; en az min_len, en fazla max_len
        for s in sorted((s for s in starts if s < L["start"]), reverse=True):
            if end - s > hi:
                break
            start = s
            if end - start >= lo:
                break
        if start is None:
            continue  # kahkahadan önce konuşma yok (ör. gösteri başı alkışı)
        raw.append({"start": round(start, 2), "end": round(end, 2)})

    # Çakışanları birleştir; birleşim çok uzunsa daha yüksek skorlu kalır
    merged = []
    for cnd in sorted(raw, key=lambda x: x["start"]):
        cnd["score"], cnd["laughs"] = _score(cnd["start"], cnd["end"], laughs, c["consecutive_bonus"])
        if merged and cnd["start"] < merged[-1]["end"]:
            prev = merged[-1]
            s, e = prev["start"], max(prev["end"], cnd["end"])
            if e - s <= hi:
                prev["end"] = e
                prev["score"], prev["laughs"] = _score(s, e, laughs, c["consecutive_bonus"])
            elif cnd["score"] > prev["score"]:
                merged[-1] = cnd
            continue
        merged.append(cnd)

    best = sorted(merged, key=lambda x: -x["score"])[:c["top_n"]]
    return [{"id": i, **b} for i, b in enumerate(sorted(best, key=lambda x: x["start"]))]


def run(video, cfg, force=False):
    def make():
        _, words = transcribe.run(video, cfg)
        return build(laughter.run(video, cfg), words, cfg["candidates"])
    return cached(work_dir(video) / "candidates.json", make, force)

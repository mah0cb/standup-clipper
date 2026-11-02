"""5) Claude ile en iyi bitleri seçer → selected.json.

Claude klip başını/sonunu cümle sınırlarındaki kelime indeksleriyle seçer; zamanlar koddan hesaplanır.
"""
import wave

from . import audio, candidates, laughter, transcribe
from .claude_client import ask_json
from .common import cached, log, work_dir

PROMPT = """Bir stand-up gösterisinden kahkaha tespitiyle bulunmuş aday bölümler var. Türk izleyici için
Instagram/TikTok/Shorts dikey kısa videoları üreteceğiz (Türkçe altyazılı).

Görevin: en iyi bitleri seç. Hedef {max_bits} bit (gösterinin her 5 dakikası için 1), bu bir ÜST SINIRDIR.
Puanı {min_score} veya üstü olan her iyi biti seç; yalnızca gerçekten zayıf (puanı {min_score}'ın altında) adayları ele.
Bu yüzden {max_bits}'den az bit dönebilir, ama iyi bitleri gereksiz yere eleme.

Seçim kuralları:
- Bit tek başına, önceki bağlamı bilmeden anlaşılır olmalı.
- Kurulum (set-up) cümlesi klibin İÇİNDE olmak zorunda. Gerekirse başlangıcı BAĞLAM kısmına kaydır.
- Klip asla cümle ortasından başlamaz/bitmez: start_word yalnızca satır başındaki ilk indeks (#BAŞ-SON'daki BAŞ),
  end_word yalnızca satır sonundaki indeks (SON) olabilir.
- Punchline'dan sonraki kahkaha klipte kalsın: end_word punchline cümlesinin SON'u olsun (kahkaha kod tarafından eklenir).
- Klip uzunluğu {min_len}-{max_len} sn. 30-60 saniyelik bitleri tercih et; 60 saniyeyi ancak kurulum (set-up) gerektiriyorsa aş, en fazla {max_len}.
- Türk izleyiciye geçmeli; Amerikan siyaseti veya çevrilemez kelime oyunu üzerine kurulu olmamalı.
- Sadece alkış, teşekkür, selamlama, gösteriye giriş/çıkış, şarkı veya seyirciyle boş muhabbet olan bölümleri ASLA seçme.
- Seçilen klipler birbiriyle çakışmamalı.

Her seçim için:
- id: adayın id'si
- start_word, end_word: yukarıdaki kurallara göre kelime indeksleri
- score: 1-10 komiklik/viral potansiyel puanı
- reason: kısa Türkçe gerekçe
- hook_tr: ilk 3 saniyede ekranda yazacak, kısa (en fazla ~40 karakter), merak uyandıran Türkçe başlık; espriyi ele vermesin
- description_tr: paylaşım için 1-2 cümlelik Türkçe açıklama
- hashtags: 5 hashtag (# ile, Türkçe ağırlıklı)

Satır biçimi: "#BAŞ-SON [zaman sn] cümle". [KAHKAHA] satırları seyirci kahkahasıdır.

ADAYLAR:
{cands}
"""

SCHEMA = {
    "type": "object",
    "required": ["bits"],
    "properties": {"bits": {"type": "array", "items": {
        "type": "object",
        "required": ["id", "start_word", "end_word", "score", "reason", "hook_tr", "description_tr", "hashtags"],
        "properties": {
            "id": {"type": "integer"}, "start_word": {"type": "integer"}, "end_word": {"type": "integer"},
            "score": {"type": "integer", "minimum": 1, "maximum": 10},
            "reason": {"type": "string"}, "hook_tr": {"type": "string", "minLength": 1},
            "description_tr": {"type": "string"},
            "hashtags": {"type": "array", "minItems": 5, "maxItems": 5, "items": {"type": "string"}},
        }}}},
}


def max_bits(duration, s):
    return min(s["max_bits"], max(1, int(duration // (s["per_minutes"] * 60))))


def clip_end(words, end_word, laughs, tail):
    """Punchline'dan hemen sonra başlayan kahkahayı klibe dahil eder; sonraki konuşmaya taşmaz."""
    w_end = words[end_word]["end"]
    end = max([w_end] + [l["end"] for l in laughs if w_end - 2 <= l["start"] <= w_end + 1.5]) + tail
    if end_word + 1 < len(words):
        end = max(min(end, words[end_word + 1]["start"]), w_end + tail)
    return round(end, 2)


def candidate_block(cd, words, laughs, sents, ctx):
    lo = cd["start"] - ctx
    out = [f'== ADAY id={cd["id"]} (kahkaha skoru {cd["score"]}, {cd["laughs"]} kahkaha)']
    shown = [(a, b) for a, b in sents if words[b]["end"] > lo and words[a]["start"] < cd["end"]]
    marked = False
    for k, (a, b) in enumerate(shown):
        if not marked and words[a]["start"] >= cd["start"] - 0.01:
            out.append("-- aday buradan başlıyor --")
            marked = True
        text = " ".join(w["word"] for w in words[a:b + 1])
        out.append(f'#{a}-{b} [{words[a]["start"]:.1f}] {text}')
        nxt = words[shown[k + 1][0]]["start"] if k + 1 < len(shown) else cd["end"]
        for l in laughs:
            if words[a]["start"] <= l["start"] < nxt:
                out.append(f'[KAHKAHA {l["end"] - l["start"]:.1f} sn]')
    return "\n".join(out)


def make_check(cands, words, laughs, starts, ends, s, c, n_max):
    ids = {cd["id"] for cd in cands}

    def check(d):
        bits = [b for b in d["bits"] if b["score"] >= s["min_score"]]
        if len(bits) < len(d["bits"]):
            log.info("[seçim] %d bit %d puan altında olduğu için elendi", len(d["bits"]) - len(bits), s["min_score"])
        if len(bits) > n_max:
            raise ValueError(f"en fazla {n_max} bit seçilebilir, {len(bits)} geldi")
        spans = []
        for b in bits:
            if b["id"] not in ids:
                raise ValueError(f"geçersiz id {b['id']}")
            if b["start_word"] not in starts:
                raise ValueError(f"id={b['id']}: start_word={b['start_word']} bir cümle başı değil (cümle ortasından başlıyor)")
            if b["end_word"] not in ends or b["end_word"] <= b["start_word"]:
                raise ValueError(f"id={b['id']}: end_word={b['end_word']} bir cümle sonu değil")
            b["start"] = words[b["start_word"]]["start"]
            b["end"] = clip_end(words, b["end_word"], laughs, c["tail"])
            dur = b["end"] - b["start"]
            if not c["min_len"] - 2 <= dur <= c["max_len"] + 2:
                raise ValueError(f"id={b['id']}: uzunluk {dur:.1f} sn, {c['min_len']}-{c['max_len']} sn olmalı")
            spans.append((b["start"], b["end"], b["id"]))
        spans.sort()
        for (s1, e1, i1), (s2, _, i2) in zip(spans, spans[1:]):
            if s2 < e1:
                raise ValueError(f"id={i1} ve id={i2} klipleri çakışıyor")
        d["bits"] = bits
    return check


def run(video, cfg, force=False):
    def make():
        _, words = transcribe.run(video, cfg)
        laughs = laughter.run(video, cfg)
        cands = candidates.run(video, cfg)
        if not cands:
            return []
        c, s = cfg["candidates"], cfg["select"]
        with wave.open(str(audio.run(video, cfg))) as w:
            duration = w.getnframes() / w.getframerate()
        n_max = max_bits(duration, s)
        starts = candidates.sentence_starts(words, c["pause"])
        ends = candidates.sentence_ends(words, c["pause"])
        sents = list(zip(starts, ends))
        blocks = "\n\n".join(candidate_block(cd, words, laughs, sents, s["context_before"]) for cd in cands)
        prompt = PROMPT.format(max_bits=n_max, min_score=s["min_score"], min_len=c["min_len"],
                               max_len=c["max_len"], cands=blocks)
        check = make_check(cands, words, laughs, set(starts), set(ends), s, c, n_max)
        bits = ask_json(prompt, SCHEMA, cfg, check, label="seçim")["bits"]
        bits.sort(key=lambda b: -b["score"])
        return [{"n": i + 1, **b} for i, b in enumerate(bits)]

    return cached(work_dir(video) / "selected.json", make, force)

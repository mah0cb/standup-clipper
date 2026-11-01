"""6) Seçilen bitleri Türkçeye çevirir, ifade gruplarını kelime zamanlarına hizalar → translations.json."""
from concurrent.futures import ThreadPoolExecutor

from . import select_llm, transcribe
from .claude_client import ask_json
from .common import cached, work_dir

SYSTEM = """Sen stand-up komedisini Türkçe altyazıya çeviren deneyimli bir çevirmensin. Kuralların:
- Mizahı koru. Punchline kelimesi/kavramı Türkçe satırın SONUNA gelsin.
- Türkçe doğal konuşma dili olsun; kelimesi kelimesine çeviri yapma. Bir Türk komedyen sahnede nasıl söylerdi, öyle yaz.
- Küfür şiddeti orijinalle birebir aynı olsun. Orijinalde olmayan küfrü ASLA ekleme. "fuck" yoksa Türkçede de ağır küfür olmaz;
  hafif argo ("shit", "damn") hafif karşılıkla çevrilir. Argoyu da yumuşatma ya da abartma.
- Altyazıda ASLA parantez kullanma.
- Çevrilemeyen kelime oyununa Türkçe eşdeğer oyun bul.
- Esprinin anlaşılması için ek bilgi gerekiyorsa (ör. CPR = kalp masajı, U.P. "you pee" gibi okunuyor), bunu altyazıya
  YAZMA; "notes" alanına yaz. Not, ilgili kelime söylendiğinde ekranın üstünde kısa süre görünür. Not en fazla {note_max}
  karakter, kısa ve net olsun (ör. "CPR = kalp masajı", "U.P. okunuşu: 'yu pii' = işiyorsun"). Gerek yoksa not ekleme.
- Türk izleyiciye anlamsız gelecek ABD referanslarını (marka, ünlü, kurum, yer) en yakın Türk karşılığıyla değiştir ve
  her değişikliği "replacements" listesinde raporla. Esprinin konusu olan referansı değiştirme; onun için not kullan."""

PROMPT = """Aşağıdaki stand-up bitini Türkçe altyazıya çevir.

- Çeviriyi ekranda birlikte görünecek ifade gruplarına böl. Her grup ardışık bir İngilizce kelime aralığını
  (word_from..word_to, dahil) karşılar; gruplar sırayla, boşluksuz ve çakışmadan 0'dan {last}'e kadar tüm kelimeleri kapsar.
- Her grup en fazla {max_lines} satır, her satır en fazla {max_chars} karakter. Kısa ve okunur tut.
- Her not için "word": notun görüneceği İngilizce kelimenin indeksi.

İNGİLİZCE (indeks:kelime [başlangıç sn]):
{words}
"""


def schema(cfg):
    s = cfg["subs"]
    return {
        "type": "object", "required": ["groups", "notes", "replacements"],
        "properties": {
            "groups": {"type": "array", "minItems": 1, "items": {
                "type": "object", "required": ["word_from", "word_to", "lines"],
                "properties": {
                    "word_from": {"type": "integer"}, "word_to": {"type": "integer"},
                    "lines": {"type": "array", "minItems": 1, "maxItems": s["max_lines"],
                              "items": {"type": "string", "minLength": 1, "maxLength": s["max_chars"]}}}}},
            "notes": {"type": "array", "items": {
                "type": "object", "required": ["word", "text"],
                "properties": {"word": {"type": "integer"},
                               "text": {"type": "string", "minLength": 1, "maxLength": s["note_max_chars"]}}}},
            "replacements": {"type": "array", "items": {
                "type": "object", "required": ["orig", "tr"],
                "properties": {"orig": {"type": "string"}, "tr": {"type": "string"}, "note": {"type": "string"}}}},
        },
    }


def check_groups(groups, n_words):
    """Grupların 0..n_words-1 aralığını sırayla, boşluksuz kapsadığını doğrular."""
    expect = 0
    for g in groups:
        if g["word_from"] != expect or g["word_to"] < g["word_from"]:
            raise ValueError(f"grup {g['word_from']}-{g['word_to']}: {expect}. kelimeden başlamalıydı")
        expect = g["word_to"] + 1
    if expect != n_words:
        raise ValueError(f"gruplar {expect - 1}. kelimede bitiyor, son kelime {n_words - 1}")


def check_text(d, n_words):
    """Parantez yasağı ve not indeksleri."""
    for g in d["groups"]:
        for line in g["lines"]:
            if any(ch in line for ch in "()[]"):
                raise ValueError(f"altyazıda parantez kullanılamaz: {line!r} (açıklamayı notes alanına taşı)")
    for nt in d["notes"]:
        if not 0 <= nt["word"] < n_words:
            raise ValueError(f"not kelime indeksi geçersiz: {nt['word']}")
        if any(ch in nt["text"] for ch in "()[]"):
            raise ValueError(f"notta parantez kullanılamaz: {nt['text']!r}")


def align(groups, words):
    """Kelime indekslerini mutlak zamanlara çevirir (LLM zaman uydurmaz)."""
    return [{"start": words[g["word_from"]]["start"], "end": words[g["word_to"]]["end"], "lines": g["lines"],
             "en": " ".join(w["word"] for w in words[g["word_from"]:g["word_to"] + 1])} for g in groups]


def bit_words(bit, all_words):
    return all_words[bit["start_word"]:bit["end_word"] + 1]


def translate_bit(bit, all_words, cfg):
    words = bit_words(bit, all_words)
    s = cfg["subs"]
    prompt = PROMPT.format(last=len(words) - 1, max_lines=s["max_lines"], max_chars=s["max_chars"],
                           words="\n".join(f'{i}:{w["word"]} [{w["start"]}]' for i, w in enumerate(words)))

    def check(d):
        check_groups(d["groups"], len(words))
        check_text(d, len(words))

    d = ask_json(prompt, schema(cfg), cfg, check, label=f"çeviri #{bit['n']}",
                 system=SYSTEM.format(note_max=s["note_max_chars"]))
    return {"groups": align(d["groups"], words),
            "notes": [{"start": words[nt["word"]]["start"], "text": nt["text"]} for nt in d["notes"]],
            "replacements": d["replacements"]}


def run(video, cfg, force=False):
    def make():
        _, words = transcribe.run(video, cfg)
        bits = select_llm.run(video, cfg)
        with ThreadPoolExecutor(cfg["claude"]["parallel"]) as ex:
            res = list(ex.map(lambda b: translate_bit(b, words, cfg), bits))
        return {str(b["n"]): r for b, r in zip(bits, res)}
    return cached(work_dir(video) / "translations.json", make, force)

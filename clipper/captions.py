"""6c) Paylaşım açıklamaları: çeviri kontrolünden sonra tek Claude çağrısıyla her bitin açıklaması → captions.json
({n: metin}). Claude yalnızca 1. satırı (hook) ve konu etiketini yazar; kredi satırı ve sabit etiketleri kod ekler.
Mevcut out/ kliplerinin .txt dosyaları da güncellenir (video yeniden render edilmez)."""
import re

from . import censor, proofread, select_llm
from .claude_client import ask_json
from .common import ROOT, cached, load_config, video_meta, work_dir

SYSTEM = """Sen Türk izleyici için stand-up kısa videolarına (Reels/TikTok/Shorts) paylaşım açıklaması yazan bir editörsün.
Her klip için iki şey yazarsın:

1) hook: açıklamanın ilk satırı. Akışta görünen tek kısım bu.
- Merak uyandıran tek cümle, tercihen soru. En fazla 80 karakter.
- Esprinin vuruşunu (punchline) ASLA ele verme; izleyen sonucu klipte öğrenmeli.
- Kendi başına anlaşılır olsun: klibi görmemiş biri okuyunca neyle ilgili olduğunu anlamalı.
  "Ama", "ve", "çünkü" gibi bağlaçlarla ya da yarım cümleyle başlama.
- Ekrandaki hook başlığını aynen ya da neredeyse aynen tekrarlama; başka bir açıdan merak uyandır.
- En fazla 1 emoji (isteğe bağlı).
- Doğal, günlük Türkçe. Reklam dili yok: "Kaçırma!", "Mutlaka izle", "Sonuna kadar izle" gibi kalıplar yasak.

2) topic: klibin konusunu anlatan TEK hashtag (ör. #evlilik, #ingilizce, #uçak, #babalık).
- Klibin içeriğine gerçekten uymalı; metne bakarak seç, komedyenin genel tarzına göre değil.
- #karamizah yalnızca klip gerçekten ölüm, hastalık, felaket gibi karanlık bir konu üzerine kuruluysa.
- Küçük harf, tek kelime, boşluksuz. #standup, #komedi, #mizah, #komik, #keşfet gibi genel etiketler yasak
  (sabit etiketler zaten ekleniyor).

Açıklamanın tamamı şöyle görünür (sen yalnızca ilk satırı ve son etiketi yazıyorsun):
Bir kısaltmayı bilmemek hayatına mal olabilir mi? 😅

🎤 ISMO – Hello

#standup #standupkomedi #türkçealtyazı #ismo #ingilizce"""

PROMPT = """Komedyen: {comedian}
Aşağıdaki {n} klibin her biri için hook ve topic yaz. "n" klibin numarasıdır.

{clips}
"""

SCHEMA = {
    "type": "object", "required": ["captions"],
    "properties": {"captions": {"type": "array", "items": {
        "type": "object", "required": ["n", "hook", "topic"],
        "properties": {"n": {"type": "integer"}, "hook": {"type": "string", "minLength": 5, "maxLength": 80},
                       "topic": {"type": "string"}}}}},
}
FIXED_TAGS = ["#standup", "#standupkomedi", "#türkçealtyazı"]
GENERIC = {"#standup", "#standupkomedi", "#türkçealtyazı", "#komedi", "#mizah", "#komik", "#keşfet", "#kesfet",
           "#fyp", "#viral", "#reels", "#shorts", "#altyazı", "#türkçe"}
BAD_START = {"ama", "ve", "çünkü", "fakat", "ancak", "yani", "ya", "de", "da"}
ADS = ("kaçırma", "mutlaka izle", "sonuna kadar izle")
# ponytail: kaba emoji tespiti (yaygın emoji blokları); nadir semboller kaçabilir
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


def tag_of(name):
    """"Joe List" → "joelist", "İsmo" → "ismo": küçük harf, boşluksuz, Türkçe karakter yok."""
    s = name.translate(str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")).lower()
    return re.sub(r"[^a-z0-9]", "", s)


def compose(hook, comedian, show, topic):
    credit = f"🎤 {comedian} – {show}" if show else f"🎤 {comedian}"
    return f"{hook.strip()}\n\n{credit}\n\n{' '.join(FIXED_TAGS + ['#' + tag_of(comedian), topic])}"


def norm(s):
    return re.sub(r"[^\w]", "", s.lower())


def make_check(bits, comedian):
    want = {b["n"]: b for b in bits}

    def check(d):
        got = {c["n"] for c in d["captions"]}
        if got != set(want):
            raise ValueError(f"her klip için tam bir açıklama olmalı: beklenen {sorted(want)}, gelen {sorted(got)}")
        for c in d["captions"]:
            h, t, n = c["hook"].strip(), c["topic"].strip().lower(), c["n"]
            first = norm(h.split()[0]) if h.split() else ""
            if first in BAD_START:
                raise ValueError(f"n={n}: hook '{h.split()[0]}' ile başlıyor, kendi başına anlaşılır olmalı")
            if len(EMOJI.findall(h)) > 1:
                raise ValueError(f"n={n}: hook'ta en fazla 1 emoji olabilir")
            if "\n" in h or "#" in h:
                raise ValueError(f"n={n}: hook tek satır olmalı, hashtag içermemeli")
            if any(a in h.lower() for a in ADS):
                raise ValueError(f"n={n}: reklam kalıbı kullanma")
            if norm(h) == norm(want[n]["hook_tr"]):
                raise ValueError(f"n={n}: hook ekrandaki başlığın aynısı olmamalı")
            if not re.fullmatch(r"#[a-zçğıöşü0-9]+", t):
                raise ValueError(f"n={n}: topic tek, küçük harfli, boşluksuz bir hashtag olmalı (ör. #evlilik)")
            if t in GENERIC or t == "#" + tag_of(comedian):
                raise ValueError(f"n={n}: {t} genel/sabit bir etiket; klibin konusunu anlatan bir etiket seç")
            c["hook"], c["topic"] = h, t
    return check


def clip_block(b, tr):
    en = " ".join(g["en"] for g in tr["groups"])
    tr_text = " ".join(" ".join(g["lines"]) for g in tr["groups"])
    return f'== n={b["n"]} | ekrandaki hook başlığı: "{b["hook_tr"]}"\nEN: {en}\nTR: {tr_text}'


def txt_text(name, n, text, cfg):
    """.txt'ye yazılacak açıklama: captions.json orijinal kalır, sansür yalnız burada uygulanır."""
    return censor.for_bit(ROOT / "work" / name, n, cfg)(text) + "\n"


def write_txts(name, caps, cfg=None):
    """out/ içindeki mevcut kliplerin .txt açıklamalarını günceller."""
    cfg = cfg or load_config()
    od = ROOT / "out" / name
    for n, text in caps.items():
        for mp4 in od.glob(f"{n}_*.mp4"):
            mp4.with_suffix(".txt").write_text(txt_text(name, n, text, cfg), encoding="utf-8")


def run(video, cfg, force=False):
    def make():
        bits = select_llm.run(video, cfg)
        if not bits:
            return {}
        trs = proofread.run(video, cfg)  # onay sayfasının da kullandığı düzeltilmiş altyazı
        comedian, show = video_meta(video)
        prompt = PROMPT.format(comedian=comedian, n=len(bits),
                               clips="\n\n".join(clip_block(b, trs[str(b["n"])]) for b in bits))
        d = ask_json(prompt, SCHEMA, cfg, make_check(bits, comedian), label="açıklama", system=SYSTEM)
        return {str(c["n"]): compose(c["hook"], comedian, show, c["topic"]) for c in d["captions"]}

    caps = cached(work_dir(video) / "captions.json", make, force)
    write_txts(work_dir(video).name, caps, cfg)
    return caps

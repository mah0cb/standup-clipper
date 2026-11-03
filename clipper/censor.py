"""Küfür sansürü: yalnız çıktıya (ASS altyazı, notlar, hook, .txt açıklama) uygulanır; çeviri JSON'ları değişmez.
İki katman: proofread'de Claude'un bit için döndürdüğü "profanity" kelimeleri + config/censor_tr.txt regex listesi.
Kelime harf sayısı kadar yıldızla değişir (uzunluk aynı → karaoke zamanlaması değişmez)."""
import functools
import re

from .common import ROOT, read_json

WORDS_FILE = ROOT / "config" / "censor_tr.txt"
WORD = re.compile(r"\w+(?:['’]\w+)*")


def low(w):
    return w.replace("I", "ı").replace("İ", "i").lower()


@functools.lru_cache
def lists(path=WORDS_FILE):
    """{"strong"|"mild"|"except": derlenmiş regex} — her bölüm tek alternasyon."""
    sec, out = None, {"strong": [], "mild": [], "except": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            sec = line.strip("[]")
        else:
            out[sec].append(f"(?:{line})")
    return {k: re.compile("|".join(v) or "(?!)") for k, v in out.items()}


def is_profane(word, cfg, extra=(), allow=()):
    """word: tek kelime. extra/allow: küçük harf tam kelime biçimleri (Claude listesi + panel ekle / kaldır)."""
    w = low(word)
    base = re.split(r"['’]", w)[0]  # Fin'e → fin; kök kontrolü kesmeden önceki kısımda
    if w in allow:
        return False
    if w in extra:
        return True
    L = lists()
    if L["except"].fullmatch(base):
        return False
    return bool(L["strong"].fullmatch(base) or (cfg.get("include_mild") and L["mild"].fullmatch(base)))


def mask(word, mode):
    return word[0] + "*" * (len(word) - 1) if mode == "first_letter" else "*" * len(word)


def apply(text, cfg, extra=(), allow=()):
    """cfg: config["censor"]. Kapalıysa metin olduğu gibi döner."""
    if not cfg.get("enabled"):
        return text
    return WORD.sub(lambda m: mask(m[0], cfg["mode"]) if is_profane(m[0], cfg, extra, allow) else m[0], text)


def found(texts, cfg, extra=(), allow=()):
    """Metinlerde sansürlenecek kelimeler (sırayla, tekrarsız)."""
    if not cfg.get("enabled"):
        return []
    hits = [m[0] for t in texts for m in WORD.finditer(t) if is_profane(m[0], cfg, extra, allow)]
    return list(dict.fromkeys(hits))


def overrides(tr, review, n):
    """(extra, allow): Claude'un profanity listesi + onay sayfasındaki ekle; kaldır listesi."""
    o = review.get("censor", {}).get(str(n), {})
    extra = {low(w) for ws in tr.get("profanity", []) for w in ws} | {low(w) for w in o.get("add", [])}
    return extra, {low(w) for w in o.get("remove", [])}


def for_bit(wd, n, cfg):
    """work klasöründeki verilerden bit n için metin → sansürlü metin fonksiyonu."""
    tp, rp = wd / "translations_checked.json", wd / "review.json"
    tr = read_json(tp).get(str(n), {}) if tp.exists() else {}
    extra, allow = overrides(tr, read_json(rp) if rp.exists() else {}, n)
    return lambda text: apply(text, cfg["censor"], extra, allow)

"""6b) Çeviri kontrolü: ikinci bir Claude çağrısı Türkçe altyazıyı İngilizcesiyle karşılaştırıp düzeltir
→ translations_checked.json (altyazı, onay sayfası ve render bu dosyayı kullanır)."""
import json
from concurrent.futures import ThreadPoolExecutor

from . import select_llm, translate
from .claude_client import ask_json
from .common import cached, work_dir

SYSTEM = """Sen Türkçe altyazı editörüsün. Bir stand-up bitinin Türkçe altyazısını İngilizce orijinaliyle grup grup
karşılaştırıp şunları düzeltirsin:
- yazım ve noktalama hataları,
- anlamsız, bozuk ya da Türkçede doğal olmayan ifadeler (kelimesi kelimesine çeviri kokanlar),
- yanlış anlam (orijinalin söylediğini söylemiyorsa),
- orijinalde olmayan ya da orijinalden ağır küfür (şiddet birebir aynı olmalı), ve yumuşatılmış küfür,
- parantez (altyazıda yasak).
Mizahı ve punchline'ın satır sonunda olmasını koru. Sorun yoksa satırı aynen bırak; gereksiz değişiklik yapma.
Notlar ekranda ayrıca gösterilen kısa açıklamalardır; gereksizse silebilir, hatalıysa düzeltebilirsin."""

PROMPT = """Her grup için düzeltilmiş satırları döndür. Grup sayısı ve sırası AYNI kalmalı ({n} grup).
Her grup en fazla {max_lines} satır, satır başına en fazla {max_chars} karakter. Not en fazla {note_max} karakter.
Notlarda "i" orijinal notun numarasıdır; silmek istediğin notu listeye koyma.
Yaptığın her değişikliği "changes" listesine yaz (type: yazım | anlamsız | yanlış anlam | eklenmiş küfür | parantez | diğer).

ALTYAZI:
{groups}

NOTLAR:
{notes}
"""


def schema(cfg):
    s = cfg["subs"]
    line = {"type": "string", "minLength": 1, "maxLength": s["max_chars"]}
    return {
        "type": "object", "required": ["groups", "notes", "changes"],
        "properties": {
            "groups": {"type": "array", "items": {"type": "array", "minItems": 1, "maxItems": s["max_lines"], "items": line}},
            "notes": {"type": "array", "items": {
                "type": "object", "required": ["i", "text"],
                "properties": {"i": {"type": "integer"},
                               "text": {"type": "string", "minLength": 1, "maxLength": s["note_max_chars"]}}}},
            "changes": {"type": "array", "items": {
                "type": "object", "required": ["group", "before", "after", "type"],
                "properties": {"group": {"type": ["integer", "null"]}, "before": {"type": "string"},
                               "after": {"type": "string"}, "type": {"type": "string"},
                               "reason": {"type": "string"}}}},
        },
    }


def proofread_bit(bit, tr, cfg):
    s = cfg["subs"]
    groups, notes = tr["groups"], tr["notes"]
    if not groups:
        return {**tr, "changes": []}
    prompt = PROMPT.format(
        n=len(groups), max_lines=s["max_lines"], max_chars=s["max_chars"], note_max=s["note_max_chars"],
        groups="\n".join(f'{k}. EN: {g["en"]}\n   TR: {json.dumps(g["lines"], ensure_ascii=False)}' for k, g in enumerate(groups)),
        notes="\n".join(f'{k}. {n["text"]}' for k, n in enumerate(notes)) or "(yok)")

    def check(d):
        if len(d["groups"]) != len(groups):
            raise ValueError(f"{len(groups)} grup olmalı, {len(d['groups'])} geldi")
        translate.check_text({"groups": [{"lines": g} for g in d["groups"]], "notes": []}, 0)
        for nt in d["notes"]:
            if not 0 <= nt["i"] < len(notes):
                raise ValueError(f"geçersiz not numarası {nt['i']}")

    d = ask_json(prompt, schema(cfg), cfg, check, label=f"kontrol #{bit['n']}", system=SYSTEM)
    return {"groups": [{**g, "lines": lines} for g, lines in zip(groups, d["groups"])],
            "notes": [{**notes[nt["i"]], "text": nt["text"]} for nt in d["notes"]],
            "replacements": tr["replacements"], "changes": d["changes"]}


def run(video, cfg, force=False):
    def make():
        bits = select_llm.run(video, cfg)
        trs = translate.run(video, cfg)
        with ThreadPoolExecutor(cfg["claude"]["parallel"]) as ex:
            res = list(ex.map(lambda b: proofread_bit(b, trs[str(b["n"])], cfg), bits))
        return {str(b["n"]): r for b, r in zip(bits, res)}
    return cached(work_dir(video) / "translations_checked.json", make, force)

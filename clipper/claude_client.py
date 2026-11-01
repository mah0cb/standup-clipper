"""Claude çağrıları: cli (`claude -p`) veya api (Anthropic SDK). JSON ister, şemayla doğrular."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

import jsonschema

from .common import ROOT, log


def extract_json(text):
    """Yanıttaki ilk JSON nesnesini/dizisini çıkarır (``` blokları dahil)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("Yanıtta JSON bulunamadı")
    data, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return data


def _call_cli(prompt, c, system=None):
    exe = shutil.which("claude") or "claude"
    cmd = [exe, "-p", "--output-format", "json", "--model", c["model"],
           "--tools", "", "--no-session-persistence", "--disable-slash-commands"]
    if system:
        cmd += ["--system-prompt", system]
    # Proje bağlamı (CLAUDE.md vb.) karışmasın diye boş bir klasörde çalıştırılır
    with tempfile.TemporaryDirectory() as tmp:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           encoding="utf-8", timeout=c["timeout"], cwd=tmp)
    if p.returncode != 0:
        raise RuntimeError(f"claude CLI hata verdi ({p.returncode}): {p.stderr.strip() or p.stdout[:500]}")
    out = json.loads(p.stdout)
    if out.get("is_error"):
        raise RuntimeError(f"claude CLI hata: {out.get('result')}")
    cost = out.get("total_cost_usd")
    return out["result"], (f"abonelik (eşdeğer ${cost:.4f})" if cost is not None else "abonelik")


def _call_api(prompt, c, system=None):
    import anthropic
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # Stand-up metinleri küfür içerebilir: ret durumunda sunucu taraflı yedek modele geçiş açık
    resp = client.beta.messages.create(
        model=c["model"], max_tokens=16000,
        output_config={"effort": c.get("effort", "high")},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        **({"system": system} if system else {}),
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude isteği reddetti (refusal)")
    text = "".join(b.text for b in resp.content if b.type == "text")
    u = resp.usage
    cost = u.input_tokens * c["price_in"] / 1e6 + u.output_tokens * c["price_out"] / 1e6
    return text, f"{u.input_tokens} in / {u.output_tokens} out token, ~${cost:.4f}"


def ask_json(prompt, schema, cfg, check=None, label="claude", system=None):
    """Claude'dan şemaya uyan JSON alır. Geçersizse hatayı ekleyip max_retries kez yeniden dener.
    check(data): ek kontrol, sorun varsa ValueError fırlatır. system: sistem prompt'u (isteğe bağlı)."""
    c = cfg["claude"]
    call = _call_cli if c["mode"] == "cli" else _call_api
    base = (prompt + "\n\nYanıt olarak YALNIZCA aşağıdaki JSON şemasına uyan geçerli JSON döndür, "
            "açıklama yazma:\n" + json.dumps(schema, ensure_ascii=False))
    p, err = base, None
    for attempt in range(c["max_retries"] + 1):
        t = time.time()
        text, usage = call(p, c, system)
        log.info("[%s] %s çağrısı %.1f sn, %s (deneme %d)", label, c["mode"], time.time() - t, usage, attempt + 1)
        try:
            data = extract_json(text)
            jsonschema.validate(data, schema)
            if check:
                check(data)
            return data
        except (ValueError, jsonschema.ValidationError) as e:
            err = e
            msg = getattr(e, "message", str(e))
            log.warning("[%s] geçersiz yanıt: %s", label, msg)
            p = base + f"\n\nÖNCEKİ YANITIN GEÇERSİZDİ: {msg}\nBu hatayı düzelterek tekrar yanıtla."
    raise RuntimeError(f"[{label}] {c['max_retries'] + 1} denemede geçerli JSON alınamadı: {err}")

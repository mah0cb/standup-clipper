"""Paylaşım takvimi: paylaşılmamış klipleri kurallarla (Claude'suz) slotlara dağıtır → work/_schedule.json
{start, entries: [{path, at: "YYYY-MM-DDTHH:MM", pinned}], moved: [...]}. Saf mantık; zaman `now` ile gelir."""
import hashlib
import re
from collections import Counter, namedtuple
from datetime import date, datetime, timedelta, timezone

from . import censor
from .captions import tag_of
from .common import ROOT, read_json

ADULT_FILE = ROOT / "config" / "adult_tr.txt"
SHORT_SECS = 45
EARLY_DAYS = 3         # hesabın ilk 3 günü: kısa, yüksek puanlı, temiz klip önceliği
RESTRICTED_DAYS = 5    # küfür/yetişkin klipler hesabın ilk 5 gününe konmaz
TOPIC_GAP = 2          # aynı konu arasında en az 2 paylaşım
EVENT_MINUTES = 15     # .ics etkinlik süresi ve hatırlatma
STARS = re.compile(r"\*{2,}")

Settings = namedtuple("Settings", "slots account grace exclude")


class ScheduleError(ValueError):
    """Arayüze 400 olarak dönen, kullanıcının düzeltebileceği hata."""


def settings(cfg):
    c = cfg.get("schedule") or {}
    try:
        slots = sorted({datetime.strptime(str(s), "%H:%M").strftime("%H:%M") for s in c["slots"]})
        account = date.fromisoformat(str(c["account_start_date"]))
        grace = timedelta(minutes=int(c.get("grace_minutes", 60)))
    except (KeyError, TypeError, ValueError) as e:
        raise ScheduleError(f'config.yaml → schedule geçersiz (saatler tırnaklı olmalı, ör. "12:30"): {e}')
    if not slots:
        raise ScheduleError("config.yaml → schedule.slots boş olamaz")
    return Settings(slots, account, grace, set(c.get("exclude") or []))


def is_adult(word):
    L = censor.lists(ADULT_FILE)
    base = re.split(r"['’]", censor.low(word))[0]
    return not L["except"].fullmatch(base) and bool(L["strong"].fullmatch(base))


def clip_info(o, wd, cfg):
    """web.outputs() öğesi + takvim özellikleri (who, topic, kufur, yetiskin, kisa, flagged)."""
    tp, rp = wd / "translations_checked.json", wd / "review.json"
    tr = read_json(tp).get(str(o["n"]), {}) if tp.exists() else {}
    extra, allow = censor.overrides(tr, read_json(rp) if rp.exists() else {}, o["n"])
    groups = tr.get("groups", [])
    texts = [o["hook"], *(" ".join(g["lines"]) for g in groups), *(x["text"] for x in tr.get("notes", []))]
    # Sansür ayarı kapalı olsa da içerik küfürlüdür: sınıflama ayardan bağımsız
    kufur = bool(censor.found(texts, {**cfg["censor"], "enabled": True}, extra, allow) or STARS.search(o["text"]))
    words = (m[0] for t in [*texts, *(g.get("en", "") for g in groups), o["text"]] for m in censor.WORD.finditer(t))
    yetiskin = any(is_adult(w) for w in words)
    tags = re.findall(r"#\S+", o["text"])
    return {**o, "who": tag_of(o["comedian"]), "topic": tags[4] if len(tags) > 4 else None, "kufur": kufur,
            "yetiskin": yetiskin, "kisa": bool(o["duration"] and o["duration"] <= SHORT_SECS), "flagged": kufur or yetiskin}


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M")


def _score(c):
    return c.get("score") or 5


def _allowed(c, day_i, evening):
    return not c["flagged"] or (evening and day_i >= RESTRICTED_DAYS)


def _fill(pool, floor, st, fixed, history, rank, balance=False):
    """Slotları sırayla gezip her birine en uygun klibi koyar → {at: klip}. fixed: dolu slotlar, history: [(at, klip)].
    Küfür/yetişkin kuralı serttir (uyan klip yoksa slot boş kalır); gerisi sözlük sırasıyla önceliklidir, bu sıra
    2 komedyende A-B, B-A; 3'te A-B, C-A, B-C verir."""
    # ponytail: açgözlü seçim, küresel en iyiyi aramaz; sıkışık veride yumuşak kurallar ihlal edilebilir
    pool, placed, spread = list(pool), {}, []
    first = floor[:10]
    posts = sorted([*history, *((a, c) for a, c in fixed.items() if a < first)], key=lambda p: p[0])
    seen = dict(posts)
    day = date.fromisoformat(first)
    while pool:
        day_i = (day - st.account).days
        for t in st.slots:
            at = f"{day}T{t}"
            if at in fixed:
                posts.append((at, fixed[at]))
                seen[at] = fixed[at]
                continue
            evening = t == st.slots[-1]
            ok = [c for c in pool if _allowed(c, day_i, evening)]
            if at <= floor or not ok:
                continue
            flagged = sum(c["flagged"] for c in pool)
            # Kalan akşamları işaretli klipler dolduruyorsa akşama temiz klip konmaz (sonda boş öğleler kalmasın)
            pressure = evening and day_i >= RESTRICTED_DAYS and flagged * len(st.slots) >= len(pool)
            prev = posts[-1] if posts else None
            same_day = prev[1]["who"] if prev and prev[0][:10] == str(day) else None
            mirror = seen.get(f"{day - timedelta(days=1)}T{t}", {}).get("who")
            recent = {c["topic"] for _, c in posts[-TOPIC_GAP:]}
            left = Counter(c["who"] for c in pool)

            def cost(c):
                return (pressure and not c["flagged"], c["who"] == same_day, c["who"] == mirror,
                        bool(c["topic"]) and c["topic"] in recent, bool(prev) and c["who"] == prev[1]["who"],
                        -left[c["who"]] if balance else 0, rank(c, day_i, pool, spread))

            best = min(ok, key=cost)
            pool.remove(best)
            placed[at] = best
            posts.append((at, best))
            seen[at] = best
            if day_i >= EARLY_DAYS:
                spread.append(_score(best))
            if not pool:
                break
        day += timedelta(days=1)
    return placed


def _spread_rank(c, day_i, pool, spread):
    """İlk günler: kısa + yüksek puan. Sonra birikimli puan ortalamaya yakın kalır: yüksekler eşit yayılır."""
    if day_i < EARLY_DAYS:
        return (not c["kisa"], -_score(c))
    total = sum(spread) + sum(_score(x) for x in pool)
    mean = total / (len(spread) + len(pool))
    return (abs(sum(spread) + _score(c) - (len(spread) + 1) * mean),)


def _split(data, by, st, now):
    """Kayıtları ayırır → (yerinde kalan, kaçırılan, sabitsiz gelecek, düşen var mı)."""
    now_s, cutoff, today = _iso(now), _iso(now - st.grace), now.date().isoformat()
    keep, missed, future, dropped = [], [], [], False
    for e in sorted(data["entries"], key=lambda e: e["at"]):
        c = by.get(e["path"])
        if not c:
            dropped = True  # dosya silindi, adı değişti ya da hariç tutuldu
        elif c["shares"]:
            if e["at"][:10] <= today:
                keep.append(e)  # geçmiş kaydı; Bugün kartında diğer platformlar işaretlenebilsin
            else:
                dropped = True  # planlanan günden önce paylaşıldı
        elif e["at"] < cutoff:
            missed.append(e)
        elif e["pinned"] or e["at"] <= now_s:
            keep.append(e)  # sabit ya da tolerans süresinde (şu an paylaşılıyor olabilir)
        else:
            future.append(e)
    return keep, missed, future, dropped


def _history(keep, clips):
    """Takvim dışında paylaşılmış klipler: (ilk paylaşım günü, klip)."""
    planned = {e["path"] for e in keep}
    return [(min(c["shares"].values()) + "T00:00", c) for c in clips if c["shares"] and c["path"] not in planned]


def _result(start, keep, placed, moved):
    entries = keep + [{"path": c["path"], "at": at, "pinned": False} for at, c in placed.items()]
    return {"start": start, "entries": sorted(entries, key=lambda e: e["at"]), "moved": moved}


def rebuild(data, clips, st, start, now):
    """Sabitleri ve geçmişi korur, kalan paylaşılmamış klipleri kurallara göre baştan dağıtır."""
    by = {c["path"]: c for c in clips}
    keep = _split(data, by, st, now)[0]
    taken = {e["path"] for e in keep}
    pool = sorted((c for c in clips if not c["shares"] and c["path"] not in taken), key=lambda c: c["path"])
    fixed = {e["at"]: by[e["path"]] for e in keep}
    floor = max(_iso(now), f"{start}T00:00")
    placed = _fill(pool, floor, st, fixed, _history(keep, clips), _spread_rank, balance=True)
    return _result(str(start), keep, placed, data.get("moved", []))


def sync(data, clips, st, now):
    """Kaçırılan slotları kaydırır, yeni klipleri ekler. Değişiklik yoksa sıraya dokunmaz (elle yapılan sıra korunur)."""
    by = {c["path"]: c for c in clips}
    keep, missed, future, dropped = _split(data, by, st, now)
    today = now.date().isoformat()
    moved = [m for m in data.get("moved", []) if m["on"] == today]
    listed = {e["path"] for e in data["entries"]}
    new = sorted((c for c in clips if not c["shares"] and c["path"] not in listed), key=lambda c: (-_score(c), c["path"]))
    if not (missed or dropped or new):
        return {**data, "moved": moved}
    queue = [by[e["path"]] for e in missed + future] + new
    order = {c["path"]: i for i, c in enumerate(queue)}
    floor = max(_iso(now), f"{data['start']}T00:00")
    fixed = {e["at"]: by[e["path"]] for e in keep}
    placed = _fill(queue, floor, st, fixed, _history(keep, clips), lambda c, *_: (order[c["path"]],))
    to = {c["path"]: at for at, c in placed.items()}
    moved += [{"path": e["path"], "hook": by[e["path"]]["hook"], "from": e["at"], "to": to[e["path"]], "on": today} for e in missed]
    return _result(data["start"], keep, placed, moved)


def _entry(data, path):
    e = next((e for e in data["entries"] if e["path"] == path), None)
    if not e:
        raise ScheduleError("Bu klip takvimde yok")
    return e


def move(data, st, path, at, now):
    """Klibi slota taşır ve sabitler; slot doluysa oradaki klip eski yerine geçer."""
    try:
        valid = datetime.strptime(at, "%Y-%m-%dT%H:%M").strftime("%H:%M") in st.slots
    except ValueError:
        valid = False
    if not valid:
        raise ScheduleError(f"Geçersiz slot: {at}")
    now_s = _iso(now)
    e = _entry(data, path)
    if at <= now_s or e["at"] <= now_s:
        raise ScheduleError("Geçmiş slotlar değiştirilemez")
    other = next((x for x in data["entries"] if x["at"] == at), None)
    if other and other is not e:
        other["at"] = e["at"]
    e.update(at=at, pinned=True)
    data["entries"].sort(key=lambda x: x["at"])
    return data


def pin(data, path, pinned):
    _entry(data, path)["pinned"] = pinned
    return data


def view(data, clips, st, now):
    """Sayfanın verisi: bugün, günler (boş slotlarla), kalan gün, kaçırılan bilgisi."""
    by = {c["path"]: c for c in clips}
    today, now_s = now.date(), _iso(now)

    def item(e):
        c, d = by[e["path"]], date.fromisoformat(e["at"][:10])
        evening = e["at"][11:] == st.slots[-1]
        return {**c, "at": e["at"], "pinned": e["pinned"], "done": bool(c["shares"]), "past": e["at"] <= now_s,
                "warn": not _allowed(c, (d - st.account).days, evening)}

    items = {e["at"]: item(e) for e in data["entries"] if e["path"] in by}
    days = []
    if items:
        d, last = today, date.fromisoformat(max(items)[:10])
        while d <= last:
            slots = [{"at": f"{d}T{t}", "clip": items.get(f"{d}T{t}")} for t in st.slots]
            slots = [s for s in slots if s["clip"] or s["at"] > now_s]
            if slots:
                days.append({"date": str(d), "slots": slots})
            d += timedelta(days=1)
    upcoming = [c for at, c in sorted(items.items()) if at[:10] >= str(today) and not c["done"]]
    return {"start": data["start"], "today": [c for at, c in sorted(items.items()) if at[:10] == str(today)], "days": days,
            "next": next((c for c in upcoming if not c["past"]), None), "days_left": len({c["at"][:10] for c in upcoming}),
            "moved": data.get("moved", []), "now": now_s}


def _ics_text(s):
    return s.strip().replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line):
    """RFC 5545: satır en fazla 75 bayt; harfler (emoji dahil) bölünmez."""
    out, cur, size = [], "", 0
    for ch in line:
        n = len(ch.encode("utf-8"))
        if size + n > 75:
            out.append(cur)
            cur, size = " ", 1
        cur += ch
        size += n
    return out + [cur]


def to_ics(data, clips, st, now):
    """Gelecekteki paylaşımlar için takvim dosyası. UID kalıcı: yeniden içe aktarma kopya üretmesin diye."""
    by = {c["path"]: c for c in clips}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//standup-clipper//takvim//TR", "CALSCALE:GREGORIAN"]
    for e in data["entries"]:
        c = by.get(e["path"])
        if not c or c["shares"] or e["at"] <= _iso(now):
            continue
        t = datetime.strptime(e["at"], "%Y-%m-%dT%H:%M")
        desc = _ics_text(c["text"])
        lines += ["BEGIN:VEVENT", f"UID:{hashlib.sha1(e['path'].encode()).hexdigest()}@standup-clipper", f"DTSTAMP:{stamp}",
                  f"DTSTART:{t:%Y%m%dT%H%M%S}", f"DTEND:{t + timedelta(minutes=EVENT_MINUTES):%Y%m%dT%H%M%S}",
                  f"SUMMARY:{_ics_text(c['comedian'] + ': ' + c['hook'])}", f"DESCRIPTION:{desc}",
                  "BEGIN:VALARM", "ACTION:DISPLAY", f"TRIGGER:-PT{EVENT_MINUTES}M", f"DESCRIPTION:{_ics_text(c['hook'])}",
                  "END:VALARM", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "".join(f"{part}\r\n" for line in lines for part in _fold(line))

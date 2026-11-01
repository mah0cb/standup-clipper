"""7) 1080x1920 karaoke ASS altyazı üretir → work/{video}/subs/{n}.ass."""
from . import proofread, select_llm
from .common import ROOT, read_json, video_info, work_dir

FONTS_DIR = ROOT / "fonts"

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{font},{size},{hl},{primary},{oc},&H80000000,{bold},0,0,0,100,100,0,0,1,{outline},0,{main_align},60,60,{margin_v},1
Style: Hook,{font},{hook_size},{primary},{primary},{oc},&H80000000,{bold},0,0,0,100,100,0,0,1,{outline},0,8,60,60,{hook_margin_v},1
Style: Note,{font},{note_size},{primary},{primary},{oc},&H80000000,0,0,0,0,100,100,0,0,1,3,0,8,60,60,{note_margin_v},1
Style: Credit,{font},{credit_size},{primary},{primary},{oc},&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,60,60,{credit_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def font_name(s):
    return s["font"] if any(FONTS_DIR.glob("Montserrat*.ttf")) else s["fallback_font"]


def ts(t):
    t = max(0.0, t)
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def esc(text):
    return text.replace("{", "(").replace("}", ")").replace("\\", "/")


def karaoke(lines, dur_cs):
    """Grup süresini kelimelere karakter uzunluğuyla orantılı dağıtır; \\k toplamı = dur_cs."""
    words = [(li, w) for li, line in enumerate(lines) for w in line.split()]
    total = sum(len(w) for _, w in words) or 1
    ks, used = [], 0
    for i, (_, w) in enumerate(words):
        k = dur_cs - used if i == len(words) - 1 else round(dur_cs * len(w) / total)
        ks.append(k)
        used += k
    out = [[] for _ in lines]
    for (li, w), k in zip(words, ks):
        out[li].append(f"{{\\k{k}}}{esc(w)}")
    return "\\N".join(" ".join(p) for p in out)


def main_position(layout, src_w, src_h, s):
    """Ana altyazının hizası ve dikey marjı. blur_bg: ortadaki videonun hemen altı (üstten hizalı)."""
    if layout == "blur_bg":
        fg_h = round(1080 * src_h / src_w / 2) * 2
        return 8, (1920 - fg_h) // 2 + fg_h + s["below_video_gap"]
    return 2, s["margin_v"]


def build_ass(bit, tr, comedian, s, main_align=2, margin_v=None):
    t0, dur = bit["start"], bit["end"] - bit["start"]
    font = font_name(s)
    head = HEADER.format(font=font, bold=0 if "Montserrat" in font else -1, hl=s["highlight"], oc=s["outline_color"],
                         main_align=main_align, margin_v=s["margin_v"] if margin_v is None else margin_v,
                         **{k: s[k] for k in ("size", "primary", "outline", "hook_size",
                                              "hook_margin_v", "credit_size", "credit_margin_v",
                                              "note_size", "note_margin_v")})
    ev = [f"Dialogue: 1,{ts(0)},{ts(min(s['hook_secs'], dur))},Hook,,0,0,0,,{{\\fad(150,300)}}{esc(bit['hook_tr'])}"]
    if comedian:
        ev.append(f"Dialogue: 0,{ts(0)},{ts(dur)},Credit,,0,0,0,,{esc(comedian)}")
    for nt in tr.get("notes", []):
        st = nt["start"] - t0
        ev.append(f"Dialogue: 1,{ts(st)},{ts(min(st + s['note_secs'], dur))},Note,,0,0,0,,"
                  f"{{\\fad(150,200)}}{esc(nt['text'])}")
    groups = tr["groups"]
    for i, g in enumerate(groups):
        start, end = g["start"] - t0, g["end"] - t0
        # Kısa boşluklarda altyazı yanıp sönmesin: bir sonraki gruba kadar ekranda kalsın
        nxt = groups[i + 1]["start"] - t0 if i + 1 < len(groups) else dur
        show_end = min(nxt, end + 1.0, dur)
        if show_end <= max(start, 0):
            continue
        ev.append(f"Dialogue: 0,{ts(start)},{ts(show_end)},Main,,0,0,0,,"
                  f"{karaoke(g['lines'], int(round((end - start) * 100)))}")
    return head + "\n".join(ev) + "\n"


def run(video, cfg, force=True, layout="crop"):
    """Hızlı ve onay sayfasındaki düzeltmelere bağlı olduğu için her seferinde yeniden üretilir."""
    wd = work_dir(video)
    vi = video_info(video)
    align, margin = main_position(layout, vi["w"], vi["h"], cfg["subs"])
    bits = select_llm.run(video, cfg)
    trs = proofread.run(video, cfg)
    meta = read_json(wd / "meta.json") if (wd / "meta.json").exists() else {}
    paths = {}
    for b in bits:
        p = wd / "subs" / f"{b['n']}.ass"
        p.parent.mkdir(exist_ok=True)
        p.write_text(build_ass(b, trs[str(b["n"])], meta.get("comedian", ""), cfg["subs"], align, margin), encoding="utf-8")
        paths[b["n"]] = p
    return paths

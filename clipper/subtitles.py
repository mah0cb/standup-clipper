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
Style: Main,{font},{size},{hl},{primary},{oc},&H80000000,{bold},0,0,0,100,100,0,0,1,{outline},0,{main_align},{ml},{mr},{margin_v},1
Style: Hook,{font},{hook_size},{primary},{primary},{oc},&H80000000,{bold},0,0,0,100,100,0,0,1,{outline},0,2,{ml},{mr},{hook_margin_v},1
Style: Note,{font},{note_size},{primary},{primary},{oc},&H80000000,0,0,0,0,100,100,0,0,1,3,0,8,{ml},{mr},{note_margin_v},1
Style: Credit,{font},{credit_size},{credit_color},{credit_color},&H80000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,8,{ml},{mr},{credit_y},1

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


def main_position(layout, src_w, src_h, s, z):
    """Ana altyazının hizası ve dikey marjı; alt kenarı güvenli alanın içinde (z["text_bottom"]).
    blur_bg: ortadaki videonun hemen altı (üstten hizalı), iki satır güvenli alana sığmıyorsa yukarı çekilir."""
    if layout == "blur_bg":
        fg_h = round(1080 * src_h / src_w / 2) * 2
        two_lines = round(2 * s["size"] * 1.25 + 2 * s["outline"])
        return 8, min((1920 - fg_h) // 2 + fg_h + s["below_video_gap"], z["text_bottom"] - two_lines)
    return 2, 1920 - z["text_bottom"]


def build_ass(bit, tr, comedian, s, z, main_align=2, margin_v=None):
    """z: güvenli alan (config safe_zone). Altyazı ve notlar text_top–text_bottom bandında, yatayda left/right
    boşluklu; hook üstte, komedyen adı hook'un hemen altında (hook alttan hizalı, uzunsa yukarı doğru büyür)."""
    t0, dur = bit["start"], bit["end"] - bit["start"]
    font = font_name(s)
    head = HEADER.format(font=font, bold=0 if "Montserrat" in font else -1, hl=s["highlight"], oc=s["outline_color"],
                         main_align=main_align, margin_v=1920 - z["text_bottom"] if margin_v is None else margin_v,
                         ml=z["left"], mr=z["right"], note_margin_v=z["text_top"],
                         hook_margin_v=1920 - s["credit_y"] + s["credit_gap"],
                         **{k: s[k] for k in ("size", "primary", "outline", "hook_size", "credit_size", "credit_color",
                                              "credit_y", "note_size")})
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
    z = cfg["safe_zone"]
    align, margin = main_position(layout, vi["w"], vi["h"], cfg["subs"], z)
    bits = select_llm.run(video, cfg)
    trs = proofread.run(video, cfg)
    meta = read_json(wd / "meta.json") if (wd / "meta.json").exists() else {}
    paths = {}
    for b in bits:
        p = wd / "subs" / f"{b['n']}.ass"
        p.parent.mkdir(exist_ok=True)
        p.write_text(build_ass(b, trs[str(b["n"])], meta.get("comedian", ""), cfg["subs"], z, align, margin), encoding="utf-8")
        paths[b["n"]] = p
    return paths

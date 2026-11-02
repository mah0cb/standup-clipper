"""10) Render öncesi onay sayfası: yerel sunucu + HTML. Kaydet → review.json, selected.json, translations_checked.json."""
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import captions, proofread, select_llm
from .common import log, read_json, work_dir, write_json

HTML = Path(__file__).with_name("review.html")
TR_FILE = "translations_checked.json"


def load(wd):
    bits, trs = read_json(wd / "selected.json"), read_json(wd / TR_FILE)
    rv = wd / "review.json"
    approved = read_json(rv)["approved"] if rv.exists() else [b["n"] for b in bits]
    cp = wd / "captions.json"
    return {"bits": bits, "translations": trs, "approved": approved, "captions": read_json(cp) if cp.exists() else {}}


def save(wd, body):
    """Gövde: {approved:[n], hooks:{n:str}, lines:{n:[[satır,...] grup başına]}, captions:{n:str}}"""
    bits, trs = read_json(wd / "selected.json"), read_json(wd / TR_FILE)
    for b in bits:
        n = str(b["n"])
        if n in body.get("hooks", {}):
            b["hook_tr"] = str(body["hooks"][n]).strip() or b["hook_tr"]
        new = body.get("lines", {}).get(n)
        if new is not None:
            groups = trs[n]["groups"]
            if len(new) != len(groups):
                raise ValueError(f"bit {n}: grup sayısı değişemez")
            for g, lines in zip(groups, new):
                lines = [l.strip() for l in lines if l.strip()]
                if lines:
                    g["lines"] = lines
    write_json(wd / "selected.json", bits)
    write_json(wd / TR_FILE, trs)
    write_json(wd / "review.json", {"approved": sorted(int(n) for n in body.get("approved", []))})
    caps = {n: t.strip() for n, t in body.get("captions", {}).items() if t.strip()}
    if caps:
        cp = wd / "captions.json"
        caps = {**(read_json(cp) if cp.exists() else {}), **caps}
        write_json(cp, caps)
        captions.write_txts(wd.name, caps)  # render edilmiş kliplerin .txt'si de güncellenir


def make_server(wd, cfg, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/data":
                d = load(wd)
                d["max_chars"] = cfg["subs"]["max_chars"]
                self._send(200, json.dumps(d, ensure_ascii=False))
            elif self.path == "/":
                self._send(200, HTML.read_text(encoding="utf-8"), "text/html; charset=utf-8")
            else:
                self._send(404, "{}")

        def do_POST(self):
            if self.path != "/save":
                return self._send(404, "{}")
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                save(wd, body)
                self._send(200, '{"ok": true}')
            except (ValueError, KeyError) as e:
                self._send(400, json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))

        def log_message(self, *a):
            pass

    try:
        return ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:  # port doluysa boş bir port
        return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def run(video, cfg, force=False, open_browser=True):
    select_llm.run(video, cfg)
    proofread.run(video, cfg)
    srv = make_server(work_dir(video), cfg, cfg["review"]["port"])
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    log.info("Onay sayfası: %s  (kaydettikten sonra Ctrl+C ile kapatın, ardından `python main.py render %s`)", url, video)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()

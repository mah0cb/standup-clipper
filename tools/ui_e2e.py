"""Arayüz uçtan uca testi: gerçek sunucu + gerçek pipeline, tarayıcıdan tıklayarak.
ekle → başlat → onay → kaydet ve render et → çıktılar; her sayfanın ekran görüntüsü work/screenshots/.

Kullanım: python tools/ui_e2e.py samples/ismo_test.mp4 ISMO "5 dk test"
Yüklü Microsoft Edge'i kullanır (tarayıcı indirmez); profil work/_edge_profile altında.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
PORT = 8771
URL = f"http://127.0.0.1:{PORT}"
SHOTS = ROOT / "work" / "screenshots"


def get(path):
    return json.loads(urllib.request.urlopen(URL + path).read())


def wait_job(job_id, timeout=3600):
    t = time.time()
    while time.time() - t < timeout:
        j = get(f"/api/jobs/{job_id}")
        if j["status"] in ("done", "error", "limit"):
            return j
        time.sleep(3)
    raise TimeoutError(job_id)


def main(src, comedian, show):
    SHOTS.mkdir(parents=True, exist_ok=True)
    srv = subprocess.Popen([sys.executable, "-m", "clipper.web", "--port", str(PORT)], cwd=ROOT)
    report = {}
    try:
        for _ in range(60):
            try:
                get("/api/videos")
                break
            except OSError:
                time.sleep(0.5)
        with sync_playwright() as pw:
            ctx = pw.chromium.launch_persistent_context(str(ROOT / "work" / "_edge_profile"), channel="msedge",
                                                        headless=True, viewport={"width": 1400, "height": 1000})
            page = ctx.new_page()
            shot = lambda name: page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)

            # a) Kütüphane: ekle
            page.goto(URL + "/#/library")
            page.wait_for_selector("tbody")
            page.set_input_files("#file", src)
            page.fill("#comedian", comedian)
            page.fill("#show", show)
            shot("1_kutuphane_ekle")
            t = time.time()
            page.click("#upload")
            page.wait_for_function("document.querySelector('#upmsg').textContent.startsWith('Eklendi') || "
                                   "document.querySelector('#upmsg').textContent.startsWith('Hata')", timeout=600_000)
            msg = page.inner_text("#upmsg")
            assert msg.startswith("Eklendi"), msg
            name = msg.split(": ", 1)[1]
            report["video"], report["upload_s"] = name, round(time.time() - t, 1)
            row = f'tr.video[data-name="{name}"]'
            page.wait_for_selector(row)
            shot("2_kutuphane_eklendi")

            # b) İşleme: başlat → canlı ilerleme
            t = time.time()
            page.click(f"{row} button.start")
            page.wait_for_url("**/#/job/*")
            job_id = page.url.rsplit("/", 1)[1]
            page.wait_for_function("document.querySelector('#log').textContent.includes('✔ transcribe')", timeout=900_000)
            shot("3_isleme_devam")
            j = wait_job(job_id)
            report["process"] = {"status": j["status"], "steps": j["steps"], "total_s": round(time.time() - t, 1),
                                 "error": j.get("error_tr")}
            page.wait_for_timeout(1500)
            shot("4_isleme_bitti")
            assert j["status"] == "done", j

            # c) Onay: kaydet ve render et
            page.goto(URL + "/#/library")
            page.wait_for_selector(f"{row} button.review:not([disabled])")
            shot("5_kutuphane_onay_bekliyor")
            page.click(f"{row} button.review")
            page.wait_for_selector(".bit")
            report["bits"] = page.locator(".bit").count()
            page.locator(".bit button.play").first.click()  # önizleme oynatıcı aralığa atlar
            page.wait_for_timeout(2500)
            report["preview_time"] = page.evaluate("document.querySelector('video.prev').currentTime")
            shot("6_onay")
            t = time.time()
            page.click("#save-render")
            page.wait_for_url("**/#/job/*")
            rj = wait_job(page.url.rsplit("/", 1)[1])
            report["render"] = {"status": rj["status"], "total_s": round(time.time() - t, 1), "error": rj.get("error_tr")}
            page.wait_for_timeout(1500)
            shot("7_render_bitti")
            assert rj["status"] == "done", rj

            # d) Çıktılar: paylaşım işareti
            page.goto(URL + "/#/outputs/" + Path(name).stem)
            page.wait_for_selector(".out video")
            page.locator(".out .share-youtube").first.check()
            page.wait_for_timeout(800)
            report["outputs"] = page.locator(".out").count()
            shot("8_ciktilar")
            page.check("#unshared")
            page.wait_for_timeout(500)
            report["unshared_after_mark"] = page.locator(".out").count()
            page.uncheck("#unshared")

            # e) Ayarlar ve iş listesi (yalnız görüntü; ayar kaydedilmez)
            page.goto(URL + "/#/settings")
            page.wait_for_selector("#save-settings")
            shot("9_ayarlar")
            page.goto(URL + "/#/jobs")
            page.wait_for_selector("table")
            shot("10_isler")
            ctx.close()
    finally:
        srv.terminate()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])

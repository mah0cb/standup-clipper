import json
import shutil
import subprocess
import types

import pytest
from fastapi.testclient import TestClient

from clipper import claude_client, common, jobs, web
from clipper.common import read_json, write_json


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Geçici samples/out/work + sahte iş çalıştırıcı; gerçek pipeline çalışmaz."""
    for d in ("samples", "out", "work"):
        (tmp_path / d).mkdir()
    shutil.copy(common.ROOT / "config.yaml", tmp_path / "config.yaml")
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(web, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(web, "OUT", tmp_path / "out")
    monkeypatch.setattr(web, "SHARES", tmp_path / "work" / "_shares.json")
    monkeypatch.setattr(web, "WORK", tmp_path / "work")
    monkeypatch.setattr(web, "CONFIG", tmp_path / "config.yaml")
    monkeypatch.setattr(web, "load_config", lambda: common.load_config(tmp_path / "config.yaml"))
    monkeypatch.setattr(jobs, "LOG_DIR", tmp_path / "work" / "_jobs")
    calls = []

    def runner(job, on_step):
        calls.append(job)
        if job["opts"].get("fail") == "limit":
            raise claude_client.ClaudeLimitError("Claude limiti doldu: usage limit reached")
        if job["opts"].get("fail"):
            raise RuntimeError("bozuk dosya")
        on_step("audio", None)
        on_step("audio", 1.5)

    q = jobs.JobQueue(path=tmp_path / "work" / "_jobs.json", runner=runner)
    client = TestClient(web.create_app(queue=q, start_worker=False))
    return types.SimpleNamespace(root=tmp_path, client=client, queue=q, calls=calls)


def tiny_video(path, w=320, h=240):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=s={w}x{h}:d=1",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def test_upload_names_file_and_rejects_duplicates(env):
    files = {"file": ("x.MP4", b"veri", "video/mp4")}
    r = env.client.post("/api/videos/upload", files=files, data={"comedian": "Zoltan Kaszas", "show": "London Fog"})
    assert r.status_code == 200 and r.json()["name"] == "zoltan_kaszas__london_fog.mp4"
    assert (env.root / "samples" / "zoltan_kaszas__london_fog.mp4").read_bytes() == b"veri"
    assert read_json(env.root / "work" / "zoltan_kaszas__london_fog" / "meta.json")["comedian"] == "Zoltan Kaszas"
    r = env.client.post("/api/videos/upload", files=files, data={"comedian": "Zoltan Kaszas", "show": "London Fog"})
    assert r.status_code == 409
    r = env.client.post("/api/videos/upload", files=files, data={"comedian": " ", "show": "x"})
    assert r.status_code == 400


def test_library_lists_videos_with_info_and_low_res_warning(env):
    tiny_video(env.root / "samples" / "zoltan_kaszas__london_fog.mp4")
    v = env.client.get("/api/videos").json()[0]
    assert v["comedian"] == "Zoltan Kaszas" and v["show"] == "London Fog"
    assert (v["width"], v["height"]) == (320, 240) and v["low_res"] and v["status"] == "yeni"
    assert v["duration"] == pytest.approx(1.0, abs=0.1)


def test_start_job_runs_and_updates_status(env):
    tiny_video(env.root / "samples" / "a__b.mp4")
    job = env.client.post("/api/videos/a__b.mp4/start", json={}).json()
    assert job["status"] == "queued"
    assert env.client.get("/api/videos").json()[0]["status"] == "sırada"
    assert env.queue.run_one()
    j = env.client.get(f"/api/jobs/{job['id']}").json()
    assert j["status"] == "done" and j["steps"] == {"audio": 1.5}
    assert "İş başladı" in j["log"] and "İş tamamlandı" in j["log"]
    assert not env.queue.run_one()  # kuyruk boş


def test_limit_error_is_recognized_and_retryable(env):
    tiny_video(env.root / "samples" / "a__b.mp4")
    job = env.client.post("/api/videos/a__b.mp4/start", json={}).json()
    env.queue._update(job["id"], opts={"fail": "limit"})
    env.queue.run_one()
    j = env.client.get(f"/api/jobs/{job['id']}").json()
    assert j["status"] == "limit" and "Claude limiti doldu" in j["error_tr"] and j["retry_at"] is None
    assert env.client.get("/api/videos").json()[0]["status"] == "hata"
    r = env.client.post(f"/api/jobs/{job['id']}/retry", json={"auto_retry": True}).json()
    assert r["status"] == "queued" and r["auto_retry"]


def test_auto_retry_schedules_after_limit(env):
    tiny_video(env.root / "samples" / "a__b.mp4")
    job = env.queue.add("a__b.mp4", "process", {"fail": "limit", "auto_retry": True})
    env.queue.run_one()
    j = env.queue.get(job["id"])
    assert j["status"] == "limit" and j["retry_at"] > 0
    assert not env.queue.run_one()  # zamanı gelmedi
    env.queue._update(job["id"], retry_at=1)
    assert env.queue.run_one()  # zamanı gelince yeniden denendi


def test_generic_error_has_turkish_summary(env):
    tiny_video(env.root / "samples" / "a__b.mp4")
    job = env.queue.add("a__b.mp4", "process", {"fail": True})
    env.queue._update(job["id"], current="transcribe")
    env.queue.run_one()
    assert env.queue.get(job["id"])["error_tr"] == "'transcribe' adımında hata: bozuk dosya"


def test_jobs_survive_restart(env):
    job = env.queue.add("a__b.mp4", "process")
    env.queue._update(job["id"], status="running")
    q2 = jobs.JobQueue(path=env.queue.path, runner=env.queue.runner)
    assert q2.get(job["id"])["status"] == "queued"  # yarıda kalan iş kuyruğa geri alındı


def test_review_get_save_and_render(env):
    tiny_video(env.root / "samples" / "a__b.mp4")
    wd = env.root / "work" / "a__b"
    write_json(wd / "selected.json", [{"n": 1, "start": 0, "end": 30, "hook_tr": "eski", "score": 8, "reason": "r"}])
    write_json(wd / "translations_checked.json", {"1": {"groups": [{"start": 0, "end": 1, "lines": ["a"], "en": "a"}],
                                                       "notes": [], "replacements": [], "uncovered": [], "changes": []}})
    d = env.client.get("/api/videos/a__b.mp4/review").json()
    assert d["approved"] == [1] and d["max_chars"] == 38
    r = env.client.post("/api/videos/a__b.mp4/review", json={"approved": [1], "hooks": {"1": "yeni"},
                                                            "lines": {"1": [["düzeltildi"]]}, "render": True, "layout": "blur_bg"}).json()
    assert r["job"]["kind"] == "render" and r["job"]["opts"]["layout"] == "blur_bg"
    assert read_json(wd / "selected.json")[0]["hook_tr"] == "yeni"
    assert read_json(wd / "translations_checked.json")["1"]["groups"][0]["lines"] == ["düzeltildi"]


def test_review_not_ready_and_path_traversal(env):
    tiny_video(env.root / "samples" / "a__b.mp4")
    assert env.client.get("/api/videos/a__b.mp4/review").status_code == 409
    assert env.client.get("/api/videos/..%5Cconfig.yaml/file").status_code == 404
    assert env.client.get("/api/videos/nope.mp4/file").status_code == 404
    assert env.client.get("/api/outputs/file", params={"path": "../config.yaml"}).status_code == 404


def test_video_file_supports_range(env):
    tiny_video(env.root / "samples" / "a__b.mp4")
    r = env.client.get("/api/videos/a__b.mp4/file", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_outputs_and_share_tracking(env):
    d = env.root / "out" / "a__b"
    d.mkdir()
    tiny_video(d / "1_Hook.mp4", 108, 192)
    (d / "1_Hook.txt").write_text("açıklama #a", encoding="utf-8")
    write_json(env.root / "work" / "a__b" / "selected.json", [{"n": 1, "hook_tr": "Hook başlığı", "score": 9}])
    o = env.client.get("/api/outputs").json()[0]
    assert o["hook"] == "Hook başlığı" and o["text"] == "açıklama #a" and o["shares"] == {}
    s = env.client.post("/api/outputs/share", json={"path": o["path"], "platform": "tiktok", "done": True}).json()
    assert "tiktok" in s
    assert env.client.get("/api/outputs").json()[0]["shares"]["tiktok"] == s["tiktok"]
    env.client.post("/api/outputs/share", json={"path": o["path"], "platform": "tiktok", "done": False})
    assert env.client.get("/api/outputs").json()[0]["shares"] == {}
    assert env.client.post("/api/outputs/share", json={"path": o["path"], "platform": "x", "done": True}).status_code == 400


def test_settings_roundtrip_keeps_comments(env):
    s = env.client.get("/api/settings").json()
    assert s["values"]["render.layout"] == "crop" and s["fields"]["upscale.mode"] == ["auto", "always", "off"]
    r = env.client.post("/api/settings", json={"upscale.mode": "auto"})
    r = env.client.post("/api/settings", json={"upscale.mode": "off", "laughter.threshold": "0.35", "select.max_bits": "8"})
    assert r.status_code == 200 and r.json()["upscale.mode"] == "off" and r.json()["laughter.threshold"] == 0.35 and r.json()["select.max_bits"] == 8
    text = (env.root / "config.yaml").read_text(encoding="utf-8")
    assert "auto (kırpılan alan 1920'den kısaysa)" in text and 'mode: "off"' in text  # yorumlar korundu
    assert env.client.post("/api/settings", json={"render.layout": "kare"}).status_code == 400
    assert env.client.post("/api/settings", json={"subs.size": "büyük"}).status_code == 400
    assert env.client.post("/api/settings", json={"bilinmeyen.alan": 1}).status_code == 400


def test_sse_stream_sends_log_and_status(env):
    tiny_video(env.root / "samples" / "a__b.mp4")
    job = env.client.post("/api/videos/a__b.mp4/start", json={}).json()
    env.queue.run_one()
    with env.client.stream("GET", f"/api/jobs/{job['id']}/stream") as r:
        body = "".join(r.iter_text())
    assert "event: log" in body and "İş tamamlandı" in body
    status = [json.loads(l[6:]) for l in body.splitlines() if l.startswith("data: {")]
    assert status[-1]["status"] == "done"


def test_cli_limit_error_detection():
    p = types.SimpleNamespace(returncode=1, stderr="", stdout=json.dumps({"is_error": True, "result": "Claude AI usage limit reached|1760000000"}))
    assert isinstance(claude_client._cli_error(p), claude_client.ClaudeLimitError)
    p = types.SimpleNamespace(returncode=1, stderr="", stdout=json.dumps({"is_error": True, "result": "bağlantı hatası"}))
    e = claude_client._cli_error(p)
    assert not isinstance(e, claude_client.ClaudeLimitError) and "bağlantı hatası" in str(e)

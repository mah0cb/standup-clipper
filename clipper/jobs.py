"""Arka plan iş kuyruğu: aynı anda 1 iş, diğerleri sırada.
Durum work/_jobs.json'da tutulur; sunucu yeniden başlarsa yarıda kalan iş kuyruğa geri alınır ve
adım cache'i sayesinde kaldığı yerden devam eder. Her işin logu work/_jobs/{id}.log."""
import logging
import threading
import time
import traceback
import uuid
from datetime import datetime

from . import pipeline
from .claude_client import ClaudeLimitError
from .common import ROOT, load_config, read_json, write_json

JOBS_FILE = ROOT / "work" / "_jobs.json"
LOG_DIR = ROOT / "work" / "_jobs"
FINAL = {"done", "error", "limit"}


def now():
    return datetime.now().isoformat(timespec="seconds")


def turkish_error(job, e):
    """Kullanıcıya gösterilecek kısa Türkçe hata özeti."""
    step = job.get("current") or "?"
    if isinstance(e, ClaudeLimitError):
        return "Claude limiti doldu. Limit sıfırlanınca kaldığı yerden devam edebilirsiniz."
    msg = str(e).strip().splitlines()[-1][:300] if str(e).strip() else type(e).__name__
    if "CUDA" in msg or "cudnn" in msg.lower():
        return f"'{step}' adımında GPU (CUDA) hatası: {msg}"
    if "ffmpeg" in msg.lower():
        return f"'{step}' adımında ffmpeg hatası: {msg}"
    return f"'{step}' adımında hata: {msg}"


def run_job(job, on_step):
    """Varsayılan iş çalıştırıcısı: CLI ile aynı pipeline fonksiyonları."""
    cfg = load_config()  # her iş güncel ayarlarla
    video = ROOT / "samples" / job["video"]
    if job["kind"] == "process":
        pipeline.process(video, cfg, job["opts"].get("comedian") or pipeline.comedian_from_filename(video), on_step)
    elif job["kind"] == "render":
        on_step("render", None)
        dt = pipeline.timed("render", video, cfg, layout=job["opts"].get("layout"), upscale=job["opts"].get("upscale"))
        on_step("render", dt)
    else:
        raise ValueError(f"bilinmeyen iş türü: {job['kind']}")


class _JobLog(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.f = open(path, "a", encoding="utf-8")
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.f.write(self.format(record) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
        super().close()


class JobQueue:
    def __init__(self, path=JOBS_FILE, runner=run_job, limit_retry_minutes=30):
        self.path, self.runner, self.retry_secs = path, runner, limit_retry_minutes * 60
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.jobs = read_json(path) if path.exists() else []
        for j in self.jobs:  # sunucu iş ortasında kapanmışsa: kaldığı yerden devam
            if j["status"] == "running":
                j["status"], j["note"] = "queued", "Sunucu yeniden başladı, kaldığı yerden devam edecek"
        self._save()
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _save(self):
        write_json(self.path, self.jobs)

    def list(self):
        with self.lock:
            return [dict(j) for j in self.jobs]

    def get(self, job_id):
        with self.lock:
            return next((dict(j) for j in self.jobs if j["id"] == job_id), None)

    def latest_for(self, video):
        with self.lock:
            js = [j for j in self.jobs if j["video"] == video]
            return dict(js[-1]) if js else None

    def add(self, video, kind, opts=None):
        job = {"id": uuid.uuid4().hex[:8], "video": video, "kind": kind, "opts": opts or {}, "status": "queued",
               "steps": {}, "current": None, "error": None, "error_tr": None, "created": now(),
               "started": None, "finished": None, "retry_at": None, "auto_retry": bool((opts or {}).get("auto_retry"))}
        with self.lock:
            self.jobs.append(job)
            self._save()
        self.wake.set()
        return dict(job)

    def retry(self, job_id, auto_retry=None):
        with self.lock:
            j = next(j for j in self.jobs if j["id"] == job_id)
            j.update(status="queued", error=None, error_tr=None, retry_at=None, finished=None)
            if auto_retry is not None:
                j["auto_retry"] = bool(auto_retry)
            self._save()
        self.wake.set()
        return dict(j)

    def log_path(self, job_id):
        return LOG_DIR / f"{job_id}.log"

    def _update(self, job_id, **kw):
        with self.lock:
            j = next(j for j in self.jobs if j["id"] == job_id)
            j.update(kw)
            self._save()
            return j

    def _next(self):
        with self.lock:
            t = time.time()
            for j in self.jobs:
                if j["status"] == "queued" or (j["status"] == "limit" and j["auto_retry"] and j["retry_at"]
                                               and j["retry_at"] <= t):
                    return j["id"]
        return None

    def run_one(self):
        """Sıradaki işi çalıştırır; iş yoksa False."""
        job_id = self._next()
        if not job_id:
            return False
        job = self._update(job_id, status="running", started=now(), error=None, error_tr=None, retry_at=None)
        handler = _JobLog(self.log_path(job_id))
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.INFO)

        def on_step(name, dt):
            with self.lock:
                j = next(j for j in self.jobs if j["id"] == job_id)
                if dt is None:
                    j["current"] = name
                else:
                    j["steps"][name] = dt
                self._save()

        try:
            logging.info("İş başladı: %s (%s)", job["video"], job["kind"])
            self.runner(dict(job), on_step)
            self._update(job_id, status="done", current=None, finished=now())
            logging.info("İş tamamlandı.")
        except ClaudeLimitError as e:
            j = self.get(job_id)
            logging.warning("Claude limiti: %s", e)
            self._update(job_id, status="limit", error=str(e), error_tr=turkish_error(j, e), finished=now(),
                         retry_at=time.time() + self.retry_secs if j["auto_retry"] else None)
        except Exception as e:
            logging.error("Hata:\n%s", traceback.format_exc())
            self._update(job_id, status="error", error=str(e), error_tr=turkish_error(self.get(job_id), e),
                         finished=now())
        finally:
            root.removeHandler(handler)
            handler.close()
        return True

    def start(self):
        def loop():
            while True:
                if not self.run_one():
                    self.wake.wait(timeout=30)  # yeni iş ya da limit yeniden deneme zamanı
                    self.wake.clear()
        threading.Thread(target=loop, daemon=True, name="job-worker").start()

"""Single-GPU workbench for upstream 4DAnyone inference and reconstruction."""
import asyncio
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import zipfile
from contextlib import asynccontextmanager
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import time_ns
from uuid import uuid4

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.staticfiles import StaticFiles

from fdanyone.assets import SMPLX_MODEL
from fdanyone.io import lock_output
from fdanyone.views import VIEWS_PER_GROUP

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("WORKBENCH_DATA", ROOT / "data" / "workbench")).resolve()
MODELS = DATA / "models"
JOBS = DATA / "jobs"
PASSWORD = os.environ.get("WORKBENCH_PASSWORD")
LIMIT = 90 * 1024**2
MODEL_LIMIT = 256 * 1024**2
MODEL_PART = 16 * 1024**2
DEFAULT_VIEWS = 6
DEFAULT_STEPS = 10000
LEGACY_PRESETS = {
    "preview": (6, [15], 5000, 2, 0.75),
    "standard": (24, [15], 30000, 2, 1.0),
    "detail": (16, [-10, 15, 35], 100000, 2, 1.0),
}
ARTIFACTS = {
    "model": "reconstruction/checkpoints/latest.npz",
    "orbit": "reconstruction/render/rgb.mp4",
    "log": "job.log",
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
operation = asyncio.Lock()
archive_lock = Lock()


def read_job(path):
    return json.loads((path / "job.json").read_text(encoding="utf-8"))


def save_job(path, job, **updates):
    job.update(updates)
    temporary = path / "job.tmp"
    temporary.write_text(json.dumps(job), encoding="utf-8")
    temporary.replace(path / "job.json")


def job_path(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404)
    path = JOBS / job_id
    if not (path / "job.json").is_file():
        raise HTTPException(404)
    return path


def latest_log(path):
    log = path / "job.log"
    if not log.exists():
        return ""
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 8192))
        return stream.read().decode("utf-8", errors="replace")


async def command(log, script, *arguments):
    process_group = sys.platform != "win32"
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", script, *arguments,
        cwd=ROOT, stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log,
        start_new_session=process_group,
    )
    try:
        code = await process.wait()
    except asyncio.CancelledError:
        if process.returncode is None:
            if process_group:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            await process.wait()
        raise
    if code:
        raise RuntimeError(f"{script} exited with code {code}. See the job log.")


async def reconstruct(state, path, job):
    multiview = path / "multiview"
    try:
        if "preset" in job:
            views, pitches, steps, batch, scale = LEGACY_PRESETS[job["preset"]]
        else:
            views, pitches, steps, batch, scale = job["views"], [15], job["steps"], 2, 1.0
        with (path / "job.log").open("ab", buffering=0) as log:
            if state.worker_limit > 1:
                if not state.models_ready:
                    save_job(path, job, state="preparing")
                async with state.model_setup:
                    if not state.models_ready:
                        await command(log, "scripts/download_model.py", f"--model_dir={MODELS}")
                        state.models_ready = True
            if not (multiview / "metadata.json").is_file():
                save_job(path, job, state="inference")
                await command(
                    log, "inference.py",
                    f"--video_path={path / job['source']}",
                    f"--output_dir={multiview}",
                    f"--model_dir={MODELS}",
                    f"--start_time={job['start']}",
                    f"--views_per_layer={views}",
                    f"--layer_pitches={json.dumps(pitches)}",
                    "--gpu_ids=[0]",
                )
            save_job(path, job, state="training")
            await command(
                log, "reconstruct.py",
                f"--data_dir={multiview}",
                f"--output_dir={path / 'reconstruction'}",
                f"--model_dir={MODELS}",
                f"--steps={steps}",
                f"--batch_size={batch}",
                f"--image_scale={scale}",
                "--device=cuda:0",
            )
        if not all((path / ARTIFACTS[name]).is_file() for name in ("model", "orbit")):
            raise RuntimeError("Reconstruction exited without producing its model and orbit video.")
        save_job(path, job, state="complete")
    except asyncio.CancelledError:
        save_job(path, job, state="interrupted")
        raise
    except Exception as exc:
        save_job(path, job, state="failed", error=str(exc))


async def run_worker(state):
    while True:
        path, job = await state.queue.get()
        state.running += 1
        try:
            await reconstruct(state, path, job)
        finally:
            state.running -= 1
            state.queue.task_done()


@asynccontextmanager
async def lifespan(app):
    setting = os.environ.get("WORKBENCH_MAX_JOBS", "1")
    try:
        limit = int(setting)
    except ValueError as exc:
        raise RuntimeError("WORKBENCH_MAX_JOBS must be a positive integer.") from exc
    if limit < 1:
        raise RuntimeError("WORKBENCH_MAX_JOBS must be a positive integer.")
    JOBS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    state = app.state
    with lock_output(JOBS):
        state.worker_limit = limit
        state.queue = asyncio.Queue()
        state.running = 0
        state.model_setup = asyncio.Lock()
        state.models_ready = False
        for model_part in MODELS.glob("incoming-*.part"):
            model_part.unlink()
        queued = []
        for record in JOBS.glob("*/job.json"):
            job = read_job(record.parent)
            if job["state"] in {"preparing", "inference", "training"}:
                save_job(record.parent, job, state="interrupted")
            elif job["state"] == "queued":
                queued.append((job.get("queued_at", record.stat().st_mtime_ns), record.parent, job))
        for _, path, job in sorted(queued):
            state.queue.put_nowait((path, job))
        workers = [asyncio.create_task(run_worker(state)) for _ in range(limit)]
        try:
            yield
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def authenticate(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if PASSWORD:
        from base64 import b64encode

        expected = b"Basic " + b64encode(f"workbench:{PASSWORD}".encode())
        supplied = request.headers.get("authorization", "").encode()
        if not secrets.compare_digest(supplied, expected):
            return PlainTextResponse("Sign in to the workbench.", status_code=401,
                                     headers={"WWW-Authenticate": 'Basic realm="4DAnyone Workbench"'})
    if request.method == "POST" and request.headers.get("x-workbench") != "1":
        return JSONResponse({"detail": "Open the workbench to submit this request."}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def home():
    return FileResponse(ROOT / "workbench" / "static" / "index.html")


@app.get("/api/jobs")
async def jobs(request: Request):
    state = request.app.state
    return {
        "smplx": (MODELS / SMPLX_MODEL).is_file(),
        "worker_limit": state.worker_limit,
        "running": state.running,
        "queued": state.queue.qsize(),
        "jobs": [read_job(record.parent)
                 for record in sorted(JOBS.glob("*/job.json"), key=lambda p: p.parent.stat().st_mtime, reverse=True)],
    }


@app.get("/api/jobs/{job_id}")
def detail(job_id: str):
    path = job_path(job_id)
    return {**read_job(path), "log": latest_log(path)}


async def store(upload, destination):
    size = 0
    try:
        with destination.open("wb") as output:
            while block := await upload.read(1024**2):
                size += len(block)
                if size > LIMIT:
                    raise HTTPException(413, "File exceeds the 90 MiB upload limit.")
                output.write(block)
        if not size:
            raise HTTPException(422, "The uploaded file is empty.")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


@app.post("/api/smplx/{upload_id}")
async def install_smplx_part(
    request: Request, upload_id: str, part: UploadFile, offset: int = Form(), total: int = Form(),
):
    from fdanyone.download import install_smplx
    from fdanyone.errors import AssetError

    if not re.fullmatch(r"[a-f0-9]{32}", upload_id) or not 0 <= offset < total <= MODEL_LIMIT:
        raise HTTPException(422, "Invalid SMPL-X upload range or file size.")
    block = await part.read(MODEL_PART + 1)
    end = offset + len(block)
    if not block or len(block) > MODEL_PART or end > total:
        raise HTTPException(422, "Invalid SMPL-X upload chunk.")
    async with operation:
        state = request.app.state
        if state.running or not state.queue.empty():
            raise HTTPException(409, "Wait for active and queued reconstructions to finish.")
        source = MODELS / f"incoming-{upload_id}.part"
        if (source.stat().st_size if source.exists() else 0) != offset:
            raise HTTPException(409, "The model upload was interrupted. Select the file and try again.")
        with source.open("ab") as stream:
            stream.write(block)
        if end < total:
            return {"received": end}
        neutral = MODELS / "SMPLX_NEUTRAL.npz"
        try:
            with zipfile.ZipFile(source) as archive:
                required = {"v_template.npy", "shapedirs.npy", "posedirs.npy", "J_regressor.npy"}
                if not required.issubset(archive.namelist()):
                    raise zipfile.BadZipFile
            source.replace(neutral)
            await asyncio.to_thread(install_smplx, neutral, MODELS, ROOT / "third_party/GVHMR")
        except zipfile.BadZipFile as exc:
            raise HTTPException(422, "Select SMPLX_NEUTRAL.npz from the official SMPL-X archive.") from exc
        except AssetError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            source.unlink(missing_ok=True)
            neutral.unlink(missing_ok=True)
    return {"received": end, "installed": True}


@app.post("/api/jobs", status_code=202)
async def create_job(
    request: Request,
    video: UploadFile,
    views: int | None = Form(None),
    steps: int | None = Form(None),
    start: float = Form(0),
    preset: str | None = Form(None),
):
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES or not math.isfinite(start) or start < 0:
        raise HTTPException(422, "Choose a supported video and a valid start time.")
    if preset is not None:
        if preset not in LEGACY_PRESETS or views is not None or steps is not None:
            raise HTTPException(422, "Choose either an existing preset or separate views and steps.")
        settings = {"preset": preset}
    else:
        views = DEFAULT_VIEWS if views is None else views
        steps = DEFAULT_STEPS if steps is None else steps
        if views <= 0 or views % VIEWS_PER_GROUP:
            raise HTTPException(422, f"Views must be a positive multiple of {VIEWS_PER_GROUP}.")
        if steps <= 0:
            raise HTTPException(422, "Training steps must be a positive whole number.")
        settings = {"views": views, "steps": steps}
    async with operation:
        if not (MODELS / SMPLX_MODEL).is_file():
            raise HTTPException(409, "Install SMPL-X before starting a reconstruction.")
        job_id = uuid4().hex
        path = JOBS / job_id
        path.mkdir()
        job = dict(id=job_id, name=video.filename, start=start, source=f"source{suffix}", **settings)
        try:
            await store(video, path / job["source"])
            save_job(path, job, state="queued", queued_at=time_ns())
        except BaseException:
            (path / job["source"]).unlink(missing_ok=True)
            (path / "job.tmp").unlink(missing_ok=True)
            path.rmdir()
            raise
        request.app.state.queue.put_nowait((path, job))
        return job


@app.post("/api/jobs/{job_id}/resume", status_code=202)
async def resume(request: Request, job_id: str):
    path = job_path(job_id)
    async with operation:
        job = read_job(path)
        if job["state"] not in {"failed", "interrupted"}:
            raise HTTPException(409, "Only failed or interrupted jobs can be resumed.")
        if not (MODELS / SMPLX_MODEL).is_file():
            raise HTTPException(409, "Install SMPL-X before resuming a reconstruction.")
        save_job(path, job, state="queued", error=None, queued_at=time_ns())
        request.app.state.queue.put_nowait((path, job))
        return job


@app.get("/api/jobs/{job_id}/files/{kind}")
def artifact(job_id: str, kind: str, download: bool = False):
    path = job_path(job_id)
    job = read_job(path)
    if kind != "log" and job["state"] != "complete":
        raise HTTPException(404)
    stem = Path(job["name"]).stem
    if kind == "zip":
        return FileResponse(download_archive(path, job), filename=f"{stem}-4dgs.zip")
    if kind not in ARTIFACTS:
        raise HTTPException(404)
    file = path / ARTIFACTS[kind]
    if not file.is_file():
        raise HTTPException(404)
    filename = f"{stem}-{file.name}" if download or kind == "model" else None
    return FileResponse(file, filename=filename)


def download_archive(path, job):
    model = path / ARTIFACTS["model"]
    if not model.is_file():
        raise HTTPException(404, "The reconstruction model is missing.")
    destination = path / "model-and-audio.zip"
    with archive_lock:
        if destination.is_file():
            return destination
        source = path / job["source"]
        with TemporaryDirectory(prefix=".export-", dir=path) as temporary:
            folder = Path(temporary)
            audio = folder / "source-audio.m4a"
            staged = folder / destination.name
            try:
                if source.is_file():
                    extract_audio(source, path / "multiview" / "metadata.json", audio)
                with zipfile.ZipFile(staged, "w") as archive:
                    archive.write(model, "latest.npz")
                    if audio.is_file():
                        archive.write(audio, audio.name)
                staged.replace(destination)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                raise HTTPException(500, "Could not prepare the archive. The NPZ is available separately.") from exc
    return destination


def extract_audio(source, metadata, output):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time:format=start_time",
         "-of", "json", str(source)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    info = json.loads(probe.stdout)
    streams = info["streams"]
    if not any(stream["codec_type"] == "audio" for stream in streams):
        return
    clip = json.loads(metadata.read_text(encoding="utf-8"))["input"]
    video = next(stream for stream in streams if stream["codec_type"] == "video")
    origin = Fraction(info.get("format", {}).get("start_time", "0").replace("N/A", "0"))
    video_start = Fraction(video.get("start_time", str(origin)).replace("N/A", str(origin)))
    start = max(0, float(video_start - origin) + clip["start_time_seconds"])
    duration = clip["num_frames"] / Fraction(clip["fps"])
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.9f}",
         "-i", str(source), "-t", f"{float(duration):.9f}", "-map", "0:a:0",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)],
        capture_output=True, check=True, timeout=120,
    )


app.mount("/static", StaticFiles(directory=ROOT / "workbench" / "static"), name="static")

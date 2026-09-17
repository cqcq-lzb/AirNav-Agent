"""HTTP API that runs the complete GPU pipeline inside this container."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader

DATA = Path(os.getenv("API_DATA_DIR", "/data")).resolve()
JOBS = DATA / "jobs"
ENGINEERING = Path("/data2/home/wcq/nnUNet/nav_project/engineering_demo")
PIPELINE = ENGINEERING / "run_stage4_server_pipeline.py"
PYTHON = Path("/data0/home/wcq/.conda/envs/perm_test/bin/python")
TOKEN = os.getenv("API_TOKEN", "")
LIMIT = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024**3)))
SLOTS = threading.Semaphore(int(os.getenv("MAX_CONCURRENT_JOBS", "1")))

app = FastAPI(
    title="Airway Navigation Complete GPU Backend",
    version="2.0.0",
    description="Research use only; not a clinical medical-device service.",
)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_dir(jid: str) -> Path:
    return JOBS / jid


def state_file(jid: str) -> Path:
    return job_dir(jid) / "status.json"


def clean_case_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not value:
        raise ValueError("case_id must contain letters or numbers")
    return value


def inferred_case_id(filename: str) -> str:
    name = Path(filename).name
    if name.lower().endswith(".nii.gz"):
        name = name[:-7]
    elif name.lower().endswith(".nii"):
        name = name[:-4]
    else:
        raise ValueError("only .nii and .nii.gz CT files are supported")
    return clean_case_id(re.sub(r"_000\d$", "", name))


def read_state(jid: str) -> dict[str, Any]:
    path = state_file(jid)
    if not path.is_file():
        raise HTTPException(404, "job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(jid: str, data: dict[str, Any]) -> None:
    data["updated_at"] = now()
    path = state_file(jid)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_state(jid: str, **values: Any) -> None:
    data = read_state(jid)
    data.update(values)
    write_state(jid, data)


def append_log(jid: str, text: str) -> None:
    with (job_dir(jid) / "pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {text.rstrip()}\n")


def locate_package(case_id: str) -> Path:
    case_dir = ENGINEERING / "cases" / case_id
    candidates = [
        case_dir / f"{case_id}_stage4_package.zip",
        case_dir / f"{case_id}_ui_package.zip",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"pipeline package not found below {case_dir}")


def run_job(jid: str) -> None:
    with SLOTS:
        try:
            job = read_state(jid)
            cid = job["case_id"]
            gpu = int(job["gpu"])
            input_file = job_dir(jid) / "input" / job["input_filename"]
            command = [
                str(PYTHON), str(PIPELINE), "--ct", str(input_file),
                "--case-id", cid, "--gpu", str(gpu), "--overwrite",
            ]
            environment = os.environ.copy()
            update_state(jid, state="running", stage="processing")
            append_log(jid, "starting complete in-container GPU pipeline")
            process = subprocess.Popen(
                command,
                cwd=str(ENGINEERING),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                append_log(jid, line)
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"pipeline exited with code {return_code}")
            source = locate_package(cid)
            target = job_dir(jid) / "result" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            update_state(
                jid,
                state="succeeded",
                stage="complete",
                package_filename=target.name,
            )
            append_log(jid, "job completed")
        except Exception as exc:
            append_log(jid, f"ERROR {type(exc).__name__}: {exc}")
            update_state(
                jid,
                state="failed",
                stage="failed",
                error=f"{type(exc).__name__}: {exc}",
            )


def auth(api_key: str | None = Security(api_key_header)) -> None:
    if TOKEN and api_key != TOKEN:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")


@app.on_event("startup")
def initialize() -> None:
    JOBS.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "airway-navigation-complete-gpu-backend",
        "pipeline_present": PIPELINE.is_file(),
        "models_present": Path(os.environ["nnUNet_results"]).is_dir(),
    }


@app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(auth)])
async def create_job(
    ct: UploadFile = File(...),
    case_id_value: str | None = Form(None, alias="case_id"),
    gpu: int | None = Form(None),
) -> dict[str, Any]:
    filename = Path(ct.filename or "ct.nii.gz").name
    try:
        cid = clean_case_id(case_id_value) if case_id_value else inferred_case_id(filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    jid = uuid.uuid4().hex
    directory = job_dir(jid)
    target = directory / "input" / filename
    target.parent.mkdir(parents=True)
    size = 0
    try:
        with target.open("wb") as handle:
            while chunk := await ct.read(1024 * 1024):
                size += len(chunk)
                if size > LIMIT:
                    raise HTTPException(413, "upload exceeds configured size limit")
                handle.write(chunk)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await ct.close()
    record = {
        "job_id": jid,
        "case_id": cid,
        "gpu": int(gpu if gpu is not None else os.getenv("DEFAULT_GPU", "0")),
        "input_filename": filename,
        "input_bytes": size,
        "state": "queued",
        "stage": "queued",
        "created_at": now(),
        "updated_at": now(),
    }
    write_state(jid, record)
    threading.Thread(target=run_job, args=(jid,), daemon=True).start()
    return record


@app.get("/api/v1/jobs/{jid}", dependencies=[Depends(auth)])
def get_job(jid: str) -> dict[str, Any]:
    return read_state(jid)


@app.get("/api/v1/jobs/{jid}/logs", dependencies=[Depends(auth)])
def get_logs(jid: str) -> dict[str, str]:
    read_state(jid)
    path = job_dir(jid) / "pipeline.log"
    return {"logs": path.read_text(encoding="utf-8") if path.is_file() else ""}


@app.get("/api/v1/jobs/{jid}/package", dependencies=[Depends(auth)])
def get_package(jid: str) -> FileResponse:
    job = read_state(jid)
    if job.get("state") != "succeeded":
        raise HTTPException(409, "job has not completed")
    path = job_dir(jid) / "result" / job["package_filename"]
    if not path.is_file():
        raise HTTPException(404, "package unavailable")
    return FileResponse(path, media_type="application/zip", filename=path.name)

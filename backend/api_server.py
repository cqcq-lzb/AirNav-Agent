"""HTTP backend for the remote stage-4 research pipeline; no GUI dependencies."""
from __future__ import annotations

import json, os, posixpath, re, shlex, shutil, threading, time, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paramiko
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader

ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_DIR = Path(os.getenv("OPTIMIZER_DIR", "/opt/airway_navigation_v1/segmentation_optimization"))
DATA = Path(os.getenv("API_DATA_DIR", "/data")).resolve()
JOBS = DATA / "jobs"
CONFIG = Path(os.getenv("SERVER_CONFIG_PATH", ROOT / "server_config.json"))
TOKEN = os.getenv("API_TOKEN", "")
LIMIT = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024**3)))
app = FastAPI(title="Airway Navigation Pipeline API", version="1.0.0", description="Research use only; not a clinical medical-device service.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def now() -> str: return datetime.now(timezone.utc).isoformat()
def jid_dir(jid: str) -> Path: return JOBS / jid
def state_file(jid: str) -> Path: return jid_dir(jid) / "status.json"
def case_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not value: raise ValueError("case_id must contain letters or numbers")
    return value
def inferred(filename: str) -> str:
    name = Path(filename).name
    if name.lower().endswith(".nii.gz"): name = name[:-7]
    elif name.lower().endswith(".nii"): name = name[:-4]
    else: raise ValueError("only .nii and .nii.gz CT files are supported")
    return case_id(re.sub(r"_000\d$", "", name))
def read(jid: str) -> dict[str, Any]:
    path = state_file(jid)
    if not path.is_file(): raise HTTPException(404, "job not found")
    return json.loads(path.read_text(encoding="utf-8"))
def write(jid: str, data: dict[str, Any]) -> None:
    data["updated_at"] = now(); path = state_file(jid); tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(path)
def update(jid: str, **values: Any) -> None:
    data = read(jid); data.update(values); write(jid, data)
def log(jid: str, text: str) -> None:
    with (jid_dir(jid) / "pipeline.log").open("a", encoding="utf-8") as f: f.write(f"[{now()}] {text.rstrip()}\n")

def config() -> dict[str, Any]:
    if not CONFIG.is_file(): raise RuntimeError(f"server configuration not found: {CONFIG}")
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    env = {"server_host": os.getenv("AIRWAY_SERVER_HOST"), "server_user": os.getenv("AIRWAY_SERVER_USER"), "remote_engineering_dir": os.getenv("AIRWAY_REMOTE_ENGINEERING_DIR")}
    data.update({k:v for k,v in env.items() if v})
    if os.getenv("AIRWAY_SERVER_PORT"): data["server_port"] = int(os.environ["AIRWAY_SERVER_PORT"])
    missing = [k for k in ("server_host", "server_user", "remote_engineering_dir") if not data.get(k)]
    if missing: raise RuntimeError("missing server configuration: " + ", ".join(missing))
    return data
def mkdir_remote(sftp: Any, directory: str) -> None:
    current = ""
    for part in posixpath.normpath(directory).strip("/").split("/"):
        current += "/" + part
        try: sftp.stat(current)
        except OSError: sftp.mkdir(current)
def extract_safe(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True); root = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents: raise RuntimeError("unsafe downloaded ZIP path")
        archive.extractall(destination)
def ssh(data: dict[str, Any]) -> paramiko.SSHClient:
    client = paramiko.SSHClient(); client.load_system_host_keys()
    known = Path(os.getenv("SSH_KNOWN_HOSTS", "/root/.ssh/known_hosts"))
    if known.is_file(): client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    args: dict[str, Any] = {"hostname":data["server_host"], "port":int(data.get("server_port",22)), "username":data["server_user"], "timeout":20, "banner_timeout":30, "auth_timeout":30, "allow_agent":False, "look_for_keys":True}
    if os.getenv("SSH_KEY_PATH"): args["key_filename"] = os.environ["SSH_KEY_PATH"]
    client.connect(**args); return client
def command(data: dict[str, Any], ct: str, script: str, cid: str, gpu: int) -> str:
    choices = " ".join(shlex.quote(str(x)) for x in data.get("remote_python_candidates", []))
    lines = ["set -o pipefail", "PYTHON=''", f'for x in {choices}; do if [ -x "$x" ]; then PYTHON="$x"; break; fi; done', 'if [ -z "$PYTHON" ]; then exit 11; fi', 'export PATH="$(dirname "$PYTHON"):$PATH"', f"export nnUNet_raw={shlex.quote(data['nnunet_raw'])}", f"export nnUNet_preprocessed={shlex.quote(data['nnunet_preprocessed'])}", f"export nnUNet_results={shlex.quote(data['nnunet_results'])}", f"export CUDA_VISIBLE_DEVICES={gpu}", f'"$PYTHON" {shlex.quote(script)} --ct {shlex.quote(ct)} --case-id {shlex.quote(cid)} --gpu {gpu} --overwrite']
    return "bash -lc " + shlex.quote("; ".join(lines))
def run(jid: str) -> None:
    client = sftp = None
    try:
        job = read(jid); data = config(); cid = job["case_id"]; gpu = int(job["gpu"]); input_file = jid_dir(jid)/"input"/job["input_filename"]
        update(jid, state="running", stage="connecting"); client = ssh(data); sftp = client.open_sftp(); base = data["remote_engineering_dir"].rstrip("/")
        upload = posixpath.join(base,"uploads",cid); mkdir_remote(sftp,upload); remote_script = posixpath.join(base,"run_stage4_server_pipeline.py")
        sftp.put(str(ROOT/"run_stage4_server_pipeline.py"), remote_script)
        optimizer = OPTIMIZER_DIR
        for name in ("run_optimized_nnunet.py","optimize_segmentations.py","optimizer_config.json"): sftp.put(str(optimizer/name),posixpath.join(base,name))
        remote_ct = posixpath.join(upload, f"{cid}_0000.nii.gz" if input_file.name.lower().endswith(".nii.gz") else f"{cid}_0000.nii")
        update(jid, stage="uploading"); sftp.put(str(input_file),remote_ct); update(jid,stage="processing")
        _i, stdout, _e = client.exec_command(command(data,remote_ct,remote_script,cid,gpu),get_pty=True); channel = stdout.channel; pending=""
        while not channel.exit_status_ready() or channel.recv_ready():
            if channel.recv_ready():
                pending += channel.recv(65536).decode("utf-8",errors="replace")
                while "\n" in pending: line,pending=pending.split("\n",1); log(jid,line)
            else: time.sleep(.1)
        if pending: log(jid,pending)
        if channel.recv_exit_status() != 0: raise RuntimeError("remote pipeline failed; see job logs")
        update(jid,stage="downloading"); package=jid_dir(jid)/"result"/f"{cid}_stage4_package.zip"; package.parent.mkdir(parents=True,exist_ok=True)
        sftp.get(posixpath.join(base,"cases",cid,package.name),str(package)); extract_safe(package,package.parent/"stage4_package")
        update(jid,state="succeeded",stage="complete",package_filename=package.name); log(jid,"job completed")
    except Exception as exc:
        log(jid,f"ERROR {type(exc).__name__}: {exc}"); update(jid,state="failed",stage="failed",error=f"{type(exc).__name__}: {exc}")
    finally:
        if sftp: sftp.close()
        if client: client.close()
def auth(api_key: str | None = Security(api_key_header)) -> None:
    if TOKEN and api_key != TOKEN: raise HTTPException(status.HTTP_401_UNAUTHORIZED,"invalid API key")
@app.on_event("startup")
def init() -> None: JOBS.mkdir(parents=True,exist_ok=True)
@app.get("/health")
def health() -> dict[str,str]: return {"status":"ok","service":"airway-navigation-api"}
@app.post("/api/v1/jobs",status_code=status.HTTP_202_ACCEPTED,dependencies=[Depends(auth)])
async def create(ct: UploadFile=File(...), case_id_value: str|None=Form(None,alias="case_id"), gpu: int|None=Form(None)) -> dict[str,Any]:
    filename=Path(ct.filename or "ct.nii.gz").name
    try: cid=case_id(case_id_value) if case_id_value else inferred(filename); data=config()
    except (ValueError,RuntimeError) as exc: raise HTTPException(422, str(exc)) from exc
    jid=uuid.uuid4().hex; directory=jid_dir(jid); target=directory/"input"/filename; target.parent.mkdir(parents=True); size=0
    try:
        with target.open("wb") as f:
            while chunk:=await ct.read(1024*1024):
                size += len(chunk)
                if size>LIMIT: raise HTTPException(413,"upload exceeds configured size limit")
                f.write(chunk)
    except Exception: shutil.rmtree(directory,ignore_errors=True); raise
    finally: await ct.close()
    record={"job_id":jid,"case_id":cid,"gpu":int(gpu if gpu is not None else data.get("default_gpu",0)),"input_filename":filename,"input_bytes":size,"state":"queued","stage":"queued","created_at":now(),"updated_at":now()}; write(jid,record)
    threading.Thread(target=run,args=(jid,),daemon=True).start(); return record
@app.get("/api/v1/jobs/{jid}",dependencies=[Depends(auth)])
def get(jid:str)->dict[str,Any]: return read(jid)
@app.get("/api/v1/jobs/{jid}/logs",dependencies=[Depends(auth)])
def logs(jid:str)->dict[str,str]: read(jid); path=jid_dir(jid)/"pipeline.log"; return {"logs":path.read_text(encoding="utf-8") if path.is_file() else ""}
@app.get("/api/v1/jobs/{jid}/package",dependencies=[Depends(auth)])
def package(jid:str)->FileResponse:
    job=read(jid)
    if job.get("state")!="succeeded": raise HTTPException(409,"job has not completed")
    path=jid_dir(jid)/"result"/job["package_filename"]
    if not path.is_file(): raise HTTPException(404,"package unavailable")
    return FileResponse(path,media_type="application/zip",filename=path.name)

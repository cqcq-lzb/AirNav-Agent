"""Qt worker that makes the desktop launcher use the Docker HTTP backend."""
from __future__ import annotations

import http.client
import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

from PySide6.QtCore import QThread, Signal


class ApiPipelineWorker(QThread):
    logMessage = Signal(str)
    stageChanged = Signal(str)
    uploadProgress = Signal(int)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, *, api_url: str, api_token: str, gpu: int, ct_path: Path, case_id: str, local_case_root: Path) -> None:
        super().__init__()
        self.api_url = api_url.rstrip("/")
        self.api_token = api_token
        self.gpu = gpu
        self.ct_path = ct_path
        self.case_id = case_id
        self.local_case_root = local_case_root

    def _parts(self) -> tuple[str, str, type[http.client.HTTPConnection]]:
        parsed = urlsplit(self.api_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Docker 后端地址必须是 http://主机:端口")
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        return parsed.netloc, parsed.path.rstrip("/"), connection_type

    def _connection(self) -> tuple[http.client.HTTPConnection, str]:
        host, prefix, connection_type = self._parts()
        return connection_type(host, timeout=60), prefix

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_token, "Accept": "application/json"}

    def _json(self, method: str, endpoint: str) -> dict:
        connection, prefix = self._connection()
        try:
            connection.request(method, prefix + endpoint, headers=self._headers())
            response = connection.getresponse()
            payload = response.read().decode("utf-8", errors="replace")
            if response.status >= 300:
                raise RuntimeError(f"API {response.status}: {payload}")
            return json.loads(payload)
        finally:
            connection.close()

    def _submit(self) -> str:
        boundary = "----AirwayApi" + uuid.uuid4().hex
        filename = self.ct_path.name
        preamble = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"case_id\"\r\n\r\n{self.case_id}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"gpu\"\r\n\r\n{self.gpu}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"ct\"; filename=\"{filename}\"\r\n"
            "Content-Type: application/gzip\r\n\r\n"
        ).encode("utf-8")
        ending = f"\r\n--{boundary}--\r\n".encode("utf-8")
        total = self.ct_path.stat().st_size
        connection, prefix = self._connection()
        headers = self._headers()
        headers.update({"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(preamble) + total + len(ending))})
        try:
            connection.putrequest("POST", prefix + "/api/v1/jobs")
            for key, value in headers.items():
                connection.putheader(key, value)
            connection.endheaders()
            connection.send(preamble)
            transferred = 0
            with self.ct_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    connection.send(chunk)
                    transferred += len(chunk)
                    self.uploadProgress.emit(int(100 * transferred / total) if total else 100)
            connection.send(ending)
            response = connection.getresponse()
            payload = response.read().decode("utf-8", errors="replace")
            if response.status != 202:
                raise RuntimeError(f"API {response.status}: {payload}")
            return str(json.loads(payload)["job_id"])
        finally:
            connection.close()

    def _download(self, job_id: str, target: Path) -> None:
        connection, prefix = self._connection()
        try:
            connection.request("GET", prefix + f"/api/v1/jobs/{job_id}/package", headers=self._headers())
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"API {response.status}: {response.read().decode('utf-8', errors='replace')}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
        finally:
            connection.close()

    @staticmethod
    def _extract(zip_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        root = destination.resolve()
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError("下载包中存在非法路径")
            archive.extractall(destination)

    def run(self) -> None:
        try:
            self.stageChanged.emit("正在上传 CT 到 Docker 后端……")
            self.logMessage.emit(f"Docker API：{self.api_url}")
            job_id = self._submit()
            self.logMessage.emit(f"任务已创建：{job_id}")
            self.stageChanged.emit("Docker 后端正在调用 GPU 服务器处理……")
            while True:
                job = self._json("GET", f"/api/v1/jobs/{job_id}")
                state = str(job.get("state", ""))
                stage = str(job.get("stage", ""))
                self.stageChanged.emit(f"后端任务：{stage or state}")
                if state == "succeeded":
                    break
                if state == "failed":
                    logs = self._json("GET", f"/api/v1/jobs/{job_id}/logs").get("logs", "")
                    raise RuntimeError(str(job.get("error", "后端任务失败")) + "\n\n" + str(logs)[-4000:])
                time.sleep(2)
            case_dir = self.local_case_root / self.case_id
            zip_path = case_dir / f"{self.case_id}_stage4_package.zip"
            stage4_dir = case_dir / "stage4_package"
            self.stageChanged.emit("正在从 Docker 后端下载病例包……")
            self._download(job_id, zip_path)
            if stage4_dir.exists():
                shutil.rmtree(stage4_dir)
            self.stageChanged.emit("正在解压病例包……")
            self._extract(zip_path, stage4_dir)
            required = [stage4_dir / "ct.nii.gz", stage4_dir / "airway_mask.nii.gz", stage4_dir / "nodule_raw.nii.gz"]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError("病例包不完整：\n" + "\n".join(missing))
            self.stageChanged.emit("处理完成")
            self.completed.emit(str(stage4_dir))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
经支气管肺结节自动导航系统 v3（Windows 客户端）
================================================

完整流程
--------
1. 用户选择一个胸部 CT NIfTI 文件；
2. 客户端上传到本机 Docker API 后端；
3. Docker 后端通过 SSH 调用 GPU 服务器流水线；
4. GPU 服务器完成气道和结节分割；
5. Docker 后端下载并提供 stage4_package.zip；
6. 客户端自动下载并解压；
7. 自动打开第四阶段交互导航界面；
8. 在本地点击不同结节即可重新规划，不需要重新分割。

放置位置
--------
将本文件与以下文件放在同一个 PyCharm 工程目录：
- interactive_navigation_stage4_v3.py
- run_stage4_server_pipeline.py
- server_config.json

推荐路径：
D:\airway_navigation_v1\

依赖
----
PySide6
paramiko
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

from api_pipeline_worker import ApiPipelineWorker

try:
    import paramiko
except ImportError:
    paramiko = None  # type: ignore[assignment]

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


WINDOW_TITLE = "经支气管肺结节自动导航系统 v3.2"

DEFAULT_CONFIG: dict[str, Any] = {
    "server_host": "192.168.8.41",
    "server_port": 22,
    "server_user": "wcq",
    "default_gpu": 5,
    "remote_engineering_dir": (
        "/data2/home/wcq/nnUNet/nav_project/engineering_demo"
    ),
    "remote_python_candidates": [
        "/data0/home/wcq/.conda/envs/perm_test/bin/python",
        "/data0/home/wcq/conda/envs/perm_test/bin/python",
        "/data2/home/wcq/conda/envs/perm_test/bin/python",
        "/data2/home/wcq/.conda/envs/perm_test/bin/python",
    ],
    "nnunet_raw": (
        "/data2/home/wcq/nnUNet/nnUNet_raw"
    ),
    "nnunet_preprocessed": (
        "/data2/home/wcq/nnUNet/nnUNet_preprocessed"
    ),
    "nnunet_results": (
        "/data2/home/wcq/nnUNet/nnUNet_results"
    ),
    "backend_api_url": "http://127.0.0.1:8000",
}


def sanitize_case_id(value: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value.strip(),
    )
    cleaned = cleaned.strip("._-")

    if not cleaned:
        raise ValueError("病例编号不能为空")

    return cleaned


def infer_case_id(ct_path: Path) -> str:
    name = ct_path.name

    if name.lower().endswith(
        ".nii.gz"
    ):
        name = name[:-7]
    elif name.lower().endswith(
        ".nii"
    ):
        name = name[:-4]

    name = re.sub(
        r"_000\d$",
        "",
        name,
    )

    return sanitize_case_id(name)


def load_config(
    config_path: Path,
) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)

    if config_path.is_file():
        loaded = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            loaded,
            dict,
        ):
            raise ValueError(
                "server_config.json 必须是 JSON 对象"
            )

        config.update(loaded)

    return config


def save_config(
    config_path: Path,
    config: dict[str, Any],
) -> None:
    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def ensure_remote_directory(
    sftp: Any,
    remote_directory: str,
) -> None:
    """
    递归创建服务器目录。只处理 Linux POSIX 路径。
    """
    normalized = posixpath.normpath(
        remote_directory
    )

    if normalized == "/":
        return

    parts = normalized.strip(
        "/"
    ).split("/")

    current = ""

    for part in parts:
        current += "/" + part

        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def safe_extract_zip(
    zip_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_root = output_dir.resolve()

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:
        for member in archive.infolist():
            target = (
                output_dir
                / member.filename
            ).resolve()

            if (
                output_root != target
                and output_root
                not in target.parents
            ):
                raise RuntimeError(
                    "ZIP 中存在非法路径："
                    f"{member.filename}"
                )

        archive.extractall(
            output_dir
        )


class RemotePipelineWorker(QThread):
    logMessage = Signal(str)
    stageChanged = Signal(str)
    uploadProgress = Signal(int)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        gpu: int,
        ct_path: Path,
        case_id: str,
        local_case_root: Path,
        engineering_dir: str,
        python_candidates: list[str],
        nnunet_raw: str,
        nnunet_preprocessed: str,
        nnunet_results: str,
        local_server_script: Path,
        local_optimizer_scripts: list[Path],
    ) -> None:
        super().__init__()

        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.gpu = gpu
        self.ct_path = ct_path
        self.case_id = case_id
        self.local_case_root = (
            local_case_root
        )
        self.engineering_dir = (
            engineering_dir.rstrip("/")
        )
        self.python_candidates = (
            python_candidates
        )
        self.nnunet_raw = nnunet_raw
        self.nnunet_preprocessed = (
            nnunet_preprocessed
        )
        self.nnunet_results = nnunet_results
        self.local_server_script = (
            local_server_script
        )
        self.local_optimizer_scripts = local_optimizer_scripts

    def emit_log(
        self,
        text: str,
    ) -> None:
        self.logMessage.emit(
            text.rstrip("\n")
        )

    def upload_callback(
        self,
        transferred: int,
        total: int,
    ) -> None:
        if total <= 0:
            return

        percent = int(
            round(
                100.0
                * transferred
                / total
            )
        )
        self.uploadProgress.emit(
            max(
                0,
                min(percent, 100),
            )
        )

    def build_remote_command(
        self,
        remote_ct_path: str,
        remote_script_path: str,
    ) -> str:
        python_values = " ".join(
            shlex.quote(path)
            for path in self.python_candidates
        )

        command_lines = [
            "set -o pipefail",
            "PYTHON=''",
            (
                f"for candidate in {python_values}; do "
                'if [ -x "$candidate" ]; then '
                'PYTHON="$candidate"; break; fi; '
                "done"
            ),
            (
                'if [ -z "$PYTHON" ]; then '
                'echo "REMOTE_PYTHON_NOT_FOUND"; '
                "exit 11; fi"
            ),
            'echo "REMOTE_PYTHON=$PYTHON"',
            'ENV_BIN="$(dirname "$PYTHON")"',
            'export PATH="$ENV_BIN:$PATH"',
            'export CONDA_PREFIX="$(dirname "$ENV_BIN")"',
            'export CONDA_DEFAULT_ENV="$(basename "$CONDA_PREFIX")"',
            "hash -r",
            'echo "REMOTE_ENV_BIN=$ENV_BIN"',
            'echo "REMOTE_CONDA_PREFIX=$CONDA_PREFIX"',
            (
                'if ! command -v nnUNetv2_predict >/dev/null 2>&1; then '
                'echo "NNUNET_PREDICT_NOT_FOUND_AFTER_PATH_FIX"; '
                'echo "PATH=$PATH"; '
                'ls -lh "$ENV_BIN"/nnUNetv2_* 2>/dev/null || true; '
                "exit 12; fi"
            ),
            'echo "REMOTE_NNUNET_PREDICT=$(command -v nnUNetv2_predict)"',
            (
                "export nnUNet_raw="
                + shlex.quote(
                    self.nnunet_raw
                )
            ),
            (
                "export nnUNet_preprocessed="
                + shlex.quote(
                    self.nnunet_preprocessed
                )
            ),
            (
                "export nnUNet_results="
                + shlex.quote(
                    self.nnunet_results
                )
            ),
            (
                "export CUDA_VISIBLE_DEVICES="
                + shlex.quote(
                    str(self.gpu)
                )
            ),
            "export PYTHONUNBUFFERED=1",
            (
                '"$PYTHON" '
                + shlex.quote(
                    remote_script_path
                )
                + " --ct "
                + shlex.quote(
                    remote_ct_path
                )
                + " --case-id "
                + shlex.quote(
                    self.case_id
                )
                + " --gpu "
                + shlex.quote(
                    str(self.gpu)
                )
                + " --overwrite"
            ),
        ]

        inner = "; ".join(
            command_lines
        )

        return (
            "bash -lc "
            + shlex.quote(inner)
        )

    def stream_channel(
        self,
        channel: Any,
    ) -> int:
        pending = ""

        while True:
            received = False

            if channel.recv_ready():
                data = channel.recv(
                    65536
                ).decode(
                    "utf-8",
                    errors="replace",
                )
                pending += data
                received = True

                while "\n" in pending:
                    line, pending = (
                        pending.split(
                            "\n",
                            1,
                        )
                    )
                    self.emit_log(line)

            if (
                channel.exit_status_ready()
                and not channel.recv_ready()
            ):
                break

            if not received:
                time.sleep(0.06)

        if pending:
            self.emit_log(pending)

        return int(
            channel.recv_exit_status()
        )

    def run(self) -> None:
        if paramiko is None:
            self.failed.emit(
                "当前 Python 环境没有安装 paramiko。\n"
                "请在 dicom 环境安装后重新运行。"
            )
            return

        client = None
        sftp = None

        try:
            self.stageChanged.emit(
                "正在连接 GPU 服务器……"
            )
            self.emit_log(
                "=" * 78
            )
            self.emit_log(
                "经支气管肺结节自动导航系统 v3"
            )
            self.emit_log(
                f"服务器：{self.host}:{self.port}"
            )
            self.emit_log(
                f"用户：{self.username}"
            )
            self.emit_log(
                f"GPU：{self.gpu}"
            )
            self.emit_log(
                f"病例：{self.case_id}"
            )
            self.emit_log(
                "=" * 78
            )

            client = (
                paramiko.SSHClient()
            )
            client.set_missing_host_key_policy(
                paramiko.AutoAddPolicy()
            )

            connect_args: dict[
                str,
                Any,
            ] = {
                "hostname": self.host,
                "port": self.port,
                "username": self.username,
                "timeout": 15,
                "banner_timeout": 20,
                "auth_timeout": 20,
                "look_for_keys": (
                    not bool(
                        self.password
                    )
                ),
                "allow_agent": True,
            }

            if self.password:
                connect_args[
                    "password"
                ] = self.password
                connect_args[
                    "look_for_keys"
                ] = False
                connect_args[
                    "allow_agent"
                ] = False

            client.connect(
                **connect_args
            )

            transport = (
                client.get_transport()
            )

            if transport is not None:
                transport.set_keepalive(30)

            self.emit_log(
                "SSH 连接成功。"
            )

            sftp = client.open_sftp()

            remote_upload_dir = (
                posixpath.join(
                    self.engineering_dir,
                    "uploads",
                    self.case_id,
                )
            )
            ensure_remote_directory(
                sftp,
                remote_upload_dir,
            )

            remote_script_path = (
                posixpath.join(
                    self.engineering_dir,
                    "run_stage4_server_pipeline.py",
                )
            )

            self.stageChanged.emit(
                "正在同步服务器流水线脚本……"
            )
            self.emit_log(
                "上传服务器脚本："
                f"{remote_script_path}"
            )

            sftp.put(
                str(
                    self.local_server_script
                ),
                remote_script_path,
            )

            try:
                sftp.chmod(
                    remote_script_path,
                    0o755,
                )
            except OSError:
                pass

            for local_optimizer_script in self.local_optimizer_scripts:
                if not local_optimizer_script.is_file():
                    raise FileNotFoundError(
                        f"缺少分割优化脚本：{local_optimizer_script}"
                    )
                remote_optimizer_path = posixpath.join(
                    self.engineering_dir,
                    local_optimizer_script.name,
                )
                self.emit_log(
                    "上传分割优化脚本："
                    f"{remote_optimizer_path}"
                )
                sftp.put(str(local_optimizer_script), remote_optimizer_path)
                try:
                    sftp.chmod(remote_optimizer_path, 0o755)
                except OSError:
                    pass

            lower_name = (
                self.ct_path.name.lower()
            )

            if lower_name.endswith(
                ".nii.gz"
            ):
                remote_ct_name = (
                    f"{self.case_id}_0000.nii.gz"
                )
            elif lower_name.endswith(
                ".nii"
            ):
                remote_ct_name = (
                    f"{self.case_id}_0000.nii"
                )
            else:
                raise ValueError(
                    "当前只支持 .nii 或 .nii.gz CT"
                )

            remote_ct_path = (
                posixpath.join(
                    remote_upload_dir,
                    remote_ct_name,
                )
            )

            self.stageChanged.emit(
                "正在上传 CT……"
            )
            self.emit_log(
                "本地 CT："
                f"{self.ct_path}"
            )
            self.emit_log(
                "服务器 CT："
                f"{remote_ct_path}"
            )

            sftp.put(
                str(self.ct_path),
                remote_ct_path,
                callback=(
                    self.upload_callback
                ),
            )
            self.uploadProgress.emit(100)
            self.emit_log(
                "CT 上传完成。"
            )

            self.stageChanged.emit(
                "服务器正在分割、质控和生成病例包……"
            )

            remote_command = (
                self.build_remote_command(
                    remote_ct_path,
                    remote_script_path,
                )
            )

            self.emit_log("")
            self.emit_log(
                "开始执行服务器流水线："
            )
            self.emit_log(
                remote_command
            )
            self.emit_log("")

            _stdin, stdout, _stderr = (
                client.exec_command(
                    remote_command,
                    get_pty=True,
                )
            )

            exit_code = (
                self.stream_channel(
                    stdout.channel
                )
            )

            self.emit_log("")
            self.emit_log(
                f"服务器流水线退出代码："
                f"{exit_code}"
            )

            remote_case_dir = (
                posixpath.join(
                    self.engineering_dir,
                    "cases",
                    self.case_id,
                )
            )

            if exit_code != 0:
                remote_status_path = (
                    posixpath.join(
                        self.engineering_dir,
                        "pipeline_status",
                        (
                            f"{self.case_id}"
                            "_stage4.json"
                        ),
                    )
                )

                status_detail = ""

                try:
                    with sftp.file(
                        remote_status_path,
                        "r",
                    ) as status_file:
                        raw_status = (
                            status_file.read()
                        )

                    if isinstance(
                        raw_status,
                        bytes,
                    ):
                        raw_status = (
                            raw_status.decode(
                                "utf-8",
                                errors="replace",
                            )
                        )

                    status_data = json.loads(
                        raw_status
                    )

                    status_detail = str(
                        status_data.get(
                            "error",
                            "",
                        )
                    ).strip()

                    remote_traceback = str(
                        status_data.get(
                            "traceback",
                            "",
                        )
                    ).strip()

                    if remote_traceback:
                        status_detail += (
                            "\n\n服务器 traceback：\n"
                            + remote_traceback
                        )

                except Exception as status_error:
                    status_detail = (
                        "无法读取服务器状态文件："
                        f"{type(status_error).__name__}: "
                        f"{status_error}"
                    )

                raise RuntimeError(
                    "服务器自动处理失败，"
                    f"退出代码：{exit_code}"
                    + (
                        "\n\n" + status_detail
                        if status_detail
                        else ""
                    )
                )
            remote_zip_path = (
                posixpath.join(
                    remote_case_dir,
                    (
                        f"{self.case_id}"
                        "_stage4_package.zip"
                    ),
                )
            )

            self.stageChanged.emit(
                "正在下载第四阶段病例包……"
            )
            self.emit_log(
                "下载："
                f"{remote_zip_path}"
            )

            local_case_dir = (
                self.local_case_root
                / self.case_id
            )
            local_case_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            local_zip_path = (
                local_case_dir
                / (
                    f"{self.case_id}"
                    "_stage4_package.zip"
                )
            )

            sftp.get(
                remote_zip_path,
                str(local_zip_path),
            )

            self.emit_log(
                "病例包下载完成："
                f"{local_zip_path}"
            )

            stage4_dir = (
                local_case_dir
                / "stage4_package"
            )

            if stage4_dir.exists():
                shutil.rmtree(
                    stage4_dir
                )

            self.stageChanged.emit(
                "正在解压本地病例包……"
            )
            safe_extract_zip(
                local_zip_path,
                stage4_dir,
            )

            required_files = [
                stage4_dir
                / "ct.nii.gz",
                stage4_dir
                / "airway_mask.nii.gz",
                stage4_dir
                / "nodule_raw.nii.gz",
            ]

            missing = [
                path
                for path in required_files
                if not path.is_file()
            ]

            if missing:
                raise RuntimeError(
                    "下载病例包不完整，缺少：\n"
                    + "\n".join(
                        str(path)
                        for path in missing
                    )
                )

            self.emit_log(
                "本地病例目录："
                f"{stage4_dir}"
            )
            self.emit_log(
                "服务器分割与本地准备全部完成。"
            )

            self.stageChanged.emit(
                "处理完成"
            )
            self.completed.emit(
                str(stage4_dir)
            )

        except Exception as error:
            message = (
                f"{type(error).__name__}: "
                f"{error}\n\n"
                f"{traceback.format_exc()}"
            )
            self.failed.emit(message)

        finally:
            try:
                if sftp is not None:
                    sftp.close()
            except Exception:
                pass

            try:
                if client is not None:
                    client.close()
            except Exception:
                pass


class NavigationLauncherWindow(
    QMainWindow
):
    def __init__(self) -> None:
        super().__init__()

        self.project_dir = Path(
            __file__
        ).resolve().parent

        self.config_path = (
            self.project_dir
            / "server_config.json"
        )
        self.ui_script_path = (
            self.project_dir
            / "interactive_navigation_stage4_v3.py"
        )
        self.server_script_path = (
            self.project_dir
            / "run_stage4_server_pipeline.py"
        )
        optimizer_dir = (
            self.project_dir.parent
            / "segmentation_optimization"
        )
        self.optimizer_script_paths = [
            optimizer_dir / "run_optimized_nnunet.py",
            optimizer_dir / "optimize_segmentations.py",
            optimizer_dir / "optimizer_config.json",
        ]

        try:
            self.config = load_config(
                self.config_path
            )
        except Exception as error:
            QMessageBox.warning(
                self,
                "配置文件读取失败",
                str(error),
            )
            self.config = dict(
                DEFAULT_CONFIG
            )

        self.worker: QThread | None = None

        self.last_stage4_dir: (
            Path
            | None
        ) = None

        self.setWindowTitle(
            WINDOW_TITLE
        )
        self.resize(1060, 760)
        self.setMinimumSize(
            900,
            680,
        )

        self.build_ui()
        self.apply_style()

    def build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(
            18,
            16,
            18,
            16,
        )
        layout.setSpacing(12)

        header = QFrame()
        header_layout = QVBoxLayout(
            header
        )

        title = QLabel(
            "经支气管肺结节自动导航系统"
        )
        title.setObjectName(
            "TitleLabel"
        )

        subtitle = QLabel(
            "选择一份胸部 CT，Docker 后端自动完成气道与结节分割，"
            "下载后进入全部结节候选交互导航"
        )
        subtitle.setObjectName(
            "SubtitleLabel"
        )

        header_layout.addWidget(
            title
        )
        header_layout.addWidget(
            subtitle
        )
        layout.addWidget(header)

        form_frame = QFrame()
        form_layout = QGridLayout(
            form_frame
        )
        form_layout.setContentsMargins(
            16,
            14,
            16,
            14,
        )
        form_layout.setHorizontalSpacing(
            10
        )
        form_layout.setVerticalSpacing(
            10
        )

        self.ct_edit = QLineEdit()
        self.ct_edit.setPlaceholderText(
            "选择 .nii 或 .nii.gz 胸部 CT"
        )

        choose_ct_button = QPushButton(
            "选择 CT"
        )
        choose_ct_button.clicked.connect(
            self.choose_ct
        )

        form_layout.addWidget(
            QLabel("CT 文件"),
            0,
            0,
        )
        form_layout.addWidget(
            self.ct_edit,
            0,
            1,
        )
        form_layout.addWidget(
            choose_ct_button,
            0,
            2,
        )

        self.case_id_edit = QLineEdit()
        self.case_id_edit.setPlaceholderText(
            "例如 LIDC_0073"
        )

        form_layout.addWidget(
            QLabel("病例编号"),
            1,
            0,
        )
        form_layout.addWidget(
            self.case_id_edit,
            1,
            1,
            1,
            2,
        )

        self.api_url_edit = QLineEdit(
            str(
                self.config[
                    "backend_api_url"
                ]
            )
        )

        form_layout.addWidget(
            QLabel("Docker 后端地址"),
            2,
            0,
        )
        form_layout.addWidget(
            self.api_url_edit,
            2,
            1,
            1,
            2,
        )

        self.api_token_edit = QLineEdit()
        self.api_token_edit.setEchoMode(
            QLineEdit.EchoMode.Password
        )
        self.api_token_edit.setPlaceholderText(
            "docker/airway_gpu_backend_bundle/.env 里的 API_TOKEN；仅本次运行使用"
        )

        form_layout.addWidget(
            QLabel("API 令牌"),
            3,
            0,
        )
        form_layout.addWidget(
            self.api_token_edit,
            3,
            1,
            1,
            2,
        )

        self.gpu_spin = QSpinBox()
        self.gpu_spin.setRange(
            0,
            31,
        )
        self.gpu_spin.setValue(
            int(
                self.config[
                    "default_gpu"
                ]
            )
        )

        form_layout.addWidget(
            QLabel("远端 GPU"),
            4,
            0,
        )
        form_layout.addWidget(
            self.gpu_spin,
            4,
            1,
            1,
            2,
        )

        default_case_root = (
            self.project_dir
            / "cases"
        )

        self.local_root_edit = (
            QLineEdit(
                str(
                    default_case_root
                )
            )
        )

        choose_root_button = QPushButton(
            "选择目录"
        )
        choose_root_button.clicked.connect(
            self.choose_local_root
        )

        form_layout.addWidget(
            QLabel("本地病例目录"),
            5,
            0,
        )
        form_layout.addWidget(
            self.local_root_edit,
            5,
            1,
        )
        form_layout.addWidget(
            choose_root_button,
            5,
            2,
        )

        self.auto_open_checkbox = (
            QCheckBox(
                "完成后自动打开交互导航界面"
            )
        )
        self.auto_open_checkbox.setChecked(
            True
        )
        form_layout.addWidget(
            self.auto_open_checkbox,
            6,
            1,
            1,
            2,
        )

        layout.addWidget(
            form_frame
        )

        button_row = QHBoxLayout()

        self.start_button = QPushButton(
            "开始 Docker 后端处理并打开导航"
        )
        self.start_button.setObjectName(
            "StartButton"
        )
        self.start_button.clicked.connect(
            self.start_pipeline
        )
        button_row.addWidget(
            self.start_button
        )

        self.open_navigation_button = (
            QPushButton(
                "打开导航界面"
            )
        )
        self.open_navigation_button.setEnabled(
            False
        )
        self.open_navigation_button.clicked.connect(
            self.open_navigation
        )
        button_row.addWidget(
            self.open_navigation_button
        )

        self.open_folder_button = (
            QPushButton(
                "打开结果目录"
            )
        )
        self.open_folder_button.setEnabled(
            False
        )
        self.open_folder_button.clicked.connect(
            self.open_result_folder
        )
        button_row.addWidget(
            self.open_folder_button
        )

        button_row.addStretch(1)

        layout.addLayout(
            button_row
        )

        self.stage_label = QLabel(
            "等待开始"
        )
        self.stage_label.setObjectName(
            "StageLabel"
        )
        layout.addWidget(
            self.stage_label
        )

        self.progress_bar = (
            QProgressBar()
        )
        self.progress_bar.setRange(
            0,
            100,
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(
            "等待上传"
        )
        layout.addWidget(
            self.progress_bar
        )

        log_title = QLabel(
            "处理日志"
        )
        log_title.setObjectName(
            "SectionLabel"
        )
        layout.addWidget(
            log_title
        )

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(
            QTextEdit.LineWrapMode.NoWrap
        )
        layout.addWidget(
            self.log_edit,
            1,
        )

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #06131d;
                color: #d9eef8;
                font-family: "Microsoft YaHei";
                font-size: 13px;
            }

            QFrame {
                background-color: #0a1c28;
                border: 1px solid #24485d;
                border-radius: 10px;
            }

            QLabel#TitleLabel {
                color: white;
                font-size: 25px;
                font-weight: 800;
            }

            QLabel#SubtitleLabel {
                color: #8bcfe8;
                font-size: 12px;
            }

            QLabel#StageLabel {
                color: #7ce0ff;
                font-size: 14px;
                font-weight: 800;
            }

            QLabel#SectionLabel {
                color: #67d6ff;
                font-size: 14px;
                font-weight: 800;
            }

            QLineEdit, QSpinBox {
                background-color: #06131d;
                border: 1px solid #2b617e;
                border-radius: 6px;
                padding: 7px;
                selection-background-color: #1f789b;
            }

            QPushButton {
                background-color: #12374d;
                border: 1px solid #2b6d8c;
                border-radius: 7px;
                padding: 8px 14px;
                color: #ecfaff;
                font-weight: 700;
            }

            QPushButton:hover {
                background-color: #1b5775;
                border-color: #47c5f4;
            }

            QPushButton:disabled {
                color: #67808d;
                background-color: #10232e;
                border-color: #233b49;
            }

            QPushButton#StartButton {
                background-color: #087bab;
                border-color: #20b9ed;
                font-size: 14px;
                padding: 10px 20px;
            }

            QPushButton#StartButton:hover {
                background-color: #0b96ca;
            }

            QProgressBar {
                border: 1px solid #2b617e;
                border-radius: 7px;
                background-color: #0a202d;
                text-align: center;
                min-height: 24px;
            }

            QProgressBar::chunk {
                background-color: #24b9e8;
                border-radius: 6px;
            }

            QTextEdit {
                background-color: #020b10;
                border: 1px solid #24485d;
                color: #bceeff;
                font-family: Consolas;
                font-size: 12px;
            }
            """
        )

    def choose_ct(self) -> None:
        selected, _filter = (
            QFileDialog.getOpenFileName(
                self,
                "选择胸部 CT",
                str(
                    self.project_dir
                ),
                (
                    "NIfTI CT "
                    "(*.nii.gz *.nii);;"
                    "全部文件 (*.*)"
                ),
            )
        )

        if not selected:
            return

        path = Path(selected)
        self.ct_edit.setText(
            str(path)
        )

        try:
            self.case_id_edit.setText(
                infer_case_id(path)
            )
        except ValueError:
            pass

    def choose_local_root(
        self,
    ) -> None:
        selected = (
            QFileDialog.getExistingDirectory(
                self,
                "选择本地病例根目录",
                self.local_root_edit.text(),
            )
        )

        if selected:
            self.local_root_edit.setText(
                selected
            )

    def append_log(
        self,
        message: str,
    ) -> None:
        self.log_edit.append(
            message
        )
        scrollbar = (
            self.log_edit
            .verticalScrollBar()
        )
        scrollbar.setValue(
            scrollbar.maximum()
        )

    def update_stage(
        self,
        message: str,
    ) -> None:
        self.stage_label.setText(
            message
        )

        if (
            "上传 CT" in message
        ):
            self.progress_bar.setRange(
                0,
                100,
            )
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(
                "CT 上传 %p%"
            )
        elif (
            "处理完成" in message
        ):
            self.progress_bar.setRange(
                0,
                100,
            )
            self.progress_bar.setValue(
                100
            )
            self.progress_bar.setFormat(
                "完成"
            )
        else:
            self.progress_bar.setRange(
                0,
                0,
            )
            self.progress_bar.setFormat(
                message
            )

    def update_upload_progress(
        self,
        value: int,
    ) -> None:
        self.progress_bar.setRange(
            0,
            100,
        )
        self.progress_bar.setValue(
            value
        )
        self.progress_bar.setFormat(
            "CT 上传 %p%"
        )

    def validate_inputs(
        self,
    ) -> tuple[
        Path,
        str,
        Path,
    ]:
        ct_path = Path(
            self.ct_edit.text().strip()
        )

        if not ct_path.is_file():
            raise FileNotFoundError(
                "请选择存在的 CT 文件"
            )

        lower_name = (
            ct_path.name.lower()
        )

        if not (
            lower_name.endswith(
                ".nii"
            )
            or lower_name.endswith(
                ".nii.gz"
            )
        ):
            raise ValueError(
                "当前只支持 .nii 或 .nii.gz"
            )

        case_id = sanitize_case_id(
            self.case_id_edit.text()
        )

        local_root = Path(
            self.local_root_edit.text().strip()
        )

        local_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.ui_script_path.is_file():
            raise FileNotFoundError(
                "缺少本地交互导航脚本：\n"
                f"{self.ui_script_path}"
            )

        return (
            ct_path,
            case_id,
            local_root,
        )

    def start_pipeline(self) -> None:
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            return

        try:
            (
                ct_path,
                case_id,
                local_root,
            ) = self.validate_inputs()

            api_url = (
                self.api_url_edit
                .text()
                .strip()
            )
            api_token = (
                self.api_token_edit
                .text()
                .strip()
            )

            if not api_url:
                raise ValueError(
                    "Docker 后端地址不能为空"
                )

            if not api_token:
                raise ValueError(
                    "API 令牌不能为空；请从 docker/.env.api 复制 API_TOKEN"
                )

            # 保存后端地址和 GPU；API 令牌不保存到配置文件。
            self.config[
                "backend_api_url"
            ] = api_url
            self.config[
                "default_gpu"
            ] = self.gpu_spin.value()

            save_config(
                self.config_path,
                self.config,
            )

            self.last_stage4_dir = None
            self.open_navigation_button.setEnabled(
                False
            )
            self.open_folder_button.setEnabled(
                False
            )
            self.start_button.setEnabled(
                False
            )
            self.log_edit.clear()

            self.worker = (
                ApiPipelineWorker(
                    api_url=api_url,
                    api_token=api_token,
                    gpu=(
                        self.gpu_spin.value()
                    ),
                    ct_path=ct_path,
                    case_id=case_id,
                    local_case_root=(
                        local_root
                    ),
                )
            )

            self.worker.logMessage.connect(
                self.append_log
            )
            self.worker.stageChanged.connect(
                self.update_stage
            )
            self.worker.uploadProgress.connect(
                self.update_upload_progress
            )
            self.worker.completed.connect(
                self.pipeline_completed
            )
            self.worker.failed.connect(
                self.pipeline_failed
            )
            self.worker.start()

        except Exception as error:
            QMessageBox.critical(
                self,
                "无法开始",
                f"{type(error).__name__}: "
                f"{error}",
            )

    def pipeline_completed(
        self,
        stage4_dir: str,
    ) -> None:
        self.start_button.setEnabled(
            True
        )
        self.progress_bar.setRange(
            0,
            100,
        )
        self.progress_bar.setValue(
            100
        )
        self.progress_bar.setFormat(
            "完成"
        )
        self.stage_label.setText(
            "Docker 后端处理完成，病例包已下载"
        )

        self.last_stage4_dir = Path(
            stage4_dir
        )

        self.open_navigation_button.setEnabled(
            True
        )
        self.open_folder_button.setEnabled(
            True
        )

        QMessageBox.information(
            self,
            "处理完成",
            "服务器气道与结节分割完成，"
            "病例包已经下载到本地。\n\n"
            f"{self.last_stage4_dir}",
        )

        if self.auto_open_checkbox.isChecked():
            self.open_navigation()

    def pipeline_failed(
        self,
        message: str,
    ) -> None:
        self.start_button.setEnabled(
            True
        )
        self.progress_bar.setRange(
            0,
            100,
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(
            "失败"
        )
        self.stage_label.setText(
            "自动处理失败"
        )
        self.append_log("")
        self.append_log(
            "自动处理失败："
        )
        self.append_log(
            message
        )

        QMessageBox.critical(
            self,
            "自动处理失败",
            message,
        )

    def open_navigation(self) -> None:
        if (
            self.last_stage4_dir is None
            or not self.last_stage4_dir.is_dir()
        ):
            try:
                case_id = sanitize_case_id(
                    self.case_id_edit.text()
                )
                candidate = (
                    Path(
                        self.local_root_edit
                        .text()
                        .strip()
                    )
                    / case_id
                    / "stage4_package"
                )

                if candidate.is_dir():
                    self.last_stage4_dir = (
                        candidate
                    )
                else:
                    raise FileNotFoundError(
                        "没有找到本地 stage4_package"
                    )
            except Exception as error:
                QMessageBox.warning(
                    self,
                    "无法打开导航",
                    str(error),
                )
                return

        command = [
            sys.executable,
            str(
                self.ui_script_path
            ),
            "--case-dir",
            str(
                self.last_stage4_dir
            ),
        ]

        try:
            creation_flags = 0

            if os.name == "nt":
                creation_flags = int(
                    getattr(
                        subprocess,
                        "CREATE_NEW_PROCESS_GROUP",
                        0,
                    )
                )

            subprocess.Popen(
                command,
                cwd=str(
                    self.project_dir
                ),
                creationflags=(
                    creation_flags
                ),
            )

            self.append_log(
                "已打开交互导航界面："
                f"{self.last_stage4_dir}"
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "导航界面启动失败",
                f"{type(error).__name__}: "
                f"{error}",
            )

    def open_result_folder(self) -> None:
        if self.last_stage4_dir is None:
            return

        try:
            if os.name == "nt":
                os.startfile(  # type: ignore[attr-defined]
                    str(
                        self.last_stage4_dir
                    )
                )
            elif sys.platform == "darwin":
                subprocess.Popen(
                    [
                        "open",
                        str(
                            self.last_stage4_dir
                        ),
                    ]
                )
            else:
                subprocess.Popen(
                    [
                        "xdg-open",
                        str(
                            self.last_stage4_dir
                        ),
                    ]
                )
        except Exception as error:
            QMessageBox.warning(
                self,
                "无法打开目录",
                str(error),
            )

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if (
            self.worker is not None
            and self.worker.isRunning()
        ):
            answer = QMessageBox.question(
                self,
                "任务仍在运行",
                "服务器任务仍在运行，确定关闭客户端吗？\n"
                "关闭窗口不会自动终止服务器上的推理进程。",
            )

            if (
                answer
                != QMessageBox.StandardButton.Yes
            ):
                event.ignore()
                return

        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(
        "AirwayNavigationSystemV3"
    )

    if paramiko is None:
        QMessageBox.warning(
            None,
            "缺少 Paramiko",
            "当前 dicom 环境中没有安装 paramiko，"
            "服务器连接功能将无法使用。",
        )

    window = (
        NavigationLauncherWindow()
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

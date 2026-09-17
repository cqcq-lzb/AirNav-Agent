#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib
import sys


PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-image", "skimage"),
    ("SimpleITK", "SimpleITK"),
    ("VTK", "vtk"),
    ("PySide6", "PySide6"),
    ("paramiko", "paramiko"),
]


def module_version(module) -> str:
    for name in [
        "__version__",
        "Version_VersionString",
    ]:
        value = getattr(
            module,
            name,
            None,
        )

        if callable(value):
            try:
                return str(value())
            except Exception:
                continue

        if value is not None:
            return str(value)

    if hasattr(module, "vtkVersion"):
        try:
            return str(
                module.vtkVersion.GetVTKVersion()
            )
        except Exception:
            pass

    return "已安装"


def main() -> int:
    print("=" * 72)
    print("经支气管肺结节自动导航系统 v3 环境检查")
    print("=" * 72)
    print("Python：", sys.executable)
    print("版本：", sys.version)
    print()

    missing = []

    for display_name, import_name in PACKAGES:
        try:
            module = importlib.import_module(
                import_name
            )
            print(
                f"[通过] {display_name}: "
                f"{module_version(module)}"
            )
        except Exception as error:
            missing.append(
                display_name
            )
            print(
                f"[缺少] {display_name}: "
                f"{type(error).__name__}: {error}"
            )

    print()

    if missing:
        print(
            "缺少依赖：",
            ", ".join(missing),
        )
        return 1

    print("导航系统环境检查全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

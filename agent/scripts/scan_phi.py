"""PHI 守卫：在数据边界上证明「本系统不持有患者可识别信息」。

    python -m agent.scripts.scan_phi                    # 自证 + 扫真实数据
    python -m agent.scripts.scan_phi --root cases       # 换数据根
    python -m agent.scripts.scan_phi --json out.json    # 落一份机器可读的报告
    python -m agent.scripts.scan_phi --scan-only        # 跳过自证（快）
    python -m agent.scripts.scan_phi --apply-out D:\\deid   # 命中时导出脱敏副本

退出码：`0` 未发现 PHI；`1` 发现命中（或自证失败）。

## 为什么是「守卫」而不是「去标识工具」

差距清单里原本写的是「上传前自动清除 DICOM 标签」。**侦察后发现这条无处置放**
（2026-09-20）：

- `cases/` 里**一个 DICOM 文件都没有**（`find cases -iname '*.dcm'` → 0 个）；
- `backend/api_server.py` 的上传口**只接受 `.nii` / `.nii.gz`**，DICOM 根本没进过这个系统；
- 于是「清 DICOM 标签」这个动作没有任何调用方 —— 写出来也没人能用。

所以改成一件真正成立的事：**把「我们不持有 PHI」从一句口头承诺，变成一条能失败、
能进回归门禁的断言**。这才是治理该有的形态（同一个项目里「依赖延后」也是这个路子：
不给承诺，给保证）。

## 扫哪里：只扫**能承载 PHI 的位置**

数据面清点（2026-09-20）：78 个 `.nii.gz` / 50 个 `.json` / 14 个 `.csv` / 12 个 `.zip`。

| 载体 | 能承载 PHI 的位置 |
|---|---|
| NIfTI（`.nii` / `.nii.gz`） | 头部 4 个**自由文本槽**：`descrip[80]` / `aux_file[24]` / `intent_name[16]` / `db_name[18]` |
| 文本（`.json` / `.csv` / `.txt`） | 键名（DICOM 标准 PHI 词汇）与值（高置信度模式） |
| DICOM（`.dcm`） | 标准 PHI 标签（本系统当前没有，但守卫要能接住将来出现的） |

NIfTI 走**字节级解析**而不是 SimpleITK：头部布局是固定的，直接按偏移读文本槽，
既快（不加载影像数据）又能顺带支持**清除**（SimpleITK 不提供写回 arbitrary text slot 的接口）。

## 🔴 这个脚本自己的第一版全是误报，教训固化在 fixture 里

第一版用一个宽松正则扫「元数据键或值」，结果 **6 处命中全是误报**：

| 误报 | 根因 |
|---|---|
| `"stage": "airway_qc_blocked"` | 模式里的 `age` 命中了 `st**age**` |
| `"message": "气道质控未通过…"` | 同上（`mess**age**`） |
| `sform_code_name = NIFTI_XFORM_SCANNER_ANAT` | 模式里的 `study`/`scanner` 命中了设备术语 |

这与项目里「打分器 `BOUNDARY_MARKERS` 做精确子串匹配 → 中文插个修饰词就匹配不上」
是同一条教训的镜像：**尺子太宽和太窄一样是坏尺子，只是坏的方向相反**。
所以三条纪律写进实现：

1. **字段名匹配必须带词边界**，且**不用短词**做子串（`age` / `sex` / `id` 这类禁用，
   改用 `\bpatient_?age\b` 这种带限定前缀的完整词）；
2. **值级模式只在文本槽里跑**，绝不在整个 JSON 正文上跑 ——
   否则 `generated_at` 的时间戳会被当成出生日期；
3. **豁免必须留痕**：命中但判定为非 PHI 时打印原因，不许静默放过
   （「静默豁免」等于放水，见评测那一节的同类教训）。

`--selftest` 把上面两类误报做成**必须不命中**的 fixture，把植入的 PHI 做成**必须命中**
的 fixture。**一个永远说「干净」的扫描器等于没有** —— 它平时永远返回 OK，
只有真出事时才该说话，所以它必须自证会叫。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]

# DICOM 标准里的患者 / 机构可识别字段。
#
# ⚠️ 两个细节都不是随手写的：
# ① **允许分隔符**：JSON 里写的是 `patient_id`、CSV 表头可能是 `Patient Name`，
#    而 DICOM 关键字是 `PatientID`。写成 `patient[_\s-]?id` 三种都能认。
#    （第一版只写死 `patientid`，结果 fixture `{"patient_id": ...}` 是靠**值**里的
#     `MRN` 蒙对的 —— 键名其实没认出来，属于「碰巧过了」。）
# ② **不用短词**：`age` / `sex` / `id` 单独出现一律不做匹配，它们会命中
#    `stage` / `message` 这类完全无关的词。
PHI_PATTERNS = (
    r"patient[_\s-]?(?:name|id|birthdate|birthtime|sex|age|size|weight|address"
    r"|comments|telephonenumbers|motherbirthname)",
    r"other[_\s-]?patient[_\s-]?(?:ids|names)",
    r"mrn",
    r"medical[_\s-]?record[_\s-]?number",
    r"referring[_\s-]?physician[_\s-]?name",
    r"performing[_\s-]?physician[_\s-]?name",
    r"operators[_\s-]?name",
    r"physicians[_\s-]?of[_\s-]?record",
    r"requesting[_\s-]?physician",
    r"institution[_\s-]?name",
    r"institution[_\s-]?address",
    r"institutional[_\s-]?department[_\s-]?name",
    r"station[_\s-]?name",
    r"accession[_\s-]?number",
    r"study[_\s-]?id",
    r"admission[_\s-]?id",
)
PHI_KEY_RE = re.compile(r"\b(?:" + "|".join(PHI_PATTERNS) + r")\b", re.I)

# 非身份信息的「patient 前缀」字段：体位/朝向是**采集几何**，不是身份。
# 它们必须被豁免 —— 但**豁免必须留痕**（打印出来），不许静默放过。
# 「静默豁免」等于放水，和评测里那条 `must_exclude` 豁免要留痕是同一个道理。
PHI_EXEMPT = {
    "patient_position": "采集体位（如 HFS），属几何信息",
    "patient_orientation": "图像朝向，属几何信息",
    "patient_state": "扫描时患者状态，非身份标识",
}
_PATIENT_FAMILY_RE = re.compile(r"\bpatient[_\s-]?\w+\b", re.I)

# 值级高置信度模式。**只在能承载自由文本的位置里使用**（见模块文档纪律 2）。
VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("中文姓名（跟在「姓名/患者/病人」后）", re.compile(r"(?:患者|病人|姓名)[:：\s]*[\u4e00-\u9fa5]{2,4}")),
    ("身份证号", re.compile(r"\b\d{17}[\dXx]\b")),
    ("手机号", re.compile(r"\b1[3-9]\d{9}\b")),
    ("邮箱", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("出生日期（带标签）", re.compile(r"(?:birth|birthdate|dob)[^\n]{0,4}[=:：]\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}", re.I)),
    ("显式键值对", re.compile(r"\bpatient_?name\b\s*[=:：]\s*\S+", re.I)),
)

# NIfTI-1 头部里 4 个自由文本槽（偏移, 长度）。NIfTI-2 偏移不同，见 _text_slots。
NIFTI1_SLOTS = (("db_name", 14, 18), ("descrip", 148, 80), ("aux_file", 228, 24),
                ("intent_name", 328, 16))
NIFTI2_SLOTS = (("db_name", 18, 18), ("descrip", 240, 80), ("aux_file", 336, 24),
                ("intent_name", 376, 16))

TEXT_SUFFIXES = (".json", ".csv", ".txt", ".md", ".tsv", ".yaml", ".yml")
SCAN_SUFFIXES = (".nii.gz", ".nii") + TEXT_SUFFIXES + (".dcm",)


@dataclass
class Hit:
    """一处命中。级别分开是为了让退出码有区分度（见 --strict）。"""

    path: str
    where: str          # 位置（NIfTI 槽名 / JSON 键 / 行号）
    detail: str         # 命中的内容（截断）
    level: str = "confirmed"   # confirmed | suspicious
    note: str = ""             # 豁免原因 / 说明


@dataclass
class Report:
    scanned: dict[str, int] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, path: str, where: str, detail: str, level: str = "confirmed",
            note: str = "") -> None:
        self.hits.append(Hit(str(path), where, detail[:120], level, note))


# ------------------------------------------------------------------ NIfTI


def _nifti_slots(path: Path) -> tuple[int, tuple[tuple[str, int, int], ...]] | None:
    """返回 (header 起始偏移, 文本槽表)。非 NIfTI 返回 None。

    ⚠️ 判断顺序别写成一长串 and/or —— 第一版写成
    `A or (B & 0xFF) == 0 and C`，运算符优先级把它变成 `A or ((B&0xFF)==0 and C)`，
    看着「也能跑」，实际两条分支的判断依据完全不同。分开写清楚。
    """
    raw = path.read_bytes()
    if path.name.lower().endswith(".gz"):
        import gzip

        try:
            with gzip.open(path, "rb") as handle:
                raw = handle.read(600)
        except OSError:
            return None
    if len(raw) < 348:
        return None

    sizeof_hdr = struct.unpack_from("<i", raw, 0)[0]
    magic = raw[344:348]
    if sizeof_hdr == 348 or magic in (b"n+1\x00", b"ni1\x00"):
        return 0, NIFTI1_SLOTS
    if sizeof_hdr == 540:
        return 0, NIFTI2_SLOTS
    if struct.unpack_from(">i", raw, 0)[0] == 540:
        return 0, NIFTI2_SLOTS
    return None


def read_nifti_text_slots(path: Path) -> dict[str, str] | None:
    """读出 4 个自由文本槽（只解压前若干字节，不加载影像数据）。"""
    raw = path.read_bytes()
    if path.name.lower().endswith(".gz"):
        import gzip

        try:
            with gzip.open(path, "rb") as handle:
                raw = handle.read(600)
        except OSError:
            return None
    detected = _nifti_slots(path)
    if detected is None:
        return None
    _, slots = detected
    out: dict[str, str] = {}
    for name, offset, length in slots:
        chunk = raw[offset:offset + length]
        out[name] = chunk.split(b"\x00", 1)[0].decode("latin-1").strip()
    return out


def clear_nifti_text_slots(path: Path, slots: dict[str, str]) -> None:
    """把指定文本槽在**文件里**清零（NUL 填充）。

    只在调用方已经拷贝出来的副本上使用 —— 这个函数不认识「原文件」这个概念。
    """
    gzipped = path.name.lower().endswith(".gz")
    if gzipped:
        import gzip

        with gzip.open(path, "rb") as handle:
            data = bytearray(handle.read())
    else:
        data = bytearray(path.read_bytes())
    detected = _nifti_slots(path)
    if detected is None:
        return
    _, table = detected
    for name, offset, length in table:
        if name in slots:
            data[offset:offset + length] = b"\x00" * length
    if gzipped:
        # mtime=0：让产物可复现（否则同样的内容每次压缩出不同字节）
        with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as handle:
            handle.write(bytes(data))
    else:
        path.write_bytes(bytes(data))


# ------------------------------------------------------------------ 扫描


def _log_exemptions(text: str, path: Path, report: Report) -> None:
    """把「形状命中但被豁免」的字段**打印出来**。

    豁免必须留痕 —— 静默豁免等于放水，你无从分辨「真没有」和「被悄悄放过了」。
    """
    for match in _PATIENT_FAMILY_RE.finditer(text):
        key = re.sub(r"[\s-]+", "_", match.group(0)).lower()
        if key in PHI_EXEMPT:
            report.notes.append(f"{path}: 豁免 {match.group(0)}（{PHI_EXEMPT[key]}）")


def scan_nifti(path: Path, report: Report) -> None:
    slots = read_nifti_text_slots(path)
    if slots is None:
        report.notes.append(f"{path}: 不是 NIfTI（跳过文本槽检查）")
        return
    report.scanned["nifti"] = report.scanned.get("nifti", 0) + 1
    for name, value in slots.items():
        if not value:
            continue
        if PHI_KEY_RE.search(value):
            report.add(path, f"NIfTI {name}", value)
            continue
        for label, pattern in VALUE_PATTERNS:
            match = pattern.search(value)
            if match:
                report.add(path, f"NIfTI {name}", f"{label}: {match.group(0)}")
                break
        _log_exemptions(value, path, report)


def scan_text(path: Path, report: Report) -> None:
    """扫文本文件。

    值级模式跑在**有界的上下文**里，而不是整段正文：
    - `.json` → 引号内的字符串值（JSON 的结构已经给了边界）
    - `.csv` / `.txt` / `.md` / …→ **逐行**（CSV 天然没有引号，只扫引号会整类漏掉 ——
      第一版就是这么漏了「联系电话,13812345678」这一行）

    逐行扫日期类模式是安全的，因为 `出生日期` 那一条**要求带 birth/dob 标签前缀**，
    不是见 `2026-09-20` 就报（否则 manifest 里的时间戳会成片误报）。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        report.notes.append(f"{path}: 读取失败 {type(error).__name__}")
        return
    report.scanned["text"] = report.scanned.get("text", 0) + 1

    # 纪律 1：键名用带边界的完整词匹配，绝不用 age/sex/id 这类短词做子串
    for match in PHI_KEY_RE.finditer(text):
        line = text[:match.start()].count("\n") + 1
        report.add(path, f"第 {line} 行键名", match.group(0))
    _log_exemptions(text, path, report)

    # 纪律 2：值级模式只跑在有界上下文里
    if path.name.lower().endswith(".json"):
        contexts = ((f"值 #{index}", m.group(1)) for index, m in
                    enumerate(re.finditer(r'"([^"\n]{1,200})"', text), 1))
    else:
        contexts = ((f"第 {number} 行", line) for number, line in enumerate(text.splitlines(), 1))

    for where, value in contexts:
        for label, pattern in VALUE_PATTERNS:
            found = pattern.search(value)
            if found:
                report.add(path, where, f"{label}: {found.group(0)}")
                break


def scan_zip(path: Path, report: Report) -> None:
    """zip 只查成员名（成员内容按需解包后单独扫）。"""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, OSError) as error:
        report.notes.append(f"{path}: zip 打不开 {type(error).__name__}")
        return
    report.scanned["zip"] = report.scanned.get("zip", 0) + 1
    for name in names:
        if PHI_KEY_RE.search(name) or VALUE_PATTERNS[1][1].search(name):
            report.add(path, "zip 成员名", name)


def scan_dicom(path: Path, report: Report) -> None:
    """本系统当前没有 DICOM，但将来出现时要能接住 —— 所以不留空实现。"""
    try:
        import pydicom
    except ImportError:
        report.notes.append("pydicom 不可用，跳过 DICOM 文件（本系统当前不产生 DICOM）")
        return
    report.scanned["dicom"] = report.scanned.get("dicom", 0) + 1
    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception as error:  # noqa: BLE001
        report.notes.append(f"{path}: DICOM 解析失败 {type(error).__name__}")
        return
    for element in dataset:
        keyword = (element.keyword or "").lower()
        if keyword and PHI_KEY_RE.search(keyword):
            report.add(path, f"DICOM {element.tag} {element.keyword}", str(element.value))


SCANNERS = (
    ((".nii.gz", ".nii"), scan_nifti),
    (TEXT_SUFFIXES, scan_text),
    ((".zip",), scan_zip),
    ((".dcm",), scan_dicom),
)


def scan_tree(root: Path, report: Report) -> None:
    if not root.is_dir():
        report.notes.append(f"{root} 不存在或不是目录（跳过）")
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        for suffixes, scanner in SCANNERS:
            if lowered.endswith(suffixes):
                try:
                    scanner(path, report)
                except Exception as error:  # noqa: BLE001 —— 扫描器不许因单个文件挂掉
                    report.notes.append(f"{path}: 扫描异常 {type(error).__name__}: {error}")
                break
        else:
            report.scanned["skipped"] = report.scanned.get("skipped", 0) + 1


# ------------------------------------------------------------------ 自证


def _write_nifti_with_phi(path: Path, descrip: str) -> None:
    """造一个最小 NIfTI-1，并在 descrip 里写上指定文本（用于自证）。"""
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)          # sizeof_hdr
    struct.pack_into("<h", header, 40, 3)           # dim[0]
    for index in (1, 2, 3):                          # dim[1..3]
        struct.pack_into("<h", header, 40 + index * 2, 4)
    struct.pack_into("<h", header, 70, 16)          # datatype = float32
    struct.pack_into("<h", header, 72, 32)          # bitpix
    struct.pack_into("<f", header, 76, 1.0)         # pixdim[1]
    struct.pack_into("<f", header, 80, 1.0)
    struct.pack_into("<f", header, 84, 1.0)
    struct.pack_into("<f", header, 108, 352.0)      # vox_offset
    struct.pack_into("<f", header, 112, 1.0)        # scl_slope
    header[148:148 + len(descrip)] = descrip.encode("latin-1")[:80]
    header[344:348] = b"n+1\x00"
    voxels = struct.pack("<" + "f" * 64, *([0.0] * 64))
    import gzip

    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as handle:
        handle.write(bytes(header) + voxels)


def selftest() -> int:
    """自证：植入的 PHI 必须命中，已知的误报形态必须不命中。"""
    print("\n[自证] 扫描器必须「该叫的时候叫，不该叫的时候闭嘴」")
    failures = 0
    total = [0]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        def run(label: str, path: Path, expect_hit: bool) -> None:
            nonlocal failures
            total[0] += 1
            report = Report()
            lowered = path.name.lower()
            for suffixes, scanner in SCANNERS:
                if lowered.endswith(suffixes):
                    scanner(path, report)
                    break
            got = [h for h in report.hits if h.level == "confirmed"]
            ok = bool(got) == expect_hit
            detail = ("命中 " + got[0].detail) if got else "无命中"
            print(f"  {'PASS' if ok else 'FAIL'}  {label}（{detail}）")
            if not ok:
                failures += 1

        # ---- 必须命中 ----
        dirty = base / "dirty.nii.gz"
        _write_nifti_with_phi(dirty, "dcm2niix PatientName=ZHANG^SAN;Age=061Y")
        run("NIfTI descrip 里有 PatientName → 命中", dirty, True)

        name_json = base / "manifest.json"
        name_json.write_text(json.dumps({"patient_id": "MRN-00012345"}), encoding="utf-8")
        run("manifest 里有 patient_id → 命中", name_json, True)

        # 键名路径要独立验一次：上面那条有可能又是被「值里的 MRN」蒙对的
        total[0] += 1
        key_only = base / "keyonly.json"
        key_only.write_text(json.dumps({"patient_birthdate": "1970-01-01",
                                        "institution_name": "某医院"}), encoding="utf-8")
        key_report = Report()
        scan_text(key_only, key_report)
        by_key = [h for h in key_report.hits if "键名" in h.where]
        ok = len(by_key) >= 2
        print(f"  {'PASS' if ok else 'FAIL'}  下划线键名本身被认出（不靠值蒙对）"
              f"（{len(by_key)} 个键名命中：{[h.detail for h in by_key]}）")
        if not ok:
            failures += 1

        # 豁免必须留痕：patient_position 是采集体位，不是身份
        total[0] += 1
        geom = base / "geom.json"
        geom.write_text(json.dumps({"patient_position": "HFS"}), encoding="utf-8")
        geom_report = Report()
        scan_text(geom, geom_report)
        ok = not geom_report.hits and any("豁免" in n for n in geom_report.notes)
        print(f"  {'PASS' if ok else 'FAIL'}  patient_position 不报但**留痕**"
              f"（命中 {len(geom_report.hits)} 处，备注 {len(geom_report.notes)} 条）")
        if not ok:
            failures += 1

        phone_csv = base / "route.csv"
        phone_csv.write_text("x,y\n0,0\n联系电话,13812345678\n", encoding="utf-8")
        run("CSV 里有手机号 → 命中", phone_csv, True)

        clean_but_named = base / "case.json"
        clean_but_named.write_text('{"name": "LIDC_0089"}', encoding="utf-8")
        run("普通 name 字段（病例编号）→ 不误判为姓名", clean_but_named, False)

        # ---- 必须不命中（这两条正是本脚本第一版的误报现场）----
        fp1 = base / "qc.json"
        fp1.write_text(json.dumps({"stage": "airway_qc_blocked",
                                   "message": "气道质控未通过"}), encoding="utf-8")
        run("误报 ①：“stage”里含 age、“message”里含 age → 必须不命中", fp1, False)

        fp2 = base / "clean.nii.gz"
        _write_nifti_with_phi(fp2, "")
        run("误报 ②：干净的 NIfTI（descrip 为空）→ 必须不命中", fp2, False)

        # ---- 清除路径必须在合成数据上跑通闭环 ----
        total[0] += 1
        target = base / "tofix.nii.gz"
        _write_nifti_with_phi(target, "PatientName=LI^SI")
        before = read_nifti_text_slots(target) or {}
        clear_nifti_text_slots(target, {"descrip": before.get("descrip", "")})
        after = read_nifti_text_slots(target) or {}
        report = Report()
        scan_nifti(target, report)
        ok = before.get("descrip") and not after.get("descrip") and not report.hits
        print(f"  {'PASS' if ok else 'FAIL'}  清除文本槽 → 再扫不命中"
              f"（清除前 {before.get('descrip')!r} → 清除后 {after.get('descrip')!r}）")
        if not ok:
            failures += 1

    print(f"\n自证{'通过' if not failures else '失败'}：{total[0] - failures}/{total[0]} 项符合预期")
    return failures


# ------------------------------------------------------------------ 入口


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PHI 守卫：扫描数据面里的患者可识别信息")
    parser.add_argument("--root", action="append", default=None,
                        help="要扫的数据根（可多次；默认 cases + api_data）")
    parser.add_argument("--json", dest="json_path", help="把报告写成 JSON")
    parser.add_argument("--scan-only", action="store_true", help="跳过自证")
    parser.add_argument("--selftest-only", action="store_true", help="只跑自证")
    parser.add_argument("--apply-out", dest="apply_out",
                        help="命中时导出脱敏副本到该目录（只清 NIfTI 文本槽）")
    parser.add_argument("--audit", action="store_true", help="把这次扫描写进审计日志")
    args = parser.parse_args(argv)

    print("=" * 68)
    print("PHI 守卫：证明本系统不持有患者可识别信息")
    print("=" * 68)

    failures = 0
    if not args.scan_only:
        failures += selftest()
    if args.selftest_only:
        return 1 if failures else 0

    roots = [Path(p) for p in args.root] if args.root else [ROOT / "cases", ROOT / "api_data"]
    report = Report()
    print("\n[扫描] 数据面")
    for root in roots:
        before = len(report.hits)
        scan_tree(root, report)
        print(f"  {root} → 命中 {len(report.hits) - before} 处")

    counts = " · ".join(f"{k} {v}" for k, v in sorted(report.scanned.items()))
    print(f"\n  扫过：{counts or '（没有可扫的文件）'}")

    confirmed = [h for h in report.hits if h.level == "confirmed"]
    suspicious = [h for h in report.hits if h.level != "confirmed"]

    if confirmed:
        print(f"\n🔴 发现 {len(confirmed)} 处疑似 PHI：")
        for hit in confirmed[:40]:
            print(f"    {hit.path}  [{hit.where}]  {hit.detail}")
        if len(confirmed) > 40:
            print(f"    …另有 {len(confirmed) - 40} 处，见 --json 报告")
    if suspicious:
        print(f"\n⚠️  {len(suspicious)} 处需人工确认：")
        for hit in suspicious[:20]:
            print(f"    {hit.path}  [{hit.where}]  {hit.detail}")
    if report.notes:
        print(f"\n说明（{len(report.notes)} 条）：")
        for note in report.notes[:10]:
            print(f"    - {note}")

    if args.apply_out and confirmed:
        out = Path(args.apply_out)
        out.mkdir(parents=True, exist_ok=True)
        cleaned = 0
        for hit in confirmed:
            source = Path(hit.path)
            if not source.name.lower().endswith((".nii", ".nii.gz")) or not source.is_file():
                continue
            slots = read_nifti_text_slots(source) or {}
            if not any(slots.values()):
                continue
            target = out / source.name
            shutil.copy2(source, target)
            clear_nifti_text_slots(target, slots)
            cleaned += 1
        print(f"\n已导出 {cleaned} 份脱敏副本到 {out}（只清 NIfTI 文本槽；"
              f"JSON/CSV 里的命中需人工处理，不做自动改写）")

    if args.json_path:
        payload = {
            "roots": [str(r) for r in roots],
            "scanned": report.scanned,
            "confirmed": [h.__dict__ for h in confirmed],
            "suspicious": [h.__dict__ for h in suspicious],
            "notes": report.notes,
            "selftest_failures": failures,
        }
        Path(args.json_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\n报告已落盘：{args.json_path}")

    if args.audit:
        try:
            from agent import audit

            record = audit.record_phi_scan(
                roots=[str(r) for r in roots],
                scanned=report.scanned,
                confirmed=len(confirmed),
                suspicious=len(suspicious),
                hits=[{"path": h.path, "where": h.where} for h in confirmed[:50]],
                selftest_failures=failures,
            )
            ok, problems = audit.verify_chain()
            print(f"\n已写审计（链{'完整' if ok else '异常：' + str(problems[:1])}）："
                  f"run_id={str((record or {}).get('hash'))[:12]}…")
        except Exception as error:  # noqa: BLE001 —— 审计不许弄坏主流程
            print(f"\n[audit] 写入失败（已忽略）：{type(error).__name__}: {error}")

    print("\n" + "=" * 68)
    if failures or confirmed:
        print(f"结论：✗ 未通过（自证失败 {failures} 项，PHI 命中 {len(confirmed)} 处）")
        return 1
    print("结论：✓ 通过 —— 自证会叫，且扫过的数据里没有患者可识别信息")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

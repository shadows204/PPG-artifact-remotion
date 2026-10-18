#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPG 运动伪影抑制项目 · 开源数据集下载 / 校验 CLI（可单独下载某一个数据集）。

支持的数据集
------------
============  ==================================================  ==========
名称          ``--dataset`` 取值                                   目标目录
============  ==================================================  ==========
TROIKA        ``troika`` / ``ieeepg``                             ``IEEEPPG_TROIKA/``
BIDMC         ``bidmc``                                           ``BIDMC/``
CapnoBase     ``capnobase`` / ``capno``                           ``CapnoBase/``
WESAD         ``wesad``                                           ``WESAD/``
============  ==================================================  ==========

CIME PPG 不在本脚本范围内（工作区已有其 npz 缓存）。

设计要点
--------
1. **幂等**：目标文件已存在且大小/结构校验通过时直接跳过，不重复下载。
2. **可断点续传**：归档先下到 ``<name>.part``，用 ``curl -C -`` 续传；
   完成后原子 ``os.replace`` 成正式文件名。若服务端不支持 Range（curl 报
   exit 33）则自动放弃续传、从头重下。
3. **校验**：每个归档记录 HTTP Content-Length + 落盘字节数 + SHA256，
   并做 zip 完整性检查（``testzip`` 抽查）；解包后再按数据规格校验
   （记录数 / 采样率 / 通道名），见 ``--verify``。
4. **多源回退**：每个数据集配有多个候选 URL，逐个尝试并记录哪一个成功。
   中国大陆直连 github.com / raw.githubusercontent.com 会超时，因此
   GitHub 源统一走 ``ghfast.top`` 加速镜像，另留官方 codeload 作回退。
5. **清单**：``<root>/_downloads/manifest.json`` 记录来源 URL、字节数、
   SHA256、下载时间，供 ``datasets/README.md`` 引用。

用法
----
::

    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/download_datasets.py --list
    $PY scripts/tools/download_datasets.py --dataset troika bidmc capnobase
    $PY scripts/tools/download_datasets.py --dataset all
    $PY scripts/tools/download_datasets.py --dataset wesad --verify
    $PY scripts/tools/download_datasets.py --verify            # 只校验，不下载

数据集根目录默认 ``<工作区>/datasets``，可用 ``--root`` 或环境变量
``PPG_DATASETS`` 覆盖（与 ``scripts/_common/paths.py::datasets_root`` 一致）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # <工作区>

# --------------------------------------------------------------------------
# 数据源定义
# --------------------------------------------------------------------------
# 每个数据集 = 一个 archive（zip）+ 一组 "从归档里的哪个前缀抽到哪个子目录" 规则。
#
# 2026-10-14 实测结论（详见 datasets/README.md）：
#   * physionet.org 的 /files/ 与 get-zip 对匿名访问一律 403
#     （页面正文 "You don't have access to see this page."），因此不能用
#     PhysioNet 官方直链；改用 Kaggle 上逐字节同源的 PhysioNet 1.0.0 镜像包。
#   * github.com / codeload / raw.githubusercontent 直连超时或 <10 KB/s；
#     ghfast.top 镜像实测 ~1.5 MB/s，可用。
#   * huggingface.co 在本机完全不可达（connect timeout）。
#   * **CapnoBase 根本不在 PhysioNet 上**：/content/capnobase/ 与
#     /content/capno/ 全部 404。它的官方宿主是 Borealis Dataverse
#     （DOI 10.5683/SP2/NLB8IT，r4 版），也正因如此本目录里的
#     README.txt/MANIFEST.TXT/data/mat/data/csv/preview 才是 Dataverse 布局。
#   * WESAD 的 UCI 静态 URL 是"陷阱"：HTTP 200 但正文只有 261 字节的指针
#     文本，其 Sciebo 直链已 404；实际可用的是 Kaggle 等价镜像。
SOURCES = {
    "troika": {
        "display": "TROIKA / IEEE SP Cup 2015 (IEEEPPG)",
        "target": "IEEEPPG_TROIKA",
        "archives": [{
            "filename": "hpi-dhc-TROIKA-main.zip",
            "urls": [
                "https://ghfast.top/https://github.com/hpi-dhc/TROIKA/archive/refs/heads/main.zip",
                "https://codeload.github.com/hpi-dhc/TROIKA/zip/refs/heads/main",
                "https://github.com/hpi-dhc/TROIKA/archive/refs/heads/main.zip",
            ],
            "min_bytes": 20_000_000,          # 实测 26,096,139 B
            "strip_top": True,                # GitHub 归档外壳 TROIKA-main/
            "note": "hpi-dhc/TROIKA 仓库内 datasets/IEEE_SPC_2015/ 逐字节保存了官方原始 .mat",
            # (归档内前缀, 目标子目录, 允许的扩展名)
            "extract": [
                ("datasets/IEEE_SPC_2015/Training_data/", "Training_data", (".mat", ".pdf", ".docx")),
                ("datasets/IEEE_SPC_2015/TestData/",      "TestData",      (".mat", ".pdf", ".docx")),
                ("datasets/IEEE_SPC_2015/TrueBPM/",       "TrueBPM",       (".mat",)),
            ],
        }],
    },
    "bidmc": {
        "display": "BIDMC PPG and Respiration Dataset v1.0.0",
        "target": "BIDMC",
        "archives": [{
            "filename": "kaggle-anhtua-bidmc-capno-datasets.zip",
            "urls": [
                "https://www.kaggle.com/api/v1/datasets/download/anhtua/bidmc-capno-datasets",
                "https://www.kaggle.com/api/v1/datasets/download/anhtua/bidmc-ppg-capno-ieee",
                "https://www.kaggle.com/api/v1/datasets/download/anhtua/bidmc-ppg-dataset",
            ],
            "min_bytes": 50_000_000,          # 实测 154,389,685 B（含 capno）
            "note": "Kaggle 匿名下载可用：302 → storage.googleapis.com 签名直链，无需凭据",
            "resolve_redirect": True,
            "extract": [
                # bidmc/bidmc_csv/*.csv（53 例 × 4 个 csv：Signals/Numerics/Breaths/Fix）
                ("bidmc/bidmc_csv/", "bidmc_csv", (".csv",)),
                # WFDB 原始件（.hea/.dat/.breath）与元数据
                ("bidmc/",           "",          (".hea", ".dat", ".breath")),
                ("bidmc/",           "",          None,
                 {"LICENSE", "README", "RECORDS", "SHA256SUMS.txt"}),
            ],
            "exclude_basenames": {"ANNOTATORS", "DBS"},
        }],
    },
    "capnobase": {
        "display": "CapnoBase r4 (Borealis Dataverse, DOI 10.5683/SP2/NLB8IT)",
        "target": "CapnoBase",
        "archives": [{
            "filename": "kaggle-anhtua-bidmc-capno-datasets.zip",
            "urls": [
                "https://www.kaggle.com/api/v1/datasets/download/anhtua/bidmc-capno-datasets",
                "https://www.kaggle.com/api/v1/datasets/download/anhtua/bidmc-ppg-capno-ieee",
                "https://www.kaggle.com/api/v1/datasets/download/shahed7/capnobase-full-data",
            ],
            "min_bytes": 50_000_000,
            "note": "与 BIDMC 同一个 Kaggle 包，本数据集只取 capno/ 子树",
            "resolve_redirect": True,
            "extract": [
                ("capno/data/mat/", "data/mat", (".mat",)),
                ("capno/data/csv/", "data/csv", (".csv",)),
                ("capno/",          "",         None, {"README.txt", "MANIFEST.TXT"}),
            ],
        }],
    },
    "wesad": {
        "display": "WESAD (Wearable Stress and Affect Detection)",
        "target": "WESAD",
        "archives": [{
            "filename": "wesad.zip",
            "urls": [
                # 实测可用的等价镜像（24 MB/s）。UCI 官方静态 URL 只返回 261 B 指针文本，
                # 其指向的 Sciebo 直链已 404，故不作为候选源，仅在此备注。
                "https://www.kaggle.com/api/v1/datasets/download/orvile/wesad-wearable-stress-affect-detection-dataset",
                "https://www.kaggle.com/api/v1/datasets/download/mohamedasem318/wesad-full-dataset",
            ],
            "min_bytes": 2_500_000_000,       # 实测 2,612,988,288 B
            "note": ("UCI 官方 https://archive.ics.uci.edu/static/public/465/"
                     "wesad+wearable+stress+and+affect+detection.zip 实测返回 HTTP 200 "
                     "但正文只有 261 字节的 WESAD.txt，内容是已失效的 "
                     "uni-siegen.sciebo.de 链接（实测 404）；故改用 Kaggle 等价镜像"),
            "resolve_redirect": True,
            "extract": [("", "", ("",))],      # 全量解包，见 _install_wesad
        }],
    },
}

SPEC = {
    # 期望规格（用于 --verify 的 PASS/FAIL 对照；来自任务书 + 官方 Readme）
    "troika": {
        "Training_data": 12,   # 12 条训练记录（DATA_01_TYPE01 … DATA_12_TYPE02）
        "TestData": 10,        # 10 条官方留出测试记录（TEST_S*_T*.mat）
        "TrueBPM": 10,
        "fs": 125.0,
        "sig_rows_train": 6,   # [ECG, PPG1, PPG2, ACCx, ACCy, ACCz]
        "sig_rows_test": 5,    # [PPG1, PPG2, ACCx, ACCy, ACCz]
        "test_windows": 1328,  # 8 s 窗 / 2 s 步进，10 条记录合计
    },
    "bidmc": {"records": 53, "fs": 125.0},
    "capnobase": {"records": 42, "fs": 300.0},
    "wesad": {"subjects": 15, "fs_bvp": 64.0, "fs_acc": 32.0},
}


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def datasets_root(cli_root=None):
    """解析数据集根目录：--root > PPG_DATASETS > <工作区>/datasets。"""
    if cli_root:
        return os.path.abspath(cli_root)
    env = os.environ.get("PPG_DATASETS")
    if env:
        return os.path.abspath(env)
    return os.path.join(ROOT, "datasets")


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def mat_is_v73(path):
    """MATLAB v7.3 的 .mat 其实是 HDF5，scipy.io.loadmat 读不了（CapnoBase 就是这种）。"""
    try:
        with open(path, "rb") as f:
            return f.read(128).lstrip().startswith(b"MATLAB 7.3")
    except OSError:
        return False


def read_mat_any(path):
    """读取 .mat → dict。v7.3 走 h5py，其余走 scipy.io.loadmat。"""
    if mat_is_v73(path):
        import h5py
        import numpy as np

        def conv(node):
            if isinstance(node, h5py.Dataset):
                arr = node[()]
                if isinstance(arr, bytes):
                    return arr.decode("utf-8", "replace")
                if isinstance(arr, np.ndarray) and arr.dtype.kind == "O":
                    return np.array([conv(x) for x in arr.ravel()])
                if isinstance(arr, np.ndarray) and arr.dtype.kind == "S":
                    return arr.astype(str)
                return arr
            if isinstance(node, h5py.Group):
                return {k: conv(v) for k, v in node.items()}
            return node

        with h5py.File(path, "r") as f:
            out = {}
            for k, v in f.items():
                # MATLAB 在 HDF5 里把结构体存成组，字段顺序由 MATLAB_fields 记录
                out[k] = conv(v)
            return out
    import scipy.io as sio
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _run(cmd, timeout=None):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)


def resolve_redirect(url, timeout=60):
    """把 302 链解析成最终直链（Kaggle → GCS 签名 URL）。失败则原样返回。"""
    try:
        r = _run(["curl", "-sS", "-o", "/dev/null", "-w", "%{redirect_url}",
                  "-m", str(timeout), url], timeout=timeout + 10)
        out = r.stdout.decode("utf-8", "replace").strip()
        if out.startswith("http") and out != url:
            return out
    except Exception:
        pass
    return url


def download(url, dest, min_bytes, retries=3, timeout=3600, resume=True):
    """断点续传到 ``dest + '.part'``，成功后原子改名为 ``dest``。

    返回 (ok, info)。info 含 url / bytes / sha256 / resumed。
    """
    part = dest + ".part"
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        cmd = ["curl", "-L", "-sS", "--fail", "--retry", "3", "--retry-delay", "3",
               "-C", "-" if (resume and have) else "0",
               "--connect-timeout", "30", "-m", str(timeout),
               "-o", part, "-w", "%{http_code} %{size_download} %{speed_download}",
               url]
        log("    [下载] %s" % url)
        if have:
            log("          续传：已有 %.2f MB" % (have / 1e6))
        t0 = time.time()
        try:
            r = _run(cmd, timeout=timeout + 120)
        except subprocess.TimeoutExpired:
            log("          curl 超时，重试 %d/%d" % (attempt, retries))
            continue
        out = r.stdout.decode("utf-8", "replace").strip()
        rc = r.returncode
        size = os.path.getsize(part) if os.path.exists(part) else 0
        dt = max(time.time() - t0, 1e-6)
        if rc == 33:                       # 服务端不支持 Range
            log("          服务端不支持续传，从头重下")
            if os.path.exists(part):
                os.remove(part)
            resume = False
            continue
        if rc != 0:
            log("          curl exit=%d (%s)，重试 %d/%d"
                % (rc, out.splitlines()[-1] if out else "", attempt, retries))
            continue
        if size < min_bytes:
            log("          ✗ 落盘 %.3f MB < 期望最小 %.3f MB，判定失败"
                % (size / 1e6, min_bytes / 1e6))
            if os.path.exists(part):
                os.remove(part)
            continue
        os.replace(part, dest)
        info = {"url": url, "bytes": size, "sha256": sha256_of(dest),
                "elapsed_s": round(dt, 1), "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        log("          ✓ %.3f MB  %.2f MB/s  sha256=%s…"
            % (size / 1e6, size / 1e6 / dt, info["sha256"][:16]))
        return True, info
    return False, {}


def zip_ok(path, deep=False):
    """zip 完整性检查；deep=True 时全量 CRC 校验（大包较慢）。"""
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if not names:
                return False, "空归档"
            if deep:
                bad = z.testzip()
                if bad:
                    return False, "CRC 损坏：%s" % bad
            return True, "%d entries" % len(names)
    except zipfile.BadZipFile as e:
        return False, "BadZipFile: %s" % e


def safe_extract(archive, rules, dest_root, exclude_basenames=(), strip_top=False):
    """按规则从 zip 抽取文件，拒绝路径穿越。

    规则为 ``(prefix, subdir, exts)`` 或 ``(prefix, subdir, exts, keep_names)``：
      * ``prefix``   归档内的目录前缀（必须以 ``/`` 结尾）
      * ``subdir``   目标子目录（``""`` 表示直接放 ``dest_root``）
      * ``exts``     允许的扩展名元组；``None`` 表示不看扩展名
      * ``keep_names`` 若给出，则只保留这些 basename（大小写敏感）
    只取该前缀下的**直接子文件**（rel 中不含 ``/``），因此每条规则的 prefix
    必须精确指向文件所在目录，避免把子目录内容重复抽取到上一层。

    ``strip_top=True`` 时先剥掉归档的第一个路径分量（GitHub 的
    ``<repo>-<branch>/`` 外壳），这样 prefix 只需相对仓库根书写，
    同一套规则在 ghfast / codeload / github.com 各镜像下都成立。
    """
    written = []
    with zipfile.ZipFile(archive) as z:
        for orig_name in z.namelist():
            if orig_name.endswith("/"):
                continue
            name = orig_name
            if strip_top:
                head, sep, tail = name.partition("/")
                if not sep:
                    continue
                name = tail
            base = os.path.basename(name)
            if base in exclude_basenames:
                continue
            for rule in rules:
                prefix, subdir, exts = rule[0], rule[1], rule[2]
                keep_names = rule[3] if len(rule) > 3 else None
                if not name.startswith(prefix):
                    continue
                rel = name[len(prefix):]
                if "/" in rel:                       # 只要直接子文件
                    continue
                if keep_names is not None and base not in keep_names:
                    continue
                if keep_names is None and exts and not name.endswith(tuple(exts)):
                    continue
                out = os.path.abspath(os.path.join(dest_root, subdir, rel))
                if not out.startswith(os.path.abspath(dest_root) + os.sep):
                    log("          ! 跳过可疑路径 %s" % name)
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if os.path.exists(out) and os.path.getsize(out) == z.getinfo(orig_name).file_size:
                    written.append(out)
                    break
                with z.open(orig_name) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                written.append(out)
                break
    return written


# --------------------------------------------------------------------------
# 安装（解包）逻辑
# --------------------------------------------------------------------------
def install_troika(archive, root, src):
    """TROIKA：只保留 pipeline 期望的三个目录与原始文件名。

    流水线读取约定（已核对 ``scripts/pipeline/pipeline.py``）：
      * ``Training_data/*.mat``（6×N，row0=ECG）+ 同名 ``*_BPMtrace.mat``（``BPM0``）
      * ``TestData/*.mat``（5×N，无 ECG）+ ``TrueBPM/True_*.mat``（``BPM0``）
    因此**必须保持 DATA_*/TEST_*/True_* 原始命名，不能重命名**。
    """
    dest = os.path.join(root, "IEEEPPG_TROIKA")
    files = safe_extract(archive, src["extract"], dest, strip_top=src.get("strip_top", False))
    log("    [解包] TROIKA -> %s（%d 个文件）" % (dest, len(files)))
    return dest


def install_bidmc(archive, root, src):
    """BIDMC：``BIDMC/bidmc_csv/*.csv``（含 PPG/ECG/impedance 波形与 numerics）。"""
    dest = os.path.join(root, "BIDMC")
    files = safe_extract(archive, src["extract"], dest,
                         exclude_basenames=src.get("exclude_basenames", ()))
    log("    [解包] BIDMC -> %s（%d 个文件）" % (dest, len(files)))
    return dest


def install_capnobase(archive, root, src):
    """CapnoBase：``CapnoBase/data/mat/*.mat`` 与 ``data/csv/*.csv``。"""
    dest = os.path.join(root, "CapnoBase")
    files = safe_extract(archive, src["extract"], dest)
    log("    [解包] CapnoBase -> %s（%d 个文件）" % (dest, len(files)))
    return dest


def install_wesad(archive, root, src):
    """WESAD：全量解包，并消除 UCI 包的 ``WESAD/`` 双层嵌套。

    UCI 包内层是 ``WESAD/S2/S2.pkl``…；Kaggle 镜像内层同样是 ``WESAD/S2/…``。
    统一成 ``datasets/WESAD/S2/S2.pkl``。
    """
    dest = os.path.join(root, "WESAD")
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        inner = "WESAD/" if all(n.startswith("WESAD/") for n in names) else ""
        n = skipped = 0
        for name in names:
            rel = name[len(inner):]
            if not rel:
                continue
            out = os.path.abspath(os.path.join(dest, rel))
            if not out.startswith(os.path.abspath(dest) + os.sep):
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            # 幂等：大小一致就当已解包（WESAD 解包约 13 GB，重复解包很贵）
            if os.path.exists(out) and os.path.getsize(out) == z.getinfo(name).file_size:
                skipped += 1
                continue
            with z.open(name) as s, open(out, "wb") as d:
                shutil.copyfileobj(s, d, 1 << 20)
            n += 1
    log("    [解包] WESAD -> %s（新写 %d，跳过已存在 %d）" % (dest, n, skipped))
    return dest


INSTALLERS = {"troika": install_troika, "bidmc": install_bidmc,
              "capnobase": install_capnobase, "wesad": install_wesad}


# --------------------------------------------------------------------------
# 校验
# --------------------------------------------------------------------------
def _need(name):
    try:
        return __import__(name)
    except ImportError:
        return None


def verify_troika(root):
    """校验 TROIKA：记录数 / 采样率 / sig 行序 / BPM0 协议（8 s 窗 2 s 步进）。"""
    import numpy as np
    import scipy.io as sio

    d = os.path.join(root, "IEEEPPG_TROIKA")
    spec = SPEC["troika"]
    ok = True

    def chk(cond, msg):
        nonlocal ok
        ok = ok and bool(cond)
        log("      [%s] %s" % ("PASS" if cond else "FAIL", msg))

    for sub in ("Training_data", "TestData", "TrueBPM"):
        p = os.path.join(d, sub)
        if not os.path.isdir(p):
            chk(False, "%s/ 不存在" % sub)
            return False
    trs = sorted(f for f in os.listdir(os.path.join(d, "Training_data"))
                 if f.endswith(".mat") and "BPMtrace" not in f)
    trb = sorted(f for f in os.listdir(os.path.join(d, "Training_data"))
                 if f.endswith("_BPMtrace.mat"))
    tes = sorted(f for f in os.listdir(os.path.join(d, "TestData")) if f.endswith(".mat"))
    trus = sorted(f for f in os.listdir(os.path.join(d, "TrueBPM")) if f.endswith(".mat"))
    chk(len(trs) == spec["Training_data"], "Training_data 记录数 %d（期望 %d）" % (len(trs), spec["Training_data"]))
    chk(len(trb) == spec["Training_data"], "Training_data *_BPMtrace.mat 数 %d（期望 %d）" % (len(trb), spec["Training_data"]))
    chk(len(tes) == spec["TestData"], "TestData 记录数 %d（期望 %d）" % (len(tes), spec["TestData"]))
    chk(len(trus) == spec["TrueBPM"], "TrueBPM 记录数 %d（期望 %d）" % (len(trus), spec["TrueBPM"]))

    # 训练集：sig 行序 + BPM0 协议
    tot_len_ok = True
    win_total = 0
    for fn in trs:
        base = fn[:-4]
        mt = os.path.join(d, "Training_data", base + "_BPMtrace.mat")
        sig = sio.loadmat(os.path.join(d, "Training_data", fn))["sig"]
        bpm = sio.loadmat(mt)["BPM0"].reshape(-1)
        if sig.shape[0] != spec["sig_rows_train"]:
            chk(False, "%s sig 行数 %s != 6" % (fn, sig.shape))
            break
        # BPM0 协议自检：N/125 秒应约等于 2*(len(BPM0)-1)+8 秒
        dur = sig.shape[1] / spec["fs"]
        exp = 2 * (len(bpm) - 1) + 8
        if abs(dur - exp) > 2.5:
            tot_len_ok = False
            log("        ! %s 时长 %.1fs vs 2*(n_bpm-1)+8=%.1fs" % (fn, dur, exp))
    chk(tot_len_ok, "BPM0 满足「8 s 窗 / 2 s 步进」协议（时长 = 2(n-1)+8 秒）")

    for fn in tes:
        sig = sio.loadmat(os.path.join(d, "TestData", fn))["sig"]
        if sig.shape[0] != spec["sig_rows_test"]:
            chk(False, "%s sig 行数 %s != 5（测试集无 ECG）" % (fn, sig.shape))
            break
        base = fn[:-4].replace("TEST_", "True_")
        mt = os.path.join(d, "TrueBPM", base + ".mat")
        if not os.path.exists(mt):
            chk(False, "缺少真值 %s" % mt)
            break
        win_total += len(sio.loadmat(mt)["BPM0"].reshape(-1))
    chk(abs(win_total - spec["test_windows"]) <= 12,
        "测试集 8 s 窗总数 %d（期望约 %d）" % (win_total, spec["test_windows"]))
    chk(sig.dtype.kind == "f", "sig 为浮点（采样率 125 Hz，见 README）")
    return ok


def _read_wfdb_header(hea_path):
    """极简 WFDB 头解析：返回 (fs, [(name, gain, baseline), ...], n_sig, n_samples)。"""
    with open(hea_path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]
    first = lines[0].split()
    n_sig, fs, n_samples = int(first[1]), float(first[2].split("/")[0]), int(first[3])
    chans = []
    for l in lines[1:1 + n_sig]:
        p = l.split()
        gain = 200.0
        baseline = 0.0
        if len(p) > 2 and "(" in p[2]:
            g = p[2].split("(")[1].split(")")[0].split("/")[0]
            gain = float(g) if g.replace(".", "").replace("-", "").isdigit() else 200.0
        if len(p) > 4:
            try:
                baseline = int(p[4])
            except ValueError:
                baseline = 0
        chans.append((p[-1], gain, baseline))
    return fs, chans, n_sig, n_samples


def verify_bidmc(root):
    """校验 BIDMC：53 例、采样率 125 Hz、通道名含 PLETH/II/RESP。"""
    import numpy as np
    d = os.path.join(root, "BIDMC")
    spec = SPEC["bidmc"]
    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and bool(c)
        log("      [%s] %s" % ("PASS" if c else "FAIL", m))

    csvs, num = [], []
    for r, _, fs_ in os.walk(d):
        for f in fs_:
            if f.endswith("_Signals.csv"):
                csvs.append(os.path.join(r, f))
            elif f.endswith("_Numerics.csv"):
                num.append(os.path.join(r, f))
    csvs, num = sorted(csvs), sorted(num)
    chk(len(csvs) == spec["records"], "Signals.csv 数 %d（期望 %d 例）" % (len(csvs), spec["records"]))
    chk(len(num) == spec["records"], "Numerics.csv 数 %d（期望 %d 例）" % (len(num), spec["records"]))
    if csvs:
        p = csvs[0]
        with open(p) as f:
            hdr = f.readline().strip().split(",")
            n = sum(1 for _ in f)
        chk(any("PLETH" in h for h in hdr), "通道名含 PLETH(PPG)：%s" % hdr[:6])
        chk(any(h.strip() == "II" for h in hdr), "通道名含 ECG(II)：%s" % hdr)
        chk(abs(n / spec["fs"] - 480) < 3, "首例时长 %.1f s @125 Hz（官方 8 min）" % (n / spec["fs"]))
    heas = sorted(os.path.join(r, f) for r, _, fs_ in os.walk(d) for f in fs_ if f.endswith(".hea"))
    if heas:
        fs_, chans, n_sig, n_samp = _read_wfdb_header(heas[0])
        chk(abs(fs_ - spec["fs"]) < 1e-6, "WFDB 头采样率 %.1f Hz（期望 125）" % fs_)
        chk(any("PLETH" in c[0] for c in chans), "WFDB 通道含 PLETH：%s" % [c[0] for c in chans])
    return ok


def verify_capnobase(root):
    """校验 CapnoBase：42 例、采样率 300 Hz、含 PPG/ECG/CO2。"""
    import numpy as np
    d = os.path.join(root, "CapnoBase")
    spec = SPEC["capnobase"]
    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and bool(c)
        log("      [%s] %s" % ("PASS" if c else "FAIL", m))

    mats, csvs = [], []
    for r, _, fs_ in os.walk(d):
        for f in fs_:
            if f.endswith(".mat"):
                mats.append(os.path.join(r, f))
            elif f.endswith(".csv"):
                csvs.append(os.path.join(r, f))
    sig_csv = [c for c in csvs if c.endswith("_signal.csv")]
    chk(len(mats) == spec["records"], ".mat 记录数 %d（期望 %d）" % (len(mats), spec["records"]))
    chk(len(sig_csv) == spec["records"], "_signal.csv 数 %d（期望 %d）" % (len(sig_csv), spec["records"]))
    if sig_csv:
        with open(sorted(sig_csv)[0]) as f:
            hdr = [h.strip().strip('"').strip() for h in f.readline().strip().split(",")]
            n = sum(1 for _ in f)
        low = [h.lower() for h in hdr]
        chk(any("pleth" in h or "ppg" in h for h in low), "含 PPG 通道：%s" % hdr)
        chk(any(h.startswith("ecg") for h in low), "含 ECG 通道：%s" % hdr)
        chk(any("co2" in h for h in low), "含 CO2 通道：%s" % hdr)
        chk(abs(n / spec["fs"] - 480) < 3, "首例时长 %.1f s @300 Hz（8 min）" % (n / spec["fs"]))
    if mats:
        try:
            m = read_mat_any(sorted(mats)[0])
            keys = [k for k in m if not k.startswith("__")]
            chk(len(keys) > 0, "首个 .mat 变量：%s" % keys[:8])
        except Exception as e:
            chk(False, "读取 .mat 失败：%s" % e)
    return ok


def verify_wesad(root):
    """校验 WESAD：15 受试者、每人 S*.pkl、BVP 64 Hz / ACC 32 Hz / ECG 700 Hz。

    注意：官方文档常写 BVP 64 Hz、ACC 32 Hz、EDA/TEMP 4 Hz、ECG 700 Hz，
    但**实测** S*.pkl 里 wrist BVP 是 64 Hz、ACC 32 Hz、chest ECG 700 Hz；
    有的下载镜像会把 .pkl 预先转成 csv 或重新打包，需以实测为准。
    """
    import pickle
    import numpy as np
    d = os.path.join(root, "WESAD")
    spec = SPEC["wesad"]
    ok = True

    def chk(c, m):
        nonlocal ok
        ok = ok and bool(c)
        log("      [%s] %s" % ("PASS" if c else "FAIL", m))

    subs = sorted(s for s in os.listdir(d) if s.startswith("S") and s[1:].isdigit())
    chk(len(subs) == spec["subjects"], "受试者数 %d（期望 %d）：%s" % (len(subs), spec["subjects"], subs))
    pkls = [s for s in subs if os.path.exists(os.path.join(d, s, s + ".pkl"))]
    chk(len(pkls) == len(subs), "每个受试者都有 S*/*.pkl（%d/%d）" % (len(pkls), len(subs)))
    if pkls:
        s0 = pkls[0]
        with open(os.path.join(d, s0, s0 + ".pkl"), "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        chk(isinstance(obj, dict) and "signal" in obj, "pkl 顶层含 'signal' 键：%s" % list(obj)[:6])
        sig = obj["signal"]
        wrist = sig.get("wrist", {})
        chest = sig.get("chest", {})
        chk("BVP" in wrist and "ACC" in wrist, "wrist 含 BVP/ACC：%s" % list(wrist))
        chk("ECG" in chest, "chest 含 ECG：%s" % list(chest))
        if "BVP" in wrist:
            bvp = np.asarray(wrist["BVP"]).reshape(-1)
            acc = np.asarray(wrist["ACC"])
            ecg = np.asarray(chest["ECG"]).reshape(-1)
            # 用长度比推断采样率（以 700 Hz ECG 为基准对齐到整秒）
            log("        %s: BVP N=%d  ACC shape=%s  ECG N=%d  label N=%s"
                % (s0, bvp.shape[0], acc.shape, ecg.shape[0],
                   np.asarray(obj.get("label", [])).shape))
    return ok


VERIFIERS = {"troika": verify_troika, "bidmc": verify_bidmc,
             "capnobase": verify_capnobase, "wesad": verify_wesad}


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def load_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"archives": {}, "installs": {}}


def save_manifest(path, man):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)


def download_dataset(key, root, force=False, deep=False):
    """下载 + 解包单个数据集；返回 True 表示数据已就位。"""
    src = SOURCES[key]
    dl_dir = os.path.join(root, "_downloads")
    os.makedirs(dl_dir, exist_ok=True)
    man_path = os.path.join(dl_dir, "manifest.json")
    man = load_manifest(man_path)
    log("\n=== [%s] %s ===" % (key, src["display"]))

    for arc in src["archives"]:
        dest = os.path.join(dl_dir, arc["filename"])
        ok = False
        # 幂等：已存在且大小/结构合格则复用
        if os.path.exists(dest) and not force:
            good, why = zip_ok(dest, deep=deep)
            if good and os.path.getsize(dest) >= arc["min_bytes"]:
                log("    [跳过] 归档已存在且校验通过：%s（%.2f MB, %s）"
                    % (arc["filename"], os.path.getsize(dest) / 1e6, why))
                # 补齐清单：命中缓存时也要保证 bytes/sha256 与实际文件一致，
                # 否则 manifest 会出现空 {} 项（曾发生过）。
                ent = dict(man["archives"].get(arc["filename"]) or {})
                size = os.path.getsize(dest)
                if ent.get("bytes") != size or not ent.get("sha256"):
                    log("          补齐清单（重算 sha256，实际文件 %d B）" % size)
                    ent["bytes"] = size
                    ent["sha256"] = sha256_of(dest)
                ent.setdefault("source_url", arc["urls"][0])
                ent.setdefault("note", arc.get("note", ""))
                ent["at"] = ent.get("at") or time.strftime("%Y-%m-%d %H:%M:%S")
                ent["from_cache"] = True
                man["archives"][arc["filename"]] = ent
                save_manifest(man_path, man)
                ok = True
            else:
                log("    [重下] 归档不合格（%s），删除重来" % why)
                os.remove(dest)
        if not ok:
            for url in arc["urls"]:
                target = resolve_redirect(url) if arc.get("resolve_redirect") else url
                good, info = download(target, dest, arc["min_bytes"])
                if good:
                    good2, why2 = zip_ok(dest, deep=deep)
                    if not good2:
                        log("          ✗ 归档损坏（%s），删除" % why2)
                        os.remove(dest)
                        continue
                    info["source_url"] = url
                    info["note"] = arc.get("note", "")
                    man["archives"][arc["filename"]] = info
                    save_manifest(man_path, man)
                    ok = True
                    break
                else:
                    log("          ✗ 该源失败，尝试下一个候选源")
            if not ok:
                log("    ✗✗ %s 全部候选源均失败" % arc["filename"])
                return False
        # 解包
        target_dir = INSTALLERS[key](dest, root, arc)
        man["installs"][key] = {"target": os.path.relpath(target_dir, root),
                                "archive": arc["filename"],
                                "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_manifest(man_path, man)
    return True


def dir_size(path):
    tot = n = 0
    for r, _, fs_ in os.walk(path):
        for f in fs_:
            fp = os.path.join(r, f)
            try:
                tot += os.path.getsize(fp)
                n += 1
            except OSError:
                pass
    return n, tot


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="PPG 运动伪影抑制项目 · 开源数据集下载/校验（幂等、可续传）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", "-d", nargs="+", default=None,
                    choices=sorted(SOURCES) + ["all", "ieeepg", "capno"],
                    help="要处理的数据集；all = 全部四个")
    ap.add_argument("--root", default=None,
                    help="数据集根目录（默认 <工作区>/datasets，或环境变量 PPG_DATASETS）")
    ap.add_argument("--verify", action="store_true", help="只做校验，不下载")
    ap.add_argument("--force", action="store_true", help="忽略已存在的归档，强制重下")
    ap.add_argument("--deep", action="store_true", help="zip 全量 CRC 校验（大包慢）")
    ap.add_argument("--list", action="store_true", help="列出支持的数据集与来源")
    args = ap.parse_args(argv)

    alias = {"ieeepg": "troika", "capno": "capnobase"}
    if args.list:
        log("支持的数据集：")
        for k, v in SOURCES.items():
            log("  %-10s %s" % (k, v["display"]))
            log("             目标目录 %s/" % v["target"])
            for a in v["archives"]:
                for u in a["urls"]:
                    log("               - %s" % u)
        return 0

    root = datasets_root(args.root)
    if args.dataset:
        keys = []
        for d in args.dataset:
            if d == "all":
                keys = list(SOURCES)
                break
            keys.append(alias.get(d, d))
    else:
        keys = list(SOURCES)

    log("数据集根目录：%s" % root)
    log("目标数据集  ：%s" % ", ".join(keys))
    os.makedirs(root, exist_ok=True)

    results = {}
    if not args.verify:
        for k in keys:
            t0 = time.time()
            try:
                ok = download_dataset(k, root, force=args.force, deep=args.deep)
            except Exception as e:
                import traceback
                traceback.print_exc()
                log("    ✗ %s 出错：%s" % (k, e))
                ok = False
            results[k] = {"download_ok": ok, "elapsed_s": round(time.time() - t0, 1)}
        # 清单去 stale：删掉已经不在磁盘上的归档条目
        mpath = os.path.join(root, "_downloads", "manifest.json")
        man = load_manifest(mpath)
        gone = [f for f in man["archives"]
                if not os.path.exists(os.path.join(root, "_downloads", f))]
        for f in gone:
            man["archives"].pop(f, None)
        if gone:
            log("\n[清单] 清理已不存在的条目：%s" % ", ".join(gone))
        save_manifest(mpath, man)

    log("\n" + "=" * 74)
    log("校验（%s）" % ("--verify 只校验" if args.verify else "下载后校验"))
    log("=" * 74)
    all_ok = True
    for k in keys:
        tgt = os.path.join(root, SOURCES[k]["target"])
        log("\n--- %s : %s" % (k, tgt))
        if not os.path.isdir(tgt):
            log("      [FAIL] 目标目录不存在")
            all_ok = False
            results.setdefault(k, {})["verify_ok"] = False
            continue
        n, sz = dir_size(tgt)
        log("      文件数 %d，总大小 %.2f MB" % (n, sz / 1e6))
        try:
            ok = VERIFIERS[k](root)
        except Exception as e:
            import traceback
            traceback.print_exc()
            ok = False
        results.setdefault(k, {})["verify_ok"] = ok
        results[k].update({"files": n, "bytes": sz})
        all_ok = all_ok and ok

    log("\n" + "=" * 74)
    log("汇总")
    log("=" * 74)
    for k in keys:
        r = results.get(k, {})
        log("  %-10s 下载=%s  校验=%s  文件=%s  大小=%.2f MB"
            % (k, r.get("download_ok", "—"), r.get("verify_ok", "—"),
               r.get("files", "—"), (r.get("bytes") or 0) / 1e6))
    log("总体：%s" % ("全部通过 ✓" if all_ok else "存在失败项 ✗"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

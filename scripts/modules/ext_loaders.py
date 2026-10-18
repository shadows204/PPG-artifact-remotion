#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""外部公开数据集加载器：TROIKA / BIDMC / CapnoBase / WESAD。

用途
----
把四个来源、四种物理格式、四种采样率的数据统一成一个 dict，供 RLS / U-Net /
心率决策层复用。CIME PPG 不走这里（它有 ``scripts/modules/prepare_cime.py``）。

统一返回结构
------------
每个 ``load_*()`` 返回一个 **record dict**：

=========================  ====================================================
键                          含义
=========================  ====================================================
``name``                   str，记录标识（如 ``"DATA_01_TYPE01"``）
``dataset``                str，``"troika"`` / ``"bidmc"`` / ``"capnobase"`` /
                           ``"wesad"``
``fs``                     float，``ppg``/``acc``/``ecg`` 共用的采样率（Hz）
``ppg``                    ``(n_lead, N)`` float64，光电容积脉搏波，n_lead∈{1,2}
``acc``                    ``(3, N)`` float64 或 ``None``，三轴加速度
``ecg``                    ``(N,)`` float64 或 ``None``，心电（作为心率参考）
``hr_ref``                 ``(M,)`` float64 或 ``None``，心率真值（bpm）
``hr_fs``                  float 或 ``None``，``hr_ref`` 的采样率（Hz）
``hr_win_s``               float 或 ``None``，每个 ``hr_ref`` 值覆盖的窗长（秒）
``meta``                   dict，来源文件、通道名、单位、坑等
=========================  ====================================================

``hr_ref`` 的语义由 ``hr_fs`` + ``hr_win_s`` 共同确定：
  * TROIKA：``hr_fs=0.5``（2 s 步进）、``hr_win_s=8.0``；
    ``hr_ref[i]`` 对应原始记录 ``[2i, 2i+8)`` 秒窗口。
  * BIDMC：``hr_fs=1.0``、``hr_win_s=1.0``（床旁监护仪 1 Hz 瞬时心率趋势）。
  * CapnoBase：**逐搏**瞬时心率（``hr_fs`` = 中位搏动率 ≈1.6 Hz，但采样不规则、
    ``hr_win_s=None``）；真实时间轴在 ``hr_ref_t``。
  * WESAD：``None``（它没有心率真值，只有情绪标签）。

``hr_ref_t`` 是 ``hr_ref`` 每个值对应的**物理时间**（秒）。TROIKA/BIDMC 是规则
网格（也可以直接用 ``hr_fs`` 推），CapnoBase 是逐搏不规则时间轴——**必须用
``hr_ref_t`` 才能正确对齐**，按 ``hr_fs`` 均匀假设会错位。

用法
----
::

    import sys; sys.path.insert(0, "scripts/modules")
    from ext_loaders import load_troika, iter_troika, to_100hz, DATASETS

    r = load_troika("DATA_01_TYPE01")          # 单条记录
    r["ppg"].shape, r["acc"].shape, r["fs"]    # (2, 37937) (3, 37937) 125.0
    r = to_100hz(r)                            # 重采样到模型用的 100 Hz

    for r in iter_troika(split="test"):        # 官方留出测试集 10 条
        ...

    r = load_bidmc("bidmc_01")
    r = load_capnobase("0009_8min")
    r = load_wesad("S2")

命令行自检（打印每个数据集的实际记录数 / 采样率 / 通道名）::

    python scripts/modules/ext_loaders.py            # 全部
    python scripts/modules/ext_loaders.py troika      # 单个

数据集根目录默认 ``<工作区>/datasets``，可用环境变量 ``PPG_DATASETS`` 覆盖
（与 ``scripts/_common/paths.py::datasets_root`` 一致）。
"""
from __future__ import annotations

import glob
import os
import pickle
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))              # <root>/scripts → _common

try:                                                   # 复用工作区统一路径解析
    from _common import paths as P
except Exception:                                      # pragma: no cover
    P = None

__all__ = [
    "DATASETS", "datasets_root", "list_records",
    "load_troika", "iter_troika", "load_bidmc", "iter_bidmc",
    "load_capnobase", "iter_capnobase", "load_wesad", "iter_wesad",
    "resample", "to_fs", "to_100hz", "read_mat_any", "mat_is_v73",
]

DATASETS = ("troika", "bidmc", "capnobase", "wesad")

# 各数据集的官方/实测采样率（Hz），供调用方对照
NATIVE_FS = {"troika": 125.0, "bidmc": 125.0, "capnobase": 300.0,
             "wesad_bvp": 64.0, "wesad_acc": 32.0, "wesad_ecg": 700.0}


def datasets_root(root: str | None = None) -> str:
    """数据集根目录：``root`` > 环境变量 ``PPG_DATASETS`` > ``<工作区>/datasets``。"""
    if root:
        return os.path.abspath(root)
    env = os.environ.get("PPG_DATASETS")
    if env:
        return os.path.abspath(env)
    if P is not None:
        return P.datasets_root()
    return os.path.abspath(os.path.join(HERE, "..", "..", "datasets"))


def _record(name, dataset, fs, ppg, acc=None, ecg=None,
            hr_ref=None, hr_ref_t=None, hr_fs=None, hr_win_s=None, **meta):
    """打包成统一结构（内部工具）：统一 dtype=float64，形状 (n_lead, N)。"""
    def _2d(x):
        if x is None:
            return None
        a = np.asarray(x, dtype=np.float64)
        return a[None, :] if a.ndim == 1 else a

    def _1d(x):
        return None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)

    hr_ref = _1d(hr_ref)
    return {
        "name": name, "dataset": dataset, "fs": float(fs),
        "ppg": _2d(ppg), "acc": _2d(acc) if acc is not None else None,
        "ecg": _1d(ecg), "hr_ref": hr_ref, "hr_ref_t": _1d(hr_ref_t),
        # hr 缺失时不要把 hr_fs 留在那里，否则调用方会以为有真值
        "hr_fs": None if (hr_ref is None or hr_fs is None) else float(hr_fs),
        "hr_win_s": None if (hr_ref is None or hr_win_s is None) else float(hr_win_s),
        "meta": meta,
    }


# ==========================================================================
# 重采样工具
# ==========================================================================
def _poly_ratio(fs_in, fs_out):
    """把 fs_out/fs_in 化成整数上/下采样比，避免浮点比导致 resample_poly 报错。"""
    f = Fraction(fs_out / fs_in).limit_denominator(1000)
    return f.numerator, f.denominator


def resample(x, fs_in, fs_out, axis=-1):
    """多相重采样任意数组（scipy.signal.resample_poly）。

    ``fs_out == fs_in`` 时原样返回（不做多余拷贝）。
    """
    if x is None or abs(float(fs_out) - float(fs_in)) < 1e-9:
        return x
    from scipy.signal import resample_poly
    up, down = _poly_ratio(fs_in, fs_out)
    return resample_poly(np.asarray(x, dtype=np.float64), up, down, axis=axis)


def to_fs(rec, fs_out):
    """把一条 record 的 ppg/acc/ecg/hr_ref 全部换算到 ``fs_out``（Hz）。

    * ``ppg`` / ``acc`` / ``ecg`` 走多相重采样；
    * ``hr_ref`` 只改 ``hr_fs`` 这个"元数据采样率"，**不插值**——
      因为 8 秒窗真值插值到 100 Hz 会产生虚假的高时间分辨率。
    * WESAD 那种"BVP 64 Hz + ACC 32 Hz"的异构情况已在 ``load_wesad``
      里先对齐，所以这里可以放心共用 ``rec["fs"]``。
    """
    fs_in = rec["fs"]
    if abs(float(fs_out) - fs_in) < 1e-9:
        return rec
    out = dict(rec)
    out["ppg"] = resample(rec["ppg"], fs_in, fs_out, axis=-1)
    out["acc"] = None if rec["acc"] is None else resample(rec["acc"], fs_in, fs_out, axis=-1)
    out["ecg"] = None if rec["ecg"] is None else resample(rec["ecg"], fs_in, fs_out, axis=-1)
    out["fs"] = float(fs_out)
    if rec.get("hr_fs"):
        # 真值的时间轴按"原始秒数"缩放：步进秒数不变，故 hr_fs 也不变
        out["hr_fs"] = rec["hr_fs"]
    out["meta"] = dict(rec.get("meta", {}))
    out["meta"]["resampled_from_fs"] = fs_in
    return out


def to_100hz(rec):
    """重采样到 CIME 模型的 100 Hz（流水线 step6/step9/step10 用的就是这一步）。"""
    return to_fs(rec, 100.0)


# ==========================================================================
# .mat 读取（v5 走 scipy，v7.3/HDF5 走 h5py）
# ==========================================================================
def mat_is_v73(path):
    """判断 .mat 是否为 MATLAB v7.3（HDF5 容器）。"""
    try:
        with open(path, "rb") as f:
            return f.read(128).lstrip().startswith(b"MATLAB 7.3")
    except OSError:
        return False


def read_mat_any(path):
    """读 .mat → dict；自动区分 v5（scipy）与 v7.3（h5py）。

    坑：CapnoBase 的 .mat 是 v7.3，``scipy.io.loadmat`` 会直接抛
    ``NotImplementedError: Please use HDF reader for matlab v7.3 files``。
    """
    if mat_is_v73(path):
        import h5py

        def decode(node):
            if isinstance(node, h5py.Dataset):
                arr = node[()]
                if (node.attrs.get("MATLAB_class") or b"") == b"char":
                    if np.asarray(arr).dtype.kind in "ui":
                        return "".join(chr(int(c)) for c in np.asarray(arr).ravel() if int(c) != 0)
                    return "".join(x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)
                                   for x in np.asarray(arr).ravel())
                return arr
            if isinstance(node, h5py.Group):
                return {k: decode(v) for k, v in node.items()}
            return node

        with h5py.File(path, "r") as f:
            return {k: decode(v) for k, v in f.items()}
    import scipy.io as sio
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def _unwrap_y(node):
    """把 CapnoBase 的 ``{'y': (1,N)}`` / ``{'x': (1,N)}`` / 纯数组统一成 (N,) 一维数组。

    注意 ``signal/*`` 用 ``y`` 存数值、``x`` 存时间；而 ``labels/*/peak`` 只有
    ``x``（就是搏动位置本身）。所以两个键都要认，优先 ``y``。
    """
    if isinstance(node, dict):
        for k in ("y", "x"):
            if k in node:
                node = node[k]
                break
    return np.asarray(node, dtype=np.float64).reshape(-1)


# ==========================================================================
# 1. TROIKA / IEEE SP Cup 2015 (IEEEPPG)
# ==========================================================================
# 目录约定（与 scripts/pipeline/pipeline.py 的 step6/step9/step10 完全一致）：
#   IEEEPPG_TROIKA/Training_data/DATA_XX_TYPEYY.mat          -> 变量 sig (6, N)
#   IEEEPPG_TROIKA/Training_data/DATA_XX_TYPEYY_BPMtrace.mat -> 变量 BPM0 (M,)
#   IEEEPPG_TROIKA/TestData/TEST_SXX_TYY.mat                 -> 变量 sig (5, N)
#   IEEEPPG_TROIKA/TrueBPM/True_SXX_TYY.mat                  -> 变量 BPM0 (M,)
TROIKA_DIRNAME = "IEEEPPG_TROIKA"


def list_records(dataset, split=None, root=None):
    """列出某数据集可用的记录 id（排序后）。"""
    dataset = dataset.lower()
    base = datasets_root(root)
    if dataset in ("troika", "ieeepg"):
        d = os.path.join(base, TROIKA_DIRNAME)
        if split == "test":
            dd = os.path.join(d, "TestData")
            return sorted(f[:-4] for f in os.listdir(dd)
                          if f.endswith(".mat")) if os.path.isdir(dd) else []
        dd = os.path.join(d, "Training_data")
        if not os.path.isdir(dd):
            return []
        return sorted(f[:-4] for f in os.listdir(dd)
                      if f.endswith(".mat") and "BPMtrace" not in f)
    if dataset == "bidmc":
        import re
        d = os.path.join(base, "BIDMC")
        ids = set()
        for r, _, fs_ in os.walk(d):
            for f in fs_:
                m = re.match(r"(bidmc_\d+)_Signals\.csv$", f)
                if m:
                    ids.add(m.group(1))
        if not ids:                                     # 回退：只用 WFDB 头
            for r, _, fs_ in os.walk(d):
                for f in fs_:
                    m = re.match(r"(bidmc\d+)\.hea$", f)
                    if m:
                        ids.add(m.group(1))
        return sorted(ids)
    if dataset in ("capnobase", "capno"):
        d = os.path.join(base, "CapnoBase", "data", "mat")
        if not os.path.isdir(d):
            d = os.path.join(base, "CapnoBase", "data", "csv")
            return sorted(f[: -len("_signal.csv")] for f in os.listdir(d)
                          if f.endswith("_signal.csv")) if os.path.isdir(d) else []
        return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".mat"))
    if dataset == "wesad":
        d = os.path.join(base, "WESAD")
        if not os.path.isdir(d):
            return []
        return sorted(s for s in os.listdir(d)
                      if s.startswith("S") and s[1:].isdigit()
                      and os.path.isdir(os.path.join(d, s)))
    raise ValueError("未知数据集：%s（可选 %s）" % (dataset, DATASETS))


def _troika_paths(record, split, root):
    d = os.path.join(datasets_root(root), TROIKA_DIRNAME)
    if split == "test" or record.startswith("TEST_"):
        rec = record if record.startswith("TEST_") else "TEST_" + record
        sig = os.path.join(d, "TestData", rec + ".mat")
        bpm = os.path.join(d, "TrueBPM", rec.replace("TEST_", "True_") + ".mat")
    else:
        sig = os.path.join(d, "Training_data", record + ".mat")
        bpm = os.path.join(d, "Training_data", record + "_BPMtrace.mat")
    return sig, bpm


def load_troika(record, split="train", root=None, fs_out=None, with_ecg=True):
    """加载 TROIKA / IEEE SP Cup 2015 的一条记录。

    字段来源
    --------
    * ``ppg``    = ``sig[1:3]``（训练集）/ ``sig[0:2]``（测试集）—— **两路 PPG 原始 ADC**
    * ``ecg``    = ``sig[0]``（**仅训练集有**；测试集 5 行里没有 ECG）
    * ``acc``    = ``sig[3:6]``（训练集）/ ``sig[2:5]``（测试集）—— 三轴加速度，**单位 g**
    * ``hr_ref`` = ``BPM0``（来自同名 ``*_BPMtrace.mat``）；``hr_win_s=8.0``、``hr_fs=0.5``
    * ``fs``     = 125.0

    坑（务必注意，踩过会直接让指标失真）
    ------------------------------------
    1. **``sig`` 第 0 行是 ECG，不是 PPG**。用频带能量实证：row0 的
       HF(5–40 Hz)/LF(0.5–4 Hz) = **3.06**，而 row1..row5 只有 0.018~0.13，
       ECG 的高频成分远多于脉搏波。按"row0=PPG"取信号会得到完全错误的 HR。
    2. **训练集 6 行 vs 测试集 5 行**：官方留出测试集 ``TestData`` 的 ``sig``
       只有 5 行 ``[PPG1, PPG2, ACCx, ACCy, ACCz]``（无 ECG）。所以测试集上
       **拿不到 ECG 参考**，任何"用 ECG 定 HR 真值"的做法只能用于训练集。
    3. **``BPM0`` 是 8 秒窗、2 秒步进**：``BPM0[i]`` 不是第 i 秒，而是
       ``[2i, 2i+8)`` 秒这个 8 秒窗的心率。窗口间重叠 6 秒。因此
       ``len(BPM0) ≈ (时长-8)/2 + 1``：实测训练记录时长 303.5 s → 148 个值
       （2×147+8 = 302 s），官方测试集 10 条合计 **1328** 个窗。
       用"每秒一个真值"去对齐会整体错位。
    4. 两路 PPG 幅度差异大（std 96 vs 162），ACC 是 g 量级（std ≈ 0.6~0.8），
       PPG 是 ADC 量级（std ≈ 1e2）。**跨数据集/跨通道必须各自归一化**，
       否则同一组 RLS 超参的等效步长完全不同。
    5. 文件名匹配：``TestData/TEST_SXX_TYY.mat`` ↔ ``TrueBPM/True_SXX_TYY.mat``
       （``TEST_`` → ``True_``）。训练集真值不在 ``sig`` 里，在
       ``<同名>_BPMtrace.mat`` 的 ``BPM0`` 里。
    """
    import scipy.io as sio
    sig_p, bpm_p = _troika_paths(record, split, root)
    if not os.path.exists(sig_p):
        raise FileNotFoundError("找不到 TROIKA 记录：%s" % sig_p)
    sig = np.asarray(sio.loadmat(sig_p)["sig"], dtype=np.float64)
    is_test = sig.shape[0] == 5
    if is_test:
        ppg, acc, ecg = sig[0:2], sig[2:5], None
    else:
        ecg, ppg, acc = sig[0], sig[1:3], sig[3:6]
    if not with_ecg:
        ecg = None
    hr = None
    if os.path.exists(bpm_p):
        hr = np.asarray(sio.loadmat(bpm_p)["BPM0"], dtype=np.float64).reshape(-1)
    # BPM0[i] 对应 [2i, 2i+8) 秒窗口 → 时间轴就是 2i
    hr_t = None if hr is None else 2.0 * np.arange(len(hr), dtype=np.float64)
    rec = _record(record, "troika", 125.0, ppg, acc=acc, ecg=ecg,
                  hr_ref=hr, hr_ref_t=hr_t, hr_fs=0.5, hr_win_s=8.0,
                  source=sig_p, bpm_source=bpm_p if os.path.exists(bpm_p) else None,
                  n_samples=sig.shape[1], raw_rows=int(sig.shape[0]),
                  split="test" if is_test else "train",
                  channels=(["PPG1", "PPG2", "ACCx", "ACCy", "ACCz"] if is_test
                            else ["ECG", "PPG1", "PPG2", "ACCx", "ACCy", "ACCz"]),
                  units="PPG=ADC(a.u.), ACC=g, ECG=ADC(a.u.)",
                  note="sig 行序已实证：0=ECG,1-2=PPG,3-5=ACC；BPM0 为 8 s 窗/2 s 步进")
    return to_fs(rec, fs_out) if fs_out else rec


def iter_troika(split="train", root=None, fs_out=None):
    """遍历 TROIKA 某 split 的全部记录（生成器）。"""
    for r in list_records("troika", split=split, root=root):
        yield load_troika(r, split=split, root=root, fs_out=fs_out)


# ==========================================================================
# 2. BIDMC（PhysioNet bidmc 1.0.0，53 例）
# ==========================================================================
def load_bidmc(record="bidmc_01", root=None, fs_out=None, use_wfdb=False):
    """加载 BIDMC PPG and Respiration Dataset 的一例（默认 8 分钟）。

    字段来源
    --------
    * ``ppg``    = ``bidmc_XX_Signals.csv`` 的 **PLETH** 列（指夹 PPG），125 Hz
    * ``ecg``    = 同表 **II** 列（II 导联 ECG），125 Hz
    * ``acc``    = ``None`` —— **该数据集没有加速度计**
    * ``hr_ref`` = ``bidmc_XX_Numerics.csv`` 的 **HR** 列，1 Hz；``hr_win_s=1.0``
    * ``fs``     = 125.0

    其他可用通道（放在 ``meta["aux"]``）：RESP（胸阻抗呼吸）、V、AVR、PULSE、
    SpO2、RESP(numerics)。

    坑
    --
    1. ``_Signals.csv`` 首列是 ``Time [s]``，**表头带空格**（``PLETH``、
       `` V``、`` II``），按列名取通道时一定要 ``strip()``。
    2. ``_Numerics.csv`` 是 1 Hz 趋势，``HR`` 来自床旁监护仪（不是从 ECG
       逐搏算出来的），和 PLETH 不是同一路信号；评测时它更接近"临床参考"
       而不是"金标准逐搏心率"。
    3. 时间列单位是**秒**（在 125 Hz 下步进 0.008 s），不要当成样本下标。
    4. 本目录同时放了 WFDB 原始件（``bidmc01.hea/.dat``，物理单位校正后约
       8 分钟 @125 Hz）；``use_wfdb=True`` 时优先用 ``wfdb`` 读（若已安装），
       否则回退 CSV。两条路径的 PLETH 数值尺度不同（WFDB 带 gain/baseline
       校正），但**同一例的波形应一致**，可互为校验。
    """
    ids = list_records("bidmc", root=root)
    if record not in ids:
        cand = [i for i in ids if i.replace("bidmc_", "bidmc") == record]
        if not cand:
            raise FileNotFoundError("找不到 BIDMC 记录 %s（可用 %d 例，如 %s）"
                                    % (record, len(ids), ids[:3]))
        record = cand[0]
    base = datasets_root(root)
    # 定位 Signals.csv（CSV 版可能在 bidmc_csv/ 子目录）
    sig_p = num_p = None
    for r, _, fs_ in os.walk(os.path.join(base, "BIDMC")):
        for f in fs_:
            if f == record + "_Signals.csv":
                sig_p = os.path.join(r, f)
            elif f == record + "_Numerics.csv":
                num_p = os.path.join(r, f)
    if sig_p is None:
        raise FileNotFoundError("找不到 %s_Signals.csv" % record)

    names, cols = _read_csv_cols(sig_p)
    idx = {n: i for i, n in enumerate(names)}
    if "PLETH" not in idx:
        raise ValueError("BIDMC 表头缺 PLETH（实际列：%s）；禁止静默取第 1 列（那是 RESP）"
                         % list(idx)[:8])
    ppg = cols[idx["PLETH"]]
    ecg = cols[idx["II"]] if "II" in idx else None
    aux = {n: cols[i] for n, i in idx.items()
           if n not in ("Time [s]", "Time", "PLETH", "II")}

    hr = None
    hr_t = None
    aux_num = {}
    if num_p and os.path.exists(num_p):
        nnames, ncols = _read_csv_cols(num_p)
        nidx = {n: i for i, n in enumerate(nnames)}
        if "HR" in nidx:
            hr = ncols[nidx["HR"]]
            hr_t = ncols[nidx["Time [s]"]] if "Time [s]" in nidx else None
        aux_num = {n: ncols[i] for n, i in nidx.items()
                   if n not in ("Time [s]", "Time", "HR")}

    rec = _record(record, "bidmc", 125.0, ppg[None, :], acc=None,
                  ecg=ecg, hr_ref=hr, hr_ref_t=hr_t, hr_fs=1.0, hr_win_s=1.0,
                  source=sig_p, numerics_source=num_p,
                  n_samples=ppg.shape[0],
                  channels=names, aux=aux, aux_numerics=aux_num,
                  units="PLETH/II = 0..1 归一化 ADC（bedside monitor）",
                  note="无 ACC；HR 为 1 Hz 床旁趋势，非逐搏金标准")
    return to_fs(rec, fs_out) if fs_out else rec


def _read_csv_cols(path, max_rows=None):
    """读 CSV（表头带空格、可能带引号）→ (列名 list, 每列 ndarray)。"""
    import csv
    with open(path, newline="") as f:
        rd = csv.reader(f)
        names = [c.strip().strip('"').strip() for c in next(rd)]
        rows = []
        for i, row in enumerate(rd):
            if max_rows is not None and i >= max_rows:
                break
            if row:
                rows.append(row)
    arr = np.array(rows, dtype=np.float64)
    return names, [arr[:, j] for j in range(arr.shape[1])]


def iter_bidmc(root=None, fs_out=None):
    """遍历 BIDMC 全部 53 例。"""
    for r in list_records("bidmc", root=root):
        yield load_bidmc(r, root=root, fs_out=fs_out)


# ==========================================================================
# 3. CapnoBase（Borealis Dataverse r4，DOI 10.5683/SP2/NLB8IT，42 例；不在 PhysioNet 上）
# ==========================================================================
def load_capnobase(record="0009_8min", root=None, fs_out=None, use_csv=False):
    """加载 CapnoBase 的一例（8 分钟，术中/麻醉场景）。

    字段来源
    --------
    * ``ppg``    = ``signal/pleth/y``（或 ``*_signal.csv`` 的 ``pleth_y`` 列）
    * ``ecg``    = ``signal/ecg/y``（或 ``ecg_y`` 列）
    * ``acc``    = ``None`` —— **该数据集没有加速度计**
    * ``hr_ref`` = ``reference/hr/pleth/y``（**逐搏瞬时心率**，bpm），
      时间轴在 ``hr_ref_t`` = ``reference/hr/pleth/x``（搏动时刻，秒，不等间隔）
    * ``fs``     = 300.0（来自 ``param/samplingrate/{pleth,ecg,co2}``）
    * 额外：``meta["co2"]`` = ``signal/co2/y``（呼气末 CO2 波形）、
      ``meta["pleth_peaks"]`` / ``meta["pleth_artifacts"]``（人工标注的搏动
      位置与伪影位置，样本下标）

    坑
    --
    1. **CapnoBase 不在 PhysioNet 上**（``/content/capnobase/`` 与
       ``/content/capno/`` 均 404）。官方宿主是 **Borealis Dataverse**，
       DOI ``10.5683/SP2/NLB8IT``（r4 版，2021-03-01 迁移到 Dataverse）；
       本目录的 ``README.txt`` / ``MANIFEST.TXT`` / ``data/mat`` / ``data/csv``
       / ``preview`` 正是 Dataverse 的包布局。许可证为"该数据集专用条款"
       （README 原文：This data is free to use for research），
       **不是** CC0，引用时请标注原论文 Karlen et al., IEEE TBME 2013。
    2. **``.mat`` 是 MATLAB v7.3（HDF5）**，``scipy.io.loadmat`` 会抛
       "Please use HDF reader for matlab v7.3 files"。本模块用 h5py 读。
       另外 v7.3 里字符数组是 uint16 码点，需要按 ``MATLAB_class == 'char'``
       解码，否则你会看到一堆数字。
    3. **没有时间向量**：``signal/*/y`` 只是 ``(1, N)`` 行向量，时间必须自己
       用 ``arange(N)/fs`` 造。别指望像 BIDMC 那样有一列 ``Time [s]``。
    4. 信号**已被原作者做过交流耦合/去均值**（``pleth`` 均值≈0，取值范围约
       ±2），不是原始 ADC；跨数据集比较时要先各自标准化。
    5. **``*_reference.csv`` / ``*_labels.csv`` 不是规整表格**：表头声明
       9 列（``rr_co2_x, rr_co2_y, hr_pleth_x, hr_pleth_y, hr_ecg_x,
       hr_ecg_y, units_*``），但数据行是一整行空格分隔的数字，列数与表头
       对不上（MATLAB 导出残留）。**HR 真值只能从 .mat 读**，别用 CSV。
    6. ``hr_ref`` 是**逐搏**瞬时心率而不是规则网格：8 分钟约 815 个点，
       中位间隔 ≈0.61 s（≈1.64 Hz）。所以 ``hr_win_s=None``，
       对齐时必须用 ``hr_ref_t``。
    """
    base = datasets_root(root)
    dmat = os.path.join(base, "CapnoBase", "data", "mat", record + ".mat")
    dcsv = os.path.join(base, "CapnoBase", "data", "csv", record + "_signal.csv")
    if use_csv and os.path.exists(dcsv):
        names, cols = _read_csv_cols(dcsv)
        # ⚠️ 必须**按列名**取：CapnoBase 的 CSV 列序是 co2_y, pleth_y, ecg_y（历史按下标 → 三列全错）
        _nm = {str(n).strip().lower(): c for n, c in zip(names, cols)}
        if not any(k.startswith("pleth") for k in _nm):
            raise ValueError("CapnoBase CSV 缺少 pleth 列：%s" % list(_nm)[:8])
        ppg = next(c for k, c in _nm.items() if k.startswith("pleth"))
        ecg = next((c for k, c in _nm.items() if k.startswith("ecg")), None)
        co2 = next((c for k, c in _nm.items() if k.startswith("co2")), None)
        fs = NATIVE_FS["capnobase"]
        src, chans = dcsv, names
        note = "CSV 路径（无时间列，列名为 pleth_y/ecg_y/co2_y）"
    else:
        if not os.path.exists(dmat):
            raise FileNotFoundError("找不到 CapnoBase 记录：%s" % dmat)
        m = read_mat_any(dmat)
        sig = m.get("signal", {})
        ppg = _unwrap_y(sig["pleth"]) if "pleth" in sig else None
        ecg = _unwrap_y(sig["ecg"]) if "ecg" in sig else None
        co2 = _unwrap_y(sig["co2"]) if "co2" in sig else None
        sr = m.get("param", {}).get("samplingrate", {})
        fs = float(_unwrap_y(sr["pleth"])[0]) if "pleth" in sr else NATIVE_FS["capnobase"]
        src, chans = dmat, ["pleth", "ecg", "co2"]
        note = "MATLAB v7.3(HDF5)，无时间向量，信号已交流耦合"
    # ---- HR 真值：取自 .mat 的 reference/hr/{pleth,ecg} ----
    # 坑：`*_reference.csv` **不是规整表格** —— 表头声明 9 列，数据行却是
    # 一整行空格分隔的数字（MATLAB 导出残留），按列解析会立刻错位。
    # 所以 HR 只从 .mat 读，CSV 仅作参考。
    # 另外 reference/hr/pleth/y 是**逐搏瞬时心率**（每个值对应一次搏动间隔），
    # x 是不等间隔的搏动时刻，不是 1 Hz 网格。
    hr = hr_t = None
    peaks = artifacts = None
    if not use_csv:
        ref = m.get("reference", {})
        hrnode = ref.get("hr", {})
        for key in ("pleth", "ecg"):
            if key in hrnode:
                hr = _unwrap_y(hrnode[key])
                hr_t = _unwrap_y(hrnode[key].get("x")) if isinstance(hrnode[key], dict) else None
                break
        lab = m.get("labels", {}).get("pleth", {})
        if isinstance(lab, dict):
            if "peak" in lab:
                peaks = _unwrap_y(lab["peak"])
            if "artif" in lab:
                artifacts = _unwrap_y(lab["artif"])
    if hr is not None and hr_t is not None and len(hr_t) > 1:
        d = np.diff(hr_t)
        d = d[d > 0]
        hr_fs = float(1.0 / np.median(d)) if len(d) else None
    else:
        hr_fs = None
    if ppg is None:
        raise ValueError("CapnoBase %s 里没找到 pleth 通道" % record)
    rec = _record(record, "capnobase", fs, ppg, acc=None, ecg=ecg,
                  hr_ref=hr, hr_ref_t=hr_t, hr_fs=hr_fs,
                  hr_win_s=None,        # 逐搏，不是固定窗
                  source=src, n_samples=len(ppg), channels=chans, co2=co2,
                  pleth_peaks=peaks, pleth_artifacts=artifacts,
                  units="a.u.（作者已去均值/交流耦合）",
                  note=note + "；HR 真值为逐搏瞬时心率（hr_ref_t 为搏动时刻）")
    return to_fs(rec, fs_out) if fs_out else rec


def iter_capnobase(root=None, fs_out=None):
    """遍历 CapnoBase 全部 42 例。"""
    for r in list_records("capnobase", root=root):
        yield load_capnobase(r, root=root, fs_out=fs_out)


# ==========================================================================
# 4. WESAD（15 受试者，腕部 BVP + ACC + 胸部 ECG）
# ==========================================================================
def load_wesad(subject="S2", root=None, fs_out=None, load_ecg=True):
    """加载 WESAD 一个受试者的腕部/胸部信号。

    字段来源
    --------
    ``WESAD/<S>/<S>.pkl`` 是一个 dict：

    * ``ppg``  = ``signal["wrist"]["BVP"]``，**64 Hz**
    * ``acc``  = ``signal["wrist"]["ACC"]``，**32 Hz** 三轴（x, y, z）
    * ``ecg``  = ``signal["chest"]["ECG"]``，**700 Hz**
    * ``hr_ref`` = ``None``（WESAD 只有 3 类情绪标签，**没有心率真值**；
      标签在 ``meta["label"]``，取值 0=undefined/1=baseline/2=stress/
      3=amusement/4=meditation/…，见官方 Readme）
    * 额外：``meta["eda"]``/``meta["temp"]``（腕部 4 Hz）、
      ``meta["label"]``（700 Hz）、``meta["subject"]``

    采样率不一致的处理
    ------------------
    WESAD 三路信号原生采样率不同（BVP 64 / ACC 32 / ECG 700 Hz）。为了让
    统一结构的 ``fs`` 有意义，本函数**把 ACC 与 ECG 重采样对齐到 BVP 的
    64 Hz**（``fs=64``），并在 ``meta["native_fs"]`` 里保留原始采样率。
    若你只是要 ACC 的原始 32 Hz，用 ``load_wesad_raw()`` 取 pickle 原件。

    坑
    --
    1. pickle 里的数组是 **float64 且体积巨大**：单受试者 ``.pkl`` 约
       0.8~1.1 GB，15 人合计约 13 GB。**不要一次性把 15 人全 load 进内存**，
       用 ``iter_wesad`` 逐个处理并及时释放。
    2. pickle 用 Python2 协议序列化，必须 ``pickle.load(f, encoding="latin1")``，
       否则在 Py3 下会 ``UnicodeDecodeError``。
    3. **受试者编号不连续**：只有 S2–S17 中除 S1、S12 以外的 15 人。
       按 ``range(1, 16)`` 去遍历会漏/错。
    4. WESAD 是**静息/情绪**数据，几乎没有运动伪影；它的作用是验证
       "去噪方法在静息场景不退化"，不是验证运动伪影抑制。
    5. ``.pkl`` 里还有 ``signal["wrist"]["EDA"]/["TEMP"]``（4 Hz）与
       ``signal["chest"]["ACC"]``（700 Hz，注意和腕部 ACC 同名但不同采样率），
       取通道时务必指定 ``wrist`` / ``chest``。
    """
    d = os.path.join(datasets_root(root), "WESAD", subject, subject + ".pkl")
    if not os.path.exists(d):
        avail = list_records("wesad", root=root)
        raise FileNotFoundError("找不到 WESAD 受试者文件 %s（可用 %s）" % (d, avail))
    with open(d, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    sig = obj.get("signal", {})
    wrist, chest = sig.get("wrist", {}), sig.get("chest", {})
    bvp = np.asarray(wrist["BVP"], dtype=np.float64).reshape(-1)
    acc_native = np.asarray(wrist["ACC"], dtype=np.float64)
    if acc_native.shape[0] == 3:
        acc_native = acc_native.T
    fs_bvp = NATIVE_FS["wesad_bvp"]
    acc = resample(acc_native, NATIVE_FS["wesad_acc"], fs_bvp, axis=0).T   # (3, N)
    ecg = None
    if load_ecg and "ECG" in chest:
        e = np.asarray(chest["ECG"], dtype=np.float64).reshape(-1)
        ecg = resample(e, NATIVE_FS["wesad_ecg"], fs_bvp)
    label = obj.get("label")
    rec = _record(subject, "wesad", fs_bvp, bvp[None, :], acc=acc, ecg=ecg,
                  hr_ref=None, hr_fs=None, hr_win_s=None,
                  source=d, subject=obj.get("subject", subject),
                  n_samples=len(bvp),
                  native_fs={"BVP": fs_bvp, "ACC": NATIVE_FS["wesad_acc"],
                             "ECG": NATIVE_FS["wesad_ecg"]},
                  channels=["BVP"], acc_channels=["ACC_x", "ACC_y", "ACC_z"],
                  label=None if label is None else np.asarray(label).reshape(-1),
                  eda=None if "EDA" not in wrist else np.asarray(wrist["EDA"]).reshape(-1),
                  temp=None if "TEMP" not in wrist else np.asarray(wrist["TEMP"]).reshape(-1),
                  units="BVP/ACC/ECG 原始 ADC（Empatica E4 + RespiBAN）",
                  note="ACC 32Hz→64Hz、ECG 700Hz→64Hz 已重采样对齐；无 HR 真值")
    return to_fs(rec, fs_out) if fs_out else rec


def iter_wesad(root=None, fs_out=None):
    """逐个受试者遍历 WESAD（生成器，避免 13 GB 同时驻留内存）。"""
    for s in list_records("wesad", root=root):
        yield load_wesad(s, root=root, fs_out=fs_out)


# ==========================================================================
# 华为官方数据集（筛出的中高强度包）
# ==========================================================================
HUAWEI_DIRNAME = "HUAWEI_MIDHIGH"


def _huawei_rows(path):
    """解析华为包 → (t_acc, acc(3,N), t_ppg, ppg(N,))，**按行序重建**。

    结构真相（务必保留）：文件是多传感器复用流；**PPG 每块恰好 15 行共享一个时间戳**
    （时间戳步长 0.602 s）→ 真实 PPG ≈ 25 Hz；**ACC 每时间戳约 10 行**（0.094 s）→ ≈ 105 Hz。
    历史坑：若按"时间戳去重"重建会得到 ~1.7 Hz 的假序列 → 心率全部混叠，
    这正是此前"华为 PPG 心搏带无峰"的根因（见 reports/23）。
    """
    import io as _io, zipfile as _zip
    with _zip.ZipFile(path) as z:
        name = next((n for n in z.namelist() if n.endswith(".txt")), None)
        if name is None:
            return None
        with z.open(name) as fh:
            tw = _io.TextIOWrapper(fh, encoding="utf-8", errors="ignore")
            hdr = tw.readline().rstrip("\n").split("\t")
            ix = {c: i for i, c in enumerate(hdr)}
            acc_c = [ix["ACC_X"], ix["ACC_Y"], ix["ACC_Z"]]
            pcs = sorted([c for c in hdr if c.startswith("PPG") and c[3:].isdigit()],
                         key=lambda c: int(c[3:]))[:8]
            pc = [ix[c] for c in pcs]
            # ⚠️ 个别包在文件中间还会再出现一次表头行（字段数相同）→ 直接 float() 会抛
            # "could not convert string to float: 'ACC_TIME'"。这里逐行尝试解析，非数值行丢弃。
            def _parse(line):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != len(hdr):
                    return None
                try:
                    return [float(x) for x in parts]
                except ValueError:
                    return None
            _rows = [r for r in (_parse(l) for l in tw) if r is not None]
            A = np.array(_rows, dtype=np.float64)
    if len(A) < 500:
        return None
    t = A[:, ix["ACC_TIME"]] / 1000.0
    an = np.abs(A[:, acc_c]).sum(axis=1) > 0
    pn = np.abs(A[:, pc]).sum(axis=1) > 0
    return dict(t=t, acc=A[:, acc_c], ppg=A[:, pc], an=an, pn=pn)


def huawei_segments(path, min_seg_s=60.0, gap_s=2.0):
    """把包切成"ACC 与 PPG 同时在采"的段，返回 [(t0, dur_s, acc(3,n), fs_acc, ppg(n,), fs_ppg)]。"""
    D = _huawei_rows(path)
    if D is None:
        return []
    t, ok = D["t"], (D["an"] | D["pn"])
    tt = np.unique(t[ok & (t > 0)])
    if len(tt) < 10:
        return []
    spans = np.split(np.arange(len(tt)), np.nonzero(np.diff(tt) > gap_s)[0] + 1)
    out = []
    for sp in spans:
        t0, t1 = tt[sp[0]], tt[sp[-1]]
        dur = t1 - t0
        if dur < min_seg_s:
            continue
        ma = D["an"] & (t >= t0) & (t <= t1)
        mp = D["pn"] & (t >= t0) & (t <= t1)
        if ma.sum() < 400 or mp.sum() < 400:
            continue
        out.append(dict(t0=float(t0), dur_s=float(dur), fs_acc=float(ma.sum() / dur),
                        fs_ppg=float(mp.sum() / dur),
                        acc=D["acc"][ma].T.copy(), ppg=D["ppg"][mp].T.copy()))
    return out


def load_huawei(record, root=None, fs_out=None, seg="best", min_seg_s=60.0, fs_ppg=25.0,
                fs_acc=100.0):
    """读一个华为包 → 记录 dict（无 HR 真值，用于**无真值验证**）。

    seg="best" 选运动最强且最长的段；也可给整数下标。
    返回的 ppg/acc 已重采样到 `fs_out`（默认与 PPG 重建率一致或 100 Hz）。
    """
    base = os.path.join(datasets_root(root), HUAWEI_DIRNAME)
    cand = record if os.path.isabs(record) else os.path.join(base, record)
    if os.path.isdir(cand):
        zs = sorted(glob.glob(os.path.join(cand, "*.zip")))
        if not zs:
            raise FileNotFoundError("目录下无 zip：%s" % cand)
        cand = zs[0]
    segs = huawei_segments(cand, min_seg_s=min_seg_s)
    if not segs:
        raise ValueError("包内没有 ≥%.0f s 的 ACC∩PPG 同时段：%s" % (min_seg_s, cand))
    if seg == "best":
        segs.sort(key=lambda g: -(g["dur_s"] * float(np.sqrt(np.mean((g["acc"] - g["acc"].mean(axis=1, keepdims=True)) ** 2)))))
        g = segs[0]
    else:
        g = segs[int(seg)]
    fs_p = g["fs_ppg"]
    # g["ppg"] 形状 (8, N)：按**通道**（对时间求和，axis=1）挑最强通道
    ppg = g["ppg"][int(np.argmax(np.abs(g["ppg"]).sum(axis=1)))]
    acc = g["acc"]
    tgt = float(fs_out or fs_p)
    ppg_r = resample(ppg, fs_p, tgt)
    acc_r = resample(acc.T, g["fs_acc"], tgt, axis=0).T
    n = min(len(ppg_r), acc_r.shape[1])
    rec = _record(os.path.basename(cand), "huawei", tgt, ppg_r[None, :n], acc=acc_r[:, :n],
                  ecg=None, hr_ref=None, hr_fs=None, hr_win_s=None,
                  source=cand, n_samples=n, native_fs={"PPG": round(fs_p, 2), "ACC": round(g["fs_acc"], 2)},
                  channels=["PPG1"], acc_channels=["ACC_x", "ACC_y", "ACC_z"],
                  units="PPG/ACC 原始 ADC 计数（DC 耦合，AC/DC 4.8–51.5%）",
                  note="段起点 t0=%.0f；段长 %.0f s；**无心率真值**（平台只提供 acc+ppg）；"
                       "PPG≈25 Hz（每块 15 行共享时间戳）、ACC≈105 Hz" % (g["t0"], g["dur_s"]),
                  seg_t0=g["t0"], seg_dur_s=g["dur_s"], n_segments=len(segs))
    return to_fs(rec, fs_out) if fs_out else rec


def iter_huawei(root=None, fs_out=None, min_seg_s=60.0):
    """逐个包遍历筛出的华为中高强度数据。"""
    base = os.path.join(datasets_root(root), HUAWEI_DIRNAME)
    for f in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if f.endswith(".zip"):
            try:
                yield load_huawei(f, root=root, fs_out=fs_out, min_seg_s=min_seg_s)
            except Exception:
                continue


# ==========================================================================
# WRIST —— PhysioNet "Wrist PPG During Exercise"（腕部 PPG + ACC + GYRO + 胸带 ECG）
# ==========================================================================
WRIST_DIRNAME = "WRIST"
WRIST_ACTIVITY = {"walk": "walk", "run": "run",
                  "low_resistance_bike": "bike_low", "high_resistance_bike": "bike_high"}


def _rpeak_hr_windows(rpeaks_s, n_samples, fs, win_s=8.0, hop_s=2.0):
    """由 R 峰时刻（秒）算逐窗 HR（与流水线 8 s 窗 / 2 s 步进口径一致）。"""
    if rpeaks_s is None or len(rpeaks_s) < 3:
        return None, None
    win, hop = int(win_s * fs), int(hop_s * fs)
    n_win = int(max(0, (n_samples - win) // hop) + 1)
    hr = np.full(n_win, np.nan)
    t = np.full(n_win, np.nan)
    for i in range(n_win):
        s, e = i * hop / fs, (i * hop + win) / fs
        q = rpeaks_s[(rpeaks_s >= s) & (rpeaks_s < e)]
        if len(q) >= 2:
            rr = np.diff(q)
            rr = rr[(rr > 0.3) & (rr < 1.5)]
            if len(rr):
                hr[i] = 60.0 / np.median(rr)
        t[i] = s + win_s / 2.0
    return hr, t


def _fill_nan(x):
    """线性插值填补 NaN（WFDB 记录中设备丢包处为 NaN；实测 WRIST 有 0.07–0.18%）。

    ⚠️ 不做这一步的后果：RLS 输出全 NaN → PSD 全 NaN → Viterbi 恒定返回带内第一个 bin
    （实测估计恒为 43.1 bpm，与输入无关，MAE ~58）——这是本会话真实踩到的坑。
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        bad = ~np.isfinite(x)
        if bad.any():
            idx = np.arange(len(x))
            x = x.copy()
            good = ~bad
            if good.sum() >= 2:
                x[bad] = np.interp(idx[bad], idx[good], x[good])
    else:
        x = np.stack([_fill_nan(r) for r in x])
    return x


def load_wrist(record, root=None, fs_out=100.0, acc_kind="low_noise"):
    """读一条 WRIST 记录（WFDB）→ 统一 record 结构（重采样到 fs_out）。

    - PPG：`wrist_ppg`；ACC：`wrist_low_noise_accelerometer_{x,y,z}`；GYRO：`wrist_gyro_{x,y,z}`
    - **真值**：`.atr` 里**人工标注的 ECG R 峰时刻** → 逐窗 HR（8 s/2 s），故 `ecg=None`、用 `hr_ref`
    - 附加 meta：`activity`（walk/run/bike_low/bike_high）、`subject`
    """
    import wfdb
    base = os.path.join(datasets_root(root), WRIST_DIRNAME)
    path = os.path.join(base, os.path.basename(record).replace(".hea", "").replace(".dat", ""))
    recw = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    fs_native = float(recw.fs)
    sig = np.asarray(recw.p_signal, dtype=np.float64)
    nm = [str(c).lower() for c in recw.sig_name]
    ix = {n: i for i, n in enumerate(nm)}
    if "wrist_ppg" not in ix:
        return None
    ppg = _fill_nan(sig[:, ix["wrist_ppg"]])
    pre = ("wrist_low_noise_accelerometer_" if acc_kind == "low_noise"
           else "wrist_wide_range_accelerometer_")
    acc = _fill_nan(np.stack([sig[:, ix[pre + ax]] for ax in "xyz"]))
    gyro = (_fill_nan(np.stack([sig[:, ix["wrist_gyro_" + ax]] for ax in "xyz"]))
            if "wrist_gyro_x" in ix else None)
    rpeaks = np.asarray(ann.sample, dtype=np.float64) / fs_native          # 秒
    ppg_r = resample(ppg, fs_native, fs_out)
    acc_r = resample(acc.T, fs_native, fs_out, axis=0).T
    gyro_r = resample(gyro.T, fs_native, fs_out, axis=0).T if gyro is not None else None
    n = len(ppg_r)
    hr_ref, hr_ref_t = _rpeak_hr_windows(rpeaks, n, fs_out)
    act = os.path.basename(path)
    act_key = next((k for k in WRIST_ACTIVITY if act.endswith(k)), "?")
    subj = act.split("_")[0]
    return _record(os.path.basename(path), "wrist", fs_out, ppg_r[None, :],
                   acc=acc_r[:, :n], ecg=None, hr_ref=hr_ref, hr_ref_t=hr_ref_t,
                   hr_fs=0.5, hr_win_s=8.0, source=path,
                   activity=WRIST_ACTIVITY.get(act_key, act_key), subject=subj,
                   n_samples=n, native_fs=fs_native,
                   channels=["wrist_ppg"],
                   acc_channels=[pre + a for a in "xyz"],
                   gyro=(gyro_r[:, :n] if gyro_r is not None else None),
                   units="物理单位（PPG/ACC=ms^-2，GYRO=degs^-1，ECG=mV）",
                   note="真值=人工标注 ECG R 峰（金标准）；256 Hz 原始，重采样到 %g Hz" % fs_out)


def iter_wrist(root=None, fs_out=100.0):
    base = os.path.join(datasets_root(root), WRIST_DIRNAME)
    rf = os.path.join(base, "RECORDS")
    names = ([l.strip() for l in open(rf) if l.strip()] if os.path.exists(rf)
             else sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(base, "*.hea"))))
    for nm in names:
        try:
            r = load_wrist(nm, root=root, fs_out=fs_out)
        except Exception:
            r = None
        if r is not None:
            yield r

# ==========================================================================
# PPG-DaLiA —— 腕部 E4（PPG 64 Hz + ACC 32 Hz）+ 胸带 ECG 700 Hz（含 R 峰金标准）
# 数据来源：Zenodo 12793711 的 **MATLAB 格式版**（按活动拆文件，367 MB；UCI 原始 2.7 GB 单连接仅 ~425 KB/s）
# ==========================================================================
DALIA_DIRNAME = "PPG_DALIA"


def _dalia_files(root=None):
    import glob
    base = os.path.join(datasets_root(root), DALIA_DIRNAME)
    return sorted(glob.glob(os.path.join(base, "ppg_dalia_*_data.mat")))


def _dalia_activity(path):
    n = os.path.basename(path)
    if n.startswith("ppg_dalia_") and n.endswith("_data.mat"):
        return n[len("ppg_dalia_"):-len("_data.mat")]
    return n


def load_dalia(record, root=None, fs_out=100.0, subj=0):
    """读 PPGL-DaLiA（MATLAB 版）某活动文件里的第 `subj` 名受试者。

    - PPG：`ppg.v`（64 Hz）；ACC：`acc_ppg_site.v`（32 Hz，**与 PPG 同侧**，正是 RLS 需要的运动参考）
    - **真值**：`ecg.rpeaks`（700 Hz 采样序号，金标准 R 峰）→ 逐窗 HR（8 s/2 s），故 `ecg=None` 用 `hr_ref`
    - meta：`subject`（受试者名）、`activity`、`age/weight/height/gender` 等
    """
    import scipy.io as sio
    base = os.path.join(datasets_root(root), DALIA_DIRNAME)
    path = record if os.path.isabs(str(record)) else os.path.join(base, str(record))
    if not path.endswith(".mat"):
        cand = _dalia_files(root)
        cand = [c for c in cand if _dalia_activity(c) == str(record)]
        if not cand:
            return None
        path = cand[0]
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    arr = d["data"]
    if subj >= len(arr):
        return None
    s = arr[subj]
    fs_ppg = float(s.ppg.fs); fs_acc = float(s.acc_ppg_site.fs); fs_ecg = float(s.ecg.fs)
    ppg = _fill_nan(np.asarray(s.ppg.v, dtype=np.float64).reshape(-1))
    acc1 = _fill_nan(np.asarray(s.acc_ppg_site.v, dtype=np.float64).reshape(-1))
    rpk = np.atleast_1d(np.asarray(s.ecg.rpeaks, dtype=np.float64)).reshape(-1) / fs_ecg
    ppg_r = resample(ppg, fs_ppg, fs_out)
    acc_r = resample(acc1, fs_acc, fs_out)
    n = min(len(ppg_r), len(acc_r))
    hr_ref, hr_ref_t = _rpeak_hr_windows(rpk, n, fs_out)
    act = _dalia_activity(path)
    f = s.fix
    name = "%s_%s" % (act, str(getattr(f, "subj_name", "S%d" % (subj + 1))).strip())
    return _record(name, "dalia", fs_out, ppg_r[None, :n],
                   acc=acc_r[None, :n], ecg=None, hr_ref=hr_ref, hr_ref_t=hr_ref_t,
                   hr_fs=0.5, hr_win_s=8.0, source=path,
                   subject=str(getattr(f, "subj_name", subj + 1)).strip(), activity=act,
                   age=getattr(f, "age", None), gender=str(getattr(f, "gender", "")),
                   exercise_freq=getattr(f, "exercise_freq", None),
                   n_samples=n, native_fs={"PPG": fs_ppg, "ACC": fs_acc, "ECG": fs_ecg},
                   channels=["E4-BVP"], acc_channels=["E4-ACC"],
                   units="PPG=原始 ADC；ACC=1/64 g",
                   note="真值=胸带 ECG 的 rpeaks（金标准）；单轴 ACC（与 PPG 同侧）")


def iter_dalia(root=None, fs_out=100.0, subj_max=None):
    for path in _dalia_files(root):
        try:
            d = None
            import scipy.io as sio
            d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
            n_sub = len(d)
        except Exception:
            continue
        if subj_max:
            n_sub = min(n_sub, subj_max)
        for i in range(n_sub):
            try:
                r = load_dalia(path, root=root, fs_out=fs_out, subj=i)
            except Exception:
                r = None
            if r is not None:
                yield r


LOADERS = {"troika": load_troika, "bidmc": load_bidmc,
           "capnobase": load_capnobase, "wesad": load_wesad, "huawei": load_huawei,
           "wrist": load_wrist, "dalia": load_dalia}


# ==========================================================================
# 命令行自检
# ==========================================================================
def _selfcheck(dataset, root=None, limit=None):
    """打印某数据集的实际记录数 / 采样率 / 通道名 / 形状，供人工对照规格。"""
    root = datasets_root(root)
    print("\n" + "=" * 76)
    print("== %s   root=%s" % (dataset.upper(), root))
    print("=" * 76)
    if dataset == "troika":
        for split in ("train", "test"):
            ids = list_records("troika", split=split, root=root)
            print("  split=%-5s 记录数=%d  样例=%s" % (split, len(ids), ids[:3]))
        ids = list_records("troika", root=root)
        for rid in (ids[:limit] if limit else ids[:2]):
            r = load_troika(rid, root=root)
            print("  %-16s ppg=%s acc=%s ecg=%s fs=%.1f"
                  % (r["name"], r["ppg"].shape,
                     None if r["acc"] is None else r["acc"].shape,
                     None if r["ecg"] is None else r["ecg"].shape, r["fs"]))
            print("                   hr_ref=%s hr_fs=%s hr_win_s=%s hr_ref_t=0..%.0fs"
                  % (None if r["hr_ref"] is None else r["hr_ref"].shape,
                     r["hr_fs"], r["hr_win_s"],
                     -1 if r["hr_ref_t"] is None else r["hr_ref_t"][-1]))
            print("                   通道=%s" % r["meta"]["channels"])
        # 测试集窗总数（官方协议：8 s 窗 / 2 s 步进）
        tot = 0
        for rid in list_records("troika", split="test", root=root):
            r = load_troika(rid, split="test", root=root)
            tot += 0 if r["hr_ref"] is None else len(r["hr_ref"])
        print("  官方测试集 8 s 窗总数 = %d   （规格：1328）" % tot)
    elif dataset == "bidmc":
        ids = list_records("bidmc", root=root)
        print("  记录数=%d  样例=%s" % (len(ids), ids[:4]))
        for rid in (ids[:limit] if limit else ids[:1]):
            r = load_bidmc(rid, root=root)
            print("  %-10s ppg=%s ecg=%s acc=%s fs=%.1f  时长=%.1fs"
                  % (r["name"], r["ppg"].shape,
                     None if r["ecg"] is None else r["ecg"].shape, r["acc"], r["fs"],
                     r["ppg"].shape[-1] / r["fs"]))
            print("             hr_ref=%s hr_fs=%s hr_ref_t=%s"
                  % (None if r["hr_ref"] is None else r["hr_ref"].shape, r["hr_fs"],
                     None if r["hr_ref_t"] is None
                     else "0..%.0fs" % r["hr_ref_t"][-1]))
            print("             通道=%s" % r["meta"]["channels"])
    elif dataset == "capnobase":
        ids = list_records("capnobase", root=root)
        print("  记录数=%d  样例=%s" % (len(ids), ids[:4]))
        for rid in (ids[:limit] if limit else ids[:1]):
            r = load_capnobase(rid, root=root)
            print("  %-12s ppg%s ecg%s acc=%s fs=%.1f  hr_ref=%s hr_fs=%s"
                  % (r["name"], r["ppg"].shape, None if r["ecg"] is None else r["ecg"].shape,
                     r["acc"], r["fs"],
                     None if r["hr_ref"] is None else r["hr_ref"].shape, r["hr_fs"]))
            if r["hr_ref_t"] is not None:
                d = np.diff(r["hr_ref_t"])
                print("               hr_ref_t 范围 %.2f~%.2f s，中位搏动间隔 %.3f s（逐搏不等间隔）"
                      % (r["hr_ref_t"][0], r["hr_ref_t"][-1], float(np.median(d[d > 0]))))
            print("               通道=%s  co2=%s  pleth_peaks=%s"
                  % (r["meta"]["channels"],
                     None if r["meta"].get("co2") is None else r["meta"]["co2"].shape,
                     None if r["meta"].get("pleth_peaks") is None
                     else r["meta"]["pleth_peaks"].shape))
    elif dataset == "wesad":
        ids = list_records("wesad", root=root)
        print("  受试者数=%d  %s" % (len(ids), ids))
        for rid in (ids[:limit] if limit else ids[:1]):
            r = load_wesad(rid, root=root)
            print("  %-5s ppg%s acc%s ecg%s fs=%.1f hr_ref=%s"
                  % (r["name"], r["ppg"].shape, None if r["acc"] is None else r["acc"].shape,
                     None if r["ecg"] is None else r["ecg"].shape, r["fs"], r["hr_ref"]))
            print("        原生采样率=%s  重采样后统一 fs=%.1f" % (r["meta"]["native_fs"], r["fs"]))
            print("        标签长度=%s" % (None if r["meta"]["label"] is None
                                          else r["meta"]["label"].shape,))
    else:
        raise ValueError(dataset)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = None
    if "--root" in argv:
        i = argv.index("--root")
        root = argv[i + 1]
        del argv[i:i + 2]
    limit = None
    if "--limit" in argv:
        i = argv.index("--limit")
        limit = int(argv[i + 1])
        del argv[i:i + 2]
    todo = [a for a in argv if not a.startswith("-")] or list(DATASETS)
    for ds in todo:
        try:
            _selfcheck(ds, root=root, limit=limit)
        except Exception as e:
            import traceback
            print("  [FAIL] %s: %s" % (ds, e))
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())

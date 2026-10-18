#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TROIKA 域经典心率 baseline：SpaMA 与 JOSS，并与本方案三方对比（报告 `reports/35`）。

口径（与主实验严格一致）
------------------------
* 数据：``ext_loaders.iter_troika(split="test", fs_out=100)``，10 条记录 / **1328 个 8 s 窗**（2 s 步进，第 i 窗 = [2i, 2i+8) 秒）；
* 真值：``rec["hr_ref"]``（官方 ``BPM0``）；PPG 取**第 0 路**（两个 baseline 原文都只用单路 PPG）；
* 指标：窗级池化 **HR MAE (bpm)**、窗级中位绝对误差、**>5 bpm 占比**，另给逐记录 MAE。

三个方法
--------
1. ``spama`` —— Salehizadeh/Dao/Bolkhovsky/Cho/Mendelson/Chon, *A Novel Time-Varying Spectral
   Filtering Algorithm ...*, Sensors 16(1):10, 2016（即文献里通称的 **SpaMA**，spectral matrix
   based method）。重采样到 20 Hz → 8 s/2 s 时频谱（periodogram，df=0.001 Hz）→ 0.5–3 Hz 内取
   PPG 谱前 3 峰 → 与加速度谱主峰比对剔除运动峰（只判第 1/2 峰）→ **峰追踪 + 峰值保持**：
   在存活峰里取"离上一窗最近"的一个，若跳变 >10 bpm 则保持上一窗值。
2. ``joss`` —— Zhang, *...via Joint Sparse Spectrum Reconstruction*, IEEE TBME 62(8):1902-1910,
   2015。PPG + 3 轴 ACC 组成 L=4 的 MMV 模型 ``Y=ΦX+V``（Φ = N=1024 冗余 DFT 基，25 Hz），
   **Regularized M-FOCUSS**（p=0.8, λ=1e-10, 4 次迭代）联合恢复行稀疏谱 → 谱减
   （逐 bin 减 3 轴 ACC 最大值，再置零 < pmax/4）→ 初始化/峰选择/峰验证/峰发现四段跟踪。
   另给 **ISTA(ℓ2,1)** 求解器变体做交叉验证。
3. ``ours`` —— 本方案 `reports/19` 定稿配方（RLS(3 轴 ACC, α=1) → PSD → Viterbi；
   记录级 ``snr_peak`` 选路 + 裕度门控 1.2 + 窗级可靠性加权 ``w=s2^0.5``），
   直接调用 `scripts/tools/dual_lead_fuse.py` 的同一函数复算（用于校验配方可复现）。

与原文的差异（详见 `reports/35` §2）：两处**必要**偏离——
(a) 原文的"跳变超限就保持上一窗"在换向/丢失锁定时会永久卡死，我们加了**连续 hold 后的全带重搜索**
    （原文 JOSS 的"峰发现"只在预测点 ±Δ3 内搜，出不去）；
(b) 原文用 Marmarelis 的 Smoother 做趋势预测，不可得，用**加权线性外推**近似。
其余（分帧、频带、峰数、阈值、M-FOCUSS 参数、谱减、验证阈值）按原文。

防泄漏
------
SpaMA 的自由度（ACC 匹配容差 tol / ACC 峰取法 / 是否做运动峰剔除 / 逃逸延迟 H）与
JOSS 的逃逸延迟 H **只在 TROIKA 12 条训练记录上选**，官方 10 条测试记录只评估一次；
另给"测试集内**留一记录**（LORO）选容差"和**全网格**结果作敏感性证据（不挑最好看的）。
随机性：仅配对自助检验用固定种子 ``seed=0``。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/troika_baselines.py --out outputs/troika_baselines
    $PY scripts/tools/troika_baselines.py --records TEST_S03_T02 --out outputs/tmp_check
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "modules"))
from _common import paths as P                                # noqa: E402
from _common import parallel as PAR                           # noqa: E402

FS_EVAL = 100.0
BAND = (0.7, 3.0)          # 本方案选峰带（reports/19 标定值）
SPAMA_FS = 20.0            # 原文：所有数据集降到 20 Hz
JOSS_FS = 25.0             # 原文：125 Hz → 25 Hz
WIN_S, HOP_S = 8.0, 2.0
JOSS_N = 1024              # 原文的谱网格数（25 Hz 下 1.46 bpm/bin）
JOSS_D1, JOSS_D2, JOSS_D3 = 15, 25, 30
JOSS_JUMP = 12.0           # 原文峰验证阈值 (bpm)
JOSS_KURT = 10.0           # 原文初始化峰度阈值

# SpaMA 待选超参（只在训练记录上选；tol=0 表示不做运动峰剔除）
SPAMA_GRID = ([dict(tol=0.0, acc_mode="any", use_ma=False, H=2)]
              + [dict(tol=t, acc_mode=m, use_ma=True, H=h)
                 for t in (0.10, 0.20, 0.30) for m in ("any", "mag") for h in (1, 2, 3)])
JOSS_H_GRID = [1, 2, 3, 4]

_MOD_CACHE = {}


def load_module(name, path):
    key = (name, path)
    if key not in _MOD_CACHE:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD_CACHE[key] = mod
    return _MOD_CACHE[key]


# ===========================================================================
# 通用工具
# ===========================================================================
def resample_to(x, fs_in, fs_out, axis=-1):
    from scipy.signal import resample_poly
    from fractions import Fraction
    if abs(fs_out - fs_in) < 1e-9:
        return np.asarray(x, dtype=np.float64)
    f = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64), f.numerator, f.denominator, axis=axis)


def bandpass(x, fs, lo=0.4, hi=4.0, order=2):
    """JOSS 原文：2 阶 Butterworth 0.4–4 Hz（原文未说明相位处理，这里用 filtfilt 零相位）。"""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [lo / (fs / 2), min(hi, fs / 2 * 0.99) / (fs / 2)], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64), axis=-1)


def _psd_fine(x, fs, nfft):
    """Hanning + 零填充 periodogram；返回 (f, P)（单边功率）。"""
    n = x.shape[-1]
    w = np.hanning(n)
    X = np.fft.rfft(x * w, n=nfft, axis=-1)
    P = (np.abs(X) ** 2) / (fs * np.sum(w ** 2)) * 2.0
    P[..., 0] /= 2.0
    return np.fft.rfftfreq(nfft, d=1.0 / fs), P


def local_peaks(P, f, lo, hi, max_n=None):
    """带内局部极大值，按功率降序；返回 [(freq, power), ...]。"""
    m = (f >= lo) & (f <= hi)
    idx = np.where(m)[0]
    if len(idx) < 3:
        return []
    out = []
    for j in idx:
        if P[j] <= 0:
            continue
        if (j == 0 or P[j] >= P[j - 1]) and (j == len(P) - 1 or P[j] >= P[j + 1]):
            out.append((float(f[j]), float(P[j])))
    out.sort(key=lambda z: -z[1])
    return out[:max_n] if max_n else out


def kurtosis(x):
    """Fisher 峰度（JOSS 原文初始化判据）。"""
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s <= 0 or len(x) < 4:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 4) - 3.0)


def _trend(hist, K=10):
    """加权线性外推（近似原文的 Smoother 趋势预测）。"""
    h = np.asarray(hist[-K:], dtype=np.float64)
    h = h[np.isfinite(h)]
    if len(h) == 0:
        return np.nan
    if len(h) < 3:
        return float(h[-1])
    x = np.arange(len(h), dtype=np.float64)
    w = np.linspace(0.3, 1.0, len(h))
    A = np.stack([x, np.ones_like(x)], axis=1)
    c = np.linalg.lstsq(A * w[:, None], h * w, rcond=None)[0]
    return float(c[0] * len(h) + c[1])


# ===========================================================================
# 1) SpaMA 前端（时频谱 + 运动峰剔除）与追踪器
# ===========================================================================
def spama_candidates(ppg, acc, fs_in, n_win, acc_band=(0.8, 4.0), df=0.001):
    """每条记录算一次：逐窗 PPG 谱的**全部**局部峰（0.5–3 Hz）与 ACC 主峰频率。

    返回 list[dict(fr, pw, acs_any, acs_mag)]：
      * ``acs_any`` = 三个轴**各自**在 0.8–4 Hz 的主峰（原文"the first peak of the
        accelerometers' spectra"的一种读法）；
      * ``acs_mag`` = 三轴功率和谱的主峰（另一种读法）。
    """
    fs = SPAMA_FS
    p20 = resample_to(ppg, fs_in, fs)
    a20 = resample_to(acc, fs_in, fs, axis=-1)
    nfft = int(round(fs / df))
    out = []
    for t in range(n_win):
        s, e = int(t * HOP_S * fs), int(t * HOP_S * fs) + int(WIN_S * fs)
        f, P = _psd_fine(p20[s:e], fs, nfft)
        pp = local_peaks(P, f, 0.5, 3.0)
        any_p, mag_p = [], None
        for k in range(a20.shape[0]):
            _, Pa = _psd_fine(a20[k, s:e], fs, nfft)
            any_p += [q[0] for q in local_peaks(Pa, f, acc_band[0], acc_band[1], max_n=1)]
            mag_p = Pa if mag_p is None else mag_p + Pa
        mag = [q[0] for q in local_peaks(mag_p, f, acc_band[0], acc_band[1], max_n=1)]
        out.append(dict(fr=np.array([p[0] for p in pp]) * 60.0,
                        pw=np.array([p[1] for p in pp]),
                        acs_any=np.array(any_p) * 60.0, acs_mag=np.array(mag) * 60.0))
    return out


def spama_band_spectra(ppg, acc, fs_in, n_win, nfft=640, band=(0.4, 4.0)):
    """SpaMA 前端（20 Hz 原始 PPG 时频谱）的带限谱矩阵，用于"换追踪器"对照实验。"""
    fs = SPAMA_FS
    p20 = resample_to(ppg, fs_in, fs)
    f, _ = _psd_fine(p20[:int(WIN_S * fs)], fs, nfft)
    m = (f >= band[0]) & (f <= band[1])
    P = np.zeros((n_win, int(m.sum())))
    for t in range(n_win):
        s, e = int(t * HOP_S * fs), int(t * HOP_S * fs) + int(WIN_S * fs)
        _, Pw = _psd_fine(p20[s:e], fs, nfft)
        P[t] = Pw[m]
    return f[m], P


def spama_track(cands, n_win, tol=0.2, acc_mode="any", use_ma=True, H=2,
                max_jump=10.0, top_k=3):
    """原文 Stage 3/4 的峰追踪 + 峰值保持（+ 连续 hold 后的全带重搜索，见模块 docstring）。

    返回 (hr[n_win], info)
    """
    hr = np.full(n_win, np.nan)
    prev, hold, n_ma, n_esc = None, 0, 0, 0
    for t in range(n_win):
        c = cands[t]
        fr, pw = c["fr"], c["pw"]
        acs = c["acs_any"] if acc_mode == "any" else c["acs_mag"]
        if len(fr) == 0:
            hr[t] = prev if prev is not None else np.nan
            continue
        # --- Stage 3：只判前 2 峰，与 ACC 主峰比对（acc_mode=mag 时 ACC 已合成单峰）---
        alive = []
        for i in range(len(fr)):
            is_ma = False
            if use_ma and i < 2:
                is_ma = bool(np.any(np.abs(acs - fr[i]) <= tol * 60.0))
                if is_ma:
                    n_ma += 1
            if not is_ma:
                alive.append((fr[i], pw[i]))
        if not alive:
            alive = [(fr[0], pw[0])]
        # --- Stage 4 ---
        if prev is None:
            val = alive[0][0]                      # 初始窗：最高峰
        else:
            inr = [z for z in alive[:top_k] if abs(z[0] - prev) <= max_jump]
            if inr:
                val = min(inr, key=lambda z: abs(z[0] - prev))[0]
                hold = 0
            else:
                hold += 1
                if hold < H:
                    val = prev                     # 峰值保持
                else:
                    val = alive[0][0]              # 全带重搜索（取存活峰里功率最高者）
                    hold = 0
                    n_esc += 1
        hr[t] = val
        prev = val
    return hr, dict(n_ma=n_ma, n_escape=n_esc)


# ===========================================================================
# 2) JOSS 前端（MMV + Regularized M-FOCUSS + 谱减）与四段追踪器
# ===========================================================================
def _mfocuss(Phi, Y, p=0.8, lam=1e-10, iters=4):
    """Regularized M-FOCUSS：``X_{k+1} = W_k Φᴴ(ΦW_kΦᴴ+λI)^{-1}Y``，``W_k=diag(||X_k(i,:)||₂^{2-p})``。

    实现约定可用 ``--selfcheck-solver`` 复现（M=200/N=1024/L=4 的高相干冗余 DFT 基，
    p=0.8 + 4 次迭代下 3/3 命中真支撑）；详见 `reports/35` §2 差异 d。
    """
    PhH = Phi.conj().T
    X = PhH @ Y
    for _ in range(iters):
        rn = np.linalg.norm(X, axis=1)
        w = np.maximum(rn, 1e-12) ** (2.0 - p)
        B = Phi * w[None, :]                       # ΦW
        G = B @ PhH                                # ΦWΦᴴ
        G[np.diag_indices_from(G)] += lam
        X = w[:, None] * (PhH @ np.linalg.solve(G, Y))
    return X


def _ista_l21(Phi, Y, lam_ratio=0.1, iters=200):
    """ℓ2,1 行稀疏的 ISTA（Task 允许的 ℓ1 近似变体）；λ = lam_ratio·max_i||(ΦᴴY)_i||₂。"""
    PhH = Phi.conj().T
    L = float(np.linalg.norm(Phi, 2) ** 2)
    lam = lam_ratio * float(np.max(np.linalg.norm(PhH @ Y, axis=1)))
    X = np.zeros((Phi.shape[1], Y.shape[1]), dtype=np.complex128)
    for _ in range(iters):
        X = X - (PhH @ (Phi @ X - Y)) / L
        rn = np.linalg.norm(X, axis=1)
        X = X * np.maximum(0.0, 1.0 - (lam / L) / np.maximum(rn, 1e-12))[:, None]
    return X


def joss_spectra(ppg, acc, fs_in, n_win, solver="mfocuss", use_sub=True):
    """每条记录算一次：返回 (spec[n_win, N], fgrid[N], info)。"""
    fs = JOSS_FS
    pp = bandpass(resample_to(ppg, fs_in, fs), fs, 0.4, 4.0, order=2)
    aa = bandpass(resample_to(acc, fs_in, fs, axis=-1), fs, 0.4, 4.0, order=2)
    M, N = int(round(WIN_S * fs)), JOSS_N
    hop = int(round(HOP_S * fs))
    m = np.arange(M)[:, None]
    nn = np.arange(N)[None, :]
    Phi = np.exp(1j * 2 * np.pi * m * nn / N)      # M×N 冗余 DFT 基
    fgrid = nn.ravel() * fs / N
    spec = np.zeros((n_win, N))
    n_solve = 0
    for t in range(n_win):
        s = t * hop
        seg = np.stack([pp[s:s + M]] + [aa[k, s:s + M] for k in range(aa.shape[0])], axis=1)
        seg = seg - seg.mean(axis=0, keepdims=True)
        sd = seg.std(axis=0, keepdims=True)
        sd[sd <= 0] = 1.0
        Y = seg / sd                                # 各列等能量（原文 II-C 要求）
        X = _mfocuss(Phi, Y) if solver == "mfocuss" else _ista_l21(Phi, Y)
        n_solve += 1
        S = np.abs(X) ** 2
        sp = S[:, 0].copy()
        if use_sub:
            sp = sp - S[:, 1:].max(axis=1)          # 逐 bin 减 3 轴 ACC 谱最大值
            pmax = float(S[:, 0].max())
            sp[sp < pmax / 4.0] = 0.0               # 置零 < pmax/4
        else:
            sp[sp < 0] = 0.0
        spec[t] = sp
    return spec, fgrid, dict(n_solve=n_solve, fs=fs, df_bpm=float(fs / N * 60.0),
                             nfft_bins=N)


def _joss_peaks(sp, fgrid, band=(0.5, 4.0)):
    mb = (fgrid >= band[0]) & (fgrid <= band[1])
    idx = np.where(mb)[0]
    return [(int(j), float(sp[j])) for j in idx
            if sp[j] > 0 and (j == 0 or sp[j] >= sp[j - 1]) and (j == len(sp) - 1 or sp[j] >= sp[j + 1])]


def joss_track(spec, fgrid, n_win, H=3, global_escape=False, use_kurt=True,
               d1=JOSS_D1, d2=JOSS_D2, d3=JOSS_D3, jump=JOSS_JUMP, kurt_th=JOSS_KURT,
               init_band=(0.8, 2.5)):
    """原文 II-D 四段跟踪：初始化（峰度门控）→ 峰选择(R1→R2) → 峰验证(±12 bpm) → 峰发现。"""
    dfb = float(fgrid[1] * 60.0)
    ib = (fgrid >= init_band[0]) & (fgrid <= init_band[1])
    hr = np.full(n_win, np.nan)
    prev, hold = None, 0
    n_blank, n_disc = 0, 0
    for t in range(n_win):
        sp = spec[t]
        peaks = _joss_peaks(sp, fgrid)
        if prev is None:
            if use_kurt and kurtosis(sp[ib]) <= kurt_th:   # 原文：非初始化阶段不输出 HR
                n_blank += 1
                continue
            prev = int(np.where(ib)[0][int(np.argmax(sp[ib]))])
            hr[t] = prev * dfb
            continue
        r1 = [(j, p) for j, p in peaks if abs(j - prev) <= d1]
        if r1:
            dmin = min(abs(j - prev) for j, _ in r1)
            tie = [j for j, _ in r1 if abs(j - prev) == dmin]
            cur = tie[0] if len(tie) == 1 else min(tie, key=lambda j: abs(j - prev))
        else:
            r2 = [(j, p) for j, p in peaks if abs(j - prev) <= d2]
            cur = max(r2, key=lambda z: z[1])[0] if r2 else None
        if cur is None:
            cur = prev
        elif abs(cur * dfb - prev * dfb) > jump:           # 峰验证
            cur = prev
        if cur == prev:
            hold += 1
        else:
            hold = 0
        if hold >= H and (global_escape or d3 > 0):        # 峰发现
            if global_escape:
                cand = [(j, p) for j, p in peaks if abs(j - prev) > 2]
            else:
                cand = [(j, p) for j, p in peaks if 2 < abs(j - prev) <= d3]
            if cand:
                cur = max(cand, key=lambda z: z[1])[0]
                hold = 0
                n_disc += 1
        hr[t] = cur * dfb
        prev = cur
    first = int(np.argmax(np.isfinite(hr))) if np.isfinite(hr).any() else None
    if first is None:
        # 兜底：整条记录的谱峰度都没超过阈值（原文会一直"不输出"）。
        # 为保证 1328 窗口径可比，退回"第 0 窗带内最高峰"作初值（见报告差异表）。
        return joss_track(spec, fgrid, n_win, H=H, global_escape=global_escape,
                          use_kurt=False, d1=d1, d2=d2, d3=d3, jump=jump, kurt_th=kurt_th,
                          init_band=init_band)
    if n_blank:
        hr[:first + 1][~np.isfinite(hr[:first + 1])] = hr[first]
    return hr, dict(n_blank=n_blank, n_discovery=n_disc, kurt_gate_used=use_kurt)


def solver_selfcheck(n_trial=3, seed=0):
    """合成稀疏恢复自检：高相干冗余 DFT 基上，M-FOCUSS / ISTA 能否命中真支撑。

    这是本文 sparse 求解器实现约定的**可复现证据**（报告 `reports/35` §2 差异 d）：
    M=200, N=1024, L=4, 真支撑 3 行；命中数越多越好。
    """
    rng = np.random.default_rng(seed)
    M, N, L = 200, JOSS_N, 4
    m = np.arange(M)[:, None]
    n = np.arange(N)[None, :]
    Phi = np.exp(1j * 2 * np.pi * m * n / N)
    out = []
    for trial in range(n_trial):
        sup = rng.choice(np.arange(20, 200), size=3, replace=False)
        X = np.zeros((N, L), dtype=np.complex128)
        X[sup] = (rng.normal(size=(3, L)) + 1j * rng.normal(size=(3, L))) * 5
        Y = Phi @ X + 0.01 * rng.normal(size=(M, L))
        for name, Xh in (("mfocuss_p0.8_it4", _mfocuss(Phi, Y)),
                         ("ista_l21", _ista_l21(Phi, Y, 0.02, 400))):
            top = np.argsort(-np.linalg.norm(Xh, axis=1))[:3]
            out.append(dict(trial=trial, solver=name, support=sorted(sup.tolist()),
                            top3=sorted(top.tolist()),
                            hit=int(len(set(top.tolist()) & set(sup.tolist())))))
    return out


# ===========================================================================
# 3) 本方案复算（reports/19 定稿配方）＋ 退化解 / 前端诊断
# ===========================================================================
def ours_lead(P_, rec, lead_idx, lam=0.5, jump=4.0):
    DL = load_module("dual_lead_fuse_mod", os.path.join(SCRIPTS, "tools", "dual_lead_fuse.py"))
    return DL.lead_res(P_, rec, lead_idx, lam, jump)


def _psd_argmax_bpm(P_, x, accmag, n_win, band=BAND):
    """带内直接取最大峰（无追踪）；x 已是 100 Hz。"""
    f, Pm = P_.psd_matrix(np.asarray(x, dtype=np.float64), accmag, FS_EVAL, n_win,
                          win_s=WIN_S, hop_s=HOP_S, use_notch=False)
    m = (f >= band[0]) & (f <= band[1])
    fb = f[m] * 60.0
    out = np.full(n_win, np.nan)
    for t in range(n_win):
        out[t] = fb[int(np.argmax(Pm[t][m]))]
    return out


def frontend_diag(P_, rec, n_win, lead_idx=0, lam=0.5, jump=4.0):
    """退化解 + 前端 top-1 对照：raw PPG / RLS 抵消后 / raw+Viterbi / 仅 ACC 步频。"""
    ppg = np.asarray(rec["ppg"], dtype=np.float64)
    acc = np.asarray(rec["acc"], dtype=np.float64)
    x = ppg[lead_idx]
    accmag = np.sqrt((acc ** 2).sum(0))
    DL = load_module("dual_lead_fuse_mod", os.path.join(SCRIPTS, "tools", "dual_lead_fuse.py"))
    m_hat, _ = P_.rls_decompose(x[None, :], acc.T[None, :], np.zeros_like(x)[None, :])
    m_hat = m_hat[0]
    out = {}
    out["raw_top1"] = _psd_argmax_bpm(P_, x, accmag, n_win)
    out["rls_top1"] = _psd_argmax_bpm(P_, x - m_hat, accmag, n_win)
    f, Pm = P_.psd_matrix(x, accmag, FS_EVAL, n_win, win_s=WIN_S, hop_s=HOP_S, use_notch=False)
    out["raw_viterbi"] = DL._viterbi(f, Pm, BAND, jump, lam)
    # 仅 ACC 步频（0.8–3.5 Hz 主峰）
    accv = np.full(n_win, np.nan)
    for t in range(n_win):
        s, e = int(t * HOP_S * FS_EVAL), int(t * HOP_S * FS_EVAL) + int(WIN_S * FS_EVAL)
        ff, P = _psd_fine(accmag[s:e], FS_EVAL, 4096)
        mm = (ff >= 0.8) & (ff <= 3.5)
        if mm.any():
            accv[t] = ff[mm][int(np.argmax(P[mm]))] * 60.0
    out["acc_only"] = accv
    return out


# ===========================================================================
# 逐记录 worker
# ===========================================================================
def _worker(payload):
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    import ext_loaders as EL
    recs = list(EL.iter_troika(split=payload["split"], fs_out=FS_EVAL))
    out = []
    for i in payload["idx"]:
        rec = recs[i]
        ppg = np.asarray(rec["ppg"], dtype=np.float64)
        acc = np.asarray(rec["acc"], dtype=np.float64)
        ref = np.asarray(rec["hr_ref"], dtype=np.float64).reshape(-1)
        n_win_full = int(max(0, (ppg.shape[1] - 8 * FS_EVAL) // (2 * FS_EVAL)) + 1)
        n = int(min(n_win_full, len(ref)))
        ref = ref[:n]
        row = dict(name=rec["name"], n=n, ref=ref.tolist())
        if payload["do_spama"]:
            cands = spama_candidates(ppg[0], acc, FS_EVAL, n)
            fb, Pb = spama_band_spectra(ppg[0], acc, FS_EVAL, n)
            DL = load_module("dual_lead_fuse_mod", os.path.join(SCRIPTS, "tools", "dual_lead_fuse.py"))
            row["spama_viterbi"] = DL._viterbi(fb, Pb, BAND, payload["jump"], payload["lam"]).tolist()
            row["spama"] = {}
            for cfg in payload["spama_cfgs"]:
                key = "tol%.2f_%s_ma%d_H%d" % (cfg["tol"], cfg["acc_mode"],
                                               int(cfg["use_ma"]), cfg["H"])
                hr, info = spama_track(cands, n, tol=cfg["tol"], acc_mode=cfg["acc_mode"],
                                       use_ma=cfg["use_ma"], H=cfg["H"])
                row["spama"][key] = hr.tolist()
            # 前端 oracle（用 tol 最大剔除档的存活集合做上界参考）
            top1, or3 = [], []
            for t in range(n):
                fr = cands[t]["fr"]
                if len(fr) == 0:
                    top1.append(np.nan); or3.append(np.nan); continue
                top1.append(float(fr[0]))
                k3 = int(np.argmin(np.abs(fr[:3] - ref[t])))
                or3.append(float(fr[:3][k3]))
            row["spama_frontend"] = dict(top1=top1, top3_oracle=or3)
        if payload["do_joss"]:
            row["joss"] = {}
            for solver, sub in (("mfocuss", True), ("mfocuss_nosub", False), ("ista", True)):
                spec, fg, info = joss_spectra(ppg[0], acc, FS_EVAL, n, solver=("mfocuss" if solver.startswith("mfocuss") else "ista"),
                                              use_sub=sub)
                for H in payload["joss_Hs"]:
                    for ge in payload["joss_escapes"]:
                        key = "%s_ge%d_H%d" % (solver, int(ge), H)
                        hr, tinfo = joss_track(spec, fg, n, H=H, global_escape=ge)
                        row["joss"][key] = hr.tolist()
                if solver == "mfocuss":
                    DL = load_module("dual_lead_fuse_mod", os.path.join(SCRIPTS, "tools", "dual_lead_fuse.py"))
                    mb = (fg >= 0.4) & (fg <= 4.0)
                    row["joss_viterbi"] = DL._viterbi(
                        fg[mb], spec[:, mb], BAND, payload["jump"], payload["lam"]).tolist()
                    top1, or3 = [], []
                    for t in range(n):
                        pk = sorted(_joss_peaks(spec[t], fg), key=lambda z: -z[1])
                        if not pk:
                            top1.append(np.nan); or3.append(np.nan); continue
                        top1.append(float(fg[pk[0][0]] * 60.0))
                        f3 = np.array([fg[j] for j, _ in pk[:3]]) * 60.0
                        or3.append(float(f3[int(np.argmin(np.abs(f3 - ref[t])))]))
                    row["joss_frontend"] = dict(top1=top1, top3_oracle=or3, df_bpm=info["df_bpm"])
        if payload["do_ours"]:
            hr_w, hr_p, snr = [], [], []
            for li in range(2):
                b = ours_lead(P_, rec, li, payload["lam"], payload["jump"])
                k = min(b["n_win"], n)
                hr_w.append(np.asarray(b["hr_w"][:k]).tolist())
                hr_p.append(np.asarray(b["hr"][:k]).tolist())
                snr.append(np.asarray(b["snr"][:k]).tolist())
            row["ours"] = dict(hr=hr_p, hr_w=hr_w, snr=snr)
            row["diag"] = {k: np.asarray(v, dtype=np.float64).tolist()
                           for k, v in frontend_diag(P_, rec, n, 0, payload["lam"],
                                                     payload["jump"]).items()}
        out.append(row)
    return out


# ===========================================================================
# 指标
# ===========================================================================
def err_stats(hr, ref):
    hr = np.asarray(hr, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    ok = np.isfinite(hr) & np.isfinite(ref)
    e = np.abs(hr[ok] - ref[ok])
    if not len(e):
        return dict(n=0, mae=float("nan"), med=float("nan"), gt5=float("nan"), gt10=float("nan"))
    return dict(n=int(len(e)), mae=float(np.mean(e)), med=float(np.median(e)),
                gt5=float(np.mean(e > 5.0)), gt10=float(np.mean(e > 10.0)))


def pooled_multi(rows, getter):
    H, R, per = [], [], []
    for r in rows:
        h = np.asarray(getter(r), dtype=np.float64)
        ref = np.asarray(r["ref"], dtype=np.float64)
        H.append(h); R.append(ref); per.append(err_stats(h, ref))
    st = err_stats(np.concatenate(H), np.concatenate(R))
    st["per_record"] = per
    st["med_record"] = float(np.nanmedian([p["mae"] for p in per]))
    return st


def _clean(st):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}


def _select(grid_err, keyfn):
    """在训练网格里选误差最小的配置（返回 key）。"""
    best, bk = None, None
    for k in grid_err:
        m = grid_err[k]["mae"]
        if best is None or m < best:
            best, bk = m, k
    return bk


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="TROIKA 经典 HR baseline（SpaMA/JOSS）三方对比")
    ap.add_argument("--out", default=P.artifact("troika_baselines"))
    ap.add_argument("--records", default="all", help="逗号分隔记录名，或 all")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--tune-split", default="train", choices=["test", "train"])
    ap.add_argument("--no-spama", action="store_true")
    ap.add_argument("--no-joss", action="store_true")
    ap.add_argument("--no-ours", action="store_true")
    ap.add_argument("--skip-tune", action="store_true", help="跳过训练集选超参（用内置默认）")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--jump", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--selfcheck-solver", action="store_true",
                    help="只跑稀疏求解器的合成恢复自检（不进数据）后退出")
    args = ap.parse_args()
    if args.selfcheck_solver:
        rows = solver_selfcheck()
        for r in rows:
            print("  trial%d %-18s support=%s top3=%s hit=%d"
                  % (r["trial"], r["solver"], r["support"], r["top3"], r["hit"]))
        hits = {}
        for r in rows:
            hits.setdefault(r["solver"], []).append(r["hit"])
        print("[selfcheck] 平均命中：", {k: round(float(np.mean(v)), 2) for k, v in hits.items()},
              "（满分为 3）")
        return None
    os.makedirs(args.out, exist_ok=True)
    PAR.log_env()
    t0 = time.time()

    import ext_loaders as EL
    allrec = [r["name"] for r in EL.iter_troika(split=args.split, fs_out=FS_EVAL)]
    names = allrec if args.records == "all" else [s.strip() for s in args.records.split(",")]
    idx = [allrec.index(nm) for nm in names]
    print("[bl] split=%s 记录 %d 条：%s" % (args.split, len(idx), ", ".join(names)))

    def run(split, idx_list, cfgs, do_spama, do_joss, do_ours, joss_Hs, joss_escapes):
        nw = args.workers or PAR.default_workers(len(idx_list), reserve=4)
        chunks = PAR.split_chunks(idx_list, min(len(idx_list), nw))
        payloads = [dict(split=split, idx=c, spama_cfgs=cfgs, joss_Hs=joss_Hs,
                         joss_escapes=joss_escapes, do_spama=do_spama, do_joss=do_joss,
                         do_ours=do_ours, lam=args.lam, jump=args.jump) for c in chunks]
        rows, dt = PAR.pmap(_worker, payloads, desc="TROIKA %s（SpaMA/JOSS/本方案）" % split)
        order = {r["name"]: k for k, r in enumerate(EL.iter_troika(split=split, fs_out=FS_EVAL))}
        rows.sort(key=lambda r: order[r["name"]])
        return rows, dt

    # ---------------- 1. 训练集上选超参 ----------------
    best_cfg, best_H, best_ge, tune = dict(tol=0.10, acc_mode="any", use_ma=True, H=2), 3, False, {}
    if not args.skip_tune:
        if not args.no_spama:
            tr_rows, _ = run(args.tune_split, list(range(len(EL.list_records("troika", split=args.tune_split)))),
                             SPAMA_GRID, True, False, False, [], [])
            g = {}
            for cfg in SPAMA_GRID:
                k = "tol%.2f_%s_ma%d_H%d" % (cfg["tol"], cfg["acc_mode"], int(cfg["use_ma"]), cfg["H"])
                st = pooled_multi(tr_rows, lambda r, _k=k: r["spama"][_k])
                g[k] = dict(mae=st["mae"], med=st["med"], gt5=st["gt5"])
            bk = _select(g, None)
            for cfg in SPAMA_GRID:
                k = "tol%.2f_%s_ma%d_H%d" % (cfg["tol"], cfg["acc_mode"], int(cfg["use_ma"]), cfg["H"])
                if k == bk:
                    best_cfg = dict(cfg)
            tune["spama_train_grid"] = {k: {kk: round(vv, 4) for kk, vv in v.items()}
                                        for k, v in g.items()}
            print("[bl] SpaMA 超参（%s 集 %d 条）选出：%s，train MAE %.3f"
                  % (args.tune_split, len(tr_rows), bk, g[bk]["mae"]))
        if not args.no_joss:
            tr_rows, _ = run(args.tune_split, list(range(len(EL.list_records("troika", split=args.tune_split)))),
                             [], False, True, False, JOSS_H_GRID, [False, True])
            g = {}
            for H in JOSS_H_GRID:
                for ge in (False, True):
                    k = "mfocuss_ge%d_H%d" % (int(ge), H)
                    st = pooled_multi(tr_rows, lambda r, _k=k: r["joss"][_k])
                    g[k] = dict(mae=st["mae"], med=st["med"], gt5=st["gt5"])
            bk = _select(g, None)
            best_H = int(bk.split("_H")[-1]); best_ge = bool(int(bk.split("_")[1][2:]))
            tune["joss_train_grid"] = {k: {kk: round(vv, 4) for kk, vv in v.items()}
                                       for k, v in g.items()}
            print("[bl] JOSS 超参（%s 集）选出：%s，train MAE %.3f"
                  % (args.tune_split, bk, g[bk]["mae"]))

    # ---------------- 2. 评估 ----------------
    rows, dt = run(args.split, idx, SPAMA_GRID if not args.no_spama else [],
                   not args.no_spama, not args.no_joss, not args.no_ours,
                   JOSS_H_GRID, [False, True])
    RS = [np.asarray(r["ref"], dtype=np.float64) for r in rows]
    n_win = int(sum(len(r) for r in RS))
    print("[bl] 评估窗口数 = %d（规格 1328）" % n_win)

    out = dict(
        protocol=dict(
            data="ext_loaders.iter_troika(split=%s, fs_out=100)" % args.split,
            n_records=len(rows), n_windows=n_win,
            metric="窗级池化 HR MAE (bpm)；中位=窗级中位绝对误差；>5bpm=误差>5 的窗占比；中位记录=逐记录 MAE 的中位",
            spama="Salehizadeh et al., Sensors 16(1):10 (2016)；20 Hz, 8s/2s, 0.5-3 Hz 前 3 峰 + ACC 主峰剔除 + 10bpm 峰值保持",
            joss="Zhang, IEEE TBME 62(8):1902-1910 (2015)；25 Hz, MMV(L=4) + Regularized M-FOCUSS(p=0.8,lam=1e-10,4iter) + 谱减 + 四段峰跟踪",
            ours="reports/19 定稿：RLS(3轴,α=1) + PSD + 可靠性加权 Viterbi，记录级 snr_peak 选路(margin 1.2)",
            spama_cfg_chosen=best_cfg, joss_cfg_chosen=dict(H=best_H, global_escape=best_ge),
            tuned_on=args.tune_split),
        seconds=round(time.time() - t0, 1), eval_seconds=round(dt, 1),
        per_record=[dict(name=r["name"], n=r["n"]) for r in rows],
        tune=tune)
    summary, per_rec_stats = {}, {}
    best_key, best_jk = None, None

    # ---- SpaMA：主结果 + 全网格 + LORO ----
    if not args.no_spama:
        key_all = ["tol%.2f_%s_ma%d_H%d" % (c["tol"], c["acc_mode"], int(c["use_ma"]), c["H"])
                   for c in SPAMA_GRID]
        best_key = "tol%.2f_%s_ma%d_H%d" % (best_cfg["tol"], best_cfg["acc_mode"],
                                            int(best_cfg["use_ma"]), best_cfg["H"])
        st = pooled_multi(rows, lambda r: r["spama"][best_key])
        summary["spama"] = _clean({k: v for k, v in st.items() if k != "per_record"})
        per_rec_stats["spama"] = st["per_record"]
        st_lit = pooled_multi(rows, lambda r: r["spama"]["tol0.00_any_ma0_H2"])
        summary["spama_noMA"] = _clean({k: v for k, v in st_lit.items() if k != "per_record"})
        grid = {}
        for k in key_all:
            s2 = pooled_multi(rows, lambda r, _k=k: r["spama"][_k])
            grid[k] = _clean({k2: v2 for k2, v2 in s2.items() if k2 != "per_record"})
        out["spama_grid_%s" % args.split] = grid
        # 前端 oracle
        o1 = err_stats(np.concatenate([np.asarray(r["spama_frontend"]["top1"]) for r in rows]),
                       np.concatenate(RS))
        o3 = err_stats(np.concatenate([np.asarray(r["spama_frontend"]["top3_oracle"]) for r in rows]),
                       np.concatenate(RS))
        out["spama_frontend_oracle"] = dict(top1=_clean(o1), top3=_clean(o3))
        # LORO：逐记录用其余记录选配置（允许的记录间留一）
        # 记录数 <3 时留一没有意义（没有"其余记录"可选），跳过。
        picks, lerr = [], []
        for i in range(len(rows) if len(rows) >= 3 else 0):
            bg, bk = None, None
            for k in key_all:
                H = np.concatenate([np.asarray(rows[j]["spama"][k]) for j in range(len(rows)) if j != i])
                R = np.concatenate([RS[j] for j in range(len(rows)) if j != i])
                m = err_stats(H, R)["mae"]
                if bg is None or m < bg:
                    bg, bk = m, k
            picks.append(bk)
            lerr.append(np.abs(np.asarray(rows[i]["spama"][bk]) - RS[i]))
        if lerr:
            summary["spama_LORO"] = dict(mae=round(float(np.mean(np.concatenate(lerr))), 4),
                                         picks={r["name"]: k for r, k in zip(rows, picks)})
        else:
            out["spama_LORO_note"] = "记录数 <3，未做记录间留一"
    # ---- JOSS ----
    if not args.no_joss:
        jkeys = ["%s_ge%d_H%d" % (s, int(ge), H) for s in ("mfocuss", "mfocuss_nosub", "ista")
                 for ge in (False, True) for H in JOSS_H_GRID]
        best_jk = "mfocuss_ge%d_H%d" % (int(best_ge), best_H)
        st = pooled_multi(rows, lambda r: r["joss"][best_jk])
        summary["joss"] = _clean({k: v for k, v in st.items() if k != "per_record"})
        per_rec_stats["joss"] = st["per_record"]
        for nm, k in (("joss_noSub", "mfocuss_nosub_ge0_H%d" % best_H),
                      ("joss_ista", "ista_ge0_H%d" % best_H)):
            s2 = pooled_multi(rows, lambda r, _k=k: r["joss"][_k])
            summary[nm] = _clean({k2: v2 for k2, v2 in s2.items() if k2 != "per_record"})
            per_rec_stats[nm] = s2["per_record"]
        grid = {}
        for k in jkeys:
            s2 = pooled_multi(rows, lambda r, _k=k: r["joss"][_k])
            grid[k] = _clean({k2: v2 for k2, v2 in s2.items() if k2 != "per_record"})
        out["joss_grid_%s" % args.split] = grid
        o1 = err_stats(np.concatenate([np.asarray(r["joss_frontend"]["top1"]) for r in rows]),
                       np.concatenate(RS))
        o3 = err_stats(np.concatenate([np.asarray(r["joss_frontend"]["top3_oracle"]) for r in rows]),
                       np.concatenate(RS))
        out["joss_frontend_oracle"] = dict(top1=_clean(o1), top3=_clean(o3),
                                           df_bpm=rows[0]["joss_frontend"]["df_bpm"])
    # ---- 本方案 + 退化解 ----
    if not args.no_ours:
        sel = []
        for r in rows:
            c1 = float(np.median(np.asarray(r["ours"]["snr"][0])))
            c2 = float(np.median(np.asarray(r["ours"]["snr"][1])))
            sel.append(1 if c2 > 1.2 * c1 else 0)
        specs = [("ours_lead1", lambda r: r["ours"]["hr"][0]),
                 ("ours_lead1_relw", lambda r: r["ours"]["hr_w"][0]),
                 ("ours_sel", lambda r: r["ours"]["hr_w"][sel[rows.index(r)]]),
                 ("diag_raw_top1", lambda r: r["diag"]["raw_top1"]),
                 ("diag_rls_top1", lambda r: r["diag"]["rls_top1"]),
                 ("diag_raw_viterbi", lambda r: r["diag"]["raw_viterbi"]),
                 ("diag_acc_only", lambda r: r["diag"]["acc_only"]),
                 ("spamaFE_oursTrack", lambda r: r["spama_viterbi"]),
                 ("jossFE_oursTrack", lambda r: r["joss_viterbi"])]
        for nm, fn in specs:
            st = pooled_multi(rows, fn)
            summary[nm] = _clean({k: v for k, v in st.items() if k != "per_record"})
            per_rec_stats[nm] = st["per_record"]
        E = np.concatenate([np.abs(np.asarray(r["ours"]["hr_w"][sel[i]]) - RS[i])
                            for i, r in enumerate(rows)])
        C = np.concatenate([np.asarray(r["ours"]["snr"][sel[i]]) for i, r in enumerate(rows)])
        order = np.argsort(-C)
        out["ours_coverage"] = {str(c): round(float(np.mean(E[order][:max(1, int(len(E) * c))])), 4)
                                for c in (1.0, 0.9, 0.8, 0.7, 0.5)}
        out["ours_lead_selection"] = {r["name"]: int(li) for r, li in zip(rows, sel)}
        # 配对自助（seed=0，窗级配对）
        rr = np.random.default_rng(0)
        base = np.concatenate([np.asarray(r["ours"]["hr_w"][sel[i]]) - RS[i]
                               for i, r in enumerate(rows)])
        bs = {}
        for nm, kk in (("spama", best_key if not args.no_spama else None),
                       ("joss", best_jk if not args.no_joss else None)):
            if kk is None or nm not in summary:
                continue
            arr = np.concatenate([(np.asarray(r["spama"][kk]) if nm == "spama"
                                   else np.asarray(r["joss"][kk])) - RS[i]
                                  for i, r in enumerate(rows)])
            d = np.abs(base) - np.abs(arr)
            boot = np.array([float(np.mean(d[rr.integers(0, len(d), len(d))])) for _ in range(2000)])
            bs[nm] = dict(diff_mae=round(float(np.mean(d)), 3),
                          ci95=[round(float(np.percentile(boot, 2.5)), 3),
                                round(float(np.percentile(boot, 97.5)), 3)],
                          p_better=round(float(np.mean(boot < 0)), 3))
        out["paired_bootstrap_ours_vs"] = bs

    out["summary"] = summary
    out["per_record_stats"] = per_rec_stats
    # 逐窗数据（前 4 个方法；便于复算）
    wd = {}
    for r in rows:
        d = dict(ref=r["ref"], n=r["n"])
        if not args.no_spama:
            d["spama"] = r["spama"]["tol%.2f_%s_ma%d_H%d" % (
                best_cfg["tol"], best_cfg["acc_mode"], int(best_cfg["use_ma"]), best_cfg["H"])]
        if not args.no_joss:
            d["joss"] = r["joss"]["mfocuss_ge%d_H%d" % (int(best_ge), best_H)]
        if not args.no_ours:
            d["ours_sel"] = r["ours"]["hr_w"][out["ours_lead_selection"][r["name"]]]
        wd[r["name"]] = d
    out["window_data"] = wd
    with open(os.path.join(args.out, "troika_baselines.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "troika_baselines_per_record.csv"), "w",
              newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "record", "n_windows", "mae", "median_abs_err", "gt5_ratio", "gt10_ratio"])
        for nm in ("spama", "joss", "joss_noSub", "joss_ista", "ours_lead1", "ours_lead1_relw",
                   "ours_sel", "diag_raw_top1", "diag_rls_top1", "diag_raw_viterbi",
                   "diag_acc_only", "spamaFE_oursTrack", "jossFE_oursTrack"):
            if nm not in per_rec_stats:
                continue
            for r, p in zip(rows, per_rec_stats[nm]):
                wr.writerow(["%s(%s)" % (nm, args.split), r["name"], p["n"], round(p["mae"], 4),
                             round(p["med"], 4), round(p["gt5"], 4), round(p["gt10"], 4)])
    print("\n[bl] ==== %s（%d 窗 / %d 条）==== " % (args.split, n_win, len(rows)))
    print("  %-20s %8s %8s %8s %9s" % ("method", "MAE", "中位", ">5bpm", "中位记录MAE"))
    for nm in ("spama", "spama_noMA", "spama_LORO", "joss", "joss_noSub", "joss_ista",
               "ours_lead1", "ours_lead1_relw", "ours_sel", "diag_raw_top1", "diag_rls_top1",
               "diag_raw_viterbi", "diag_acc_only", "spamaFE_oursTrack", "jossFE_oursTrack"):
        if nm not in summary:
            continue
        st = summary[nm]
        if nm == "spama_LORO":
            print("  %-20s %8.2f %8s %8s %9s" % (nm + "(留一)", st["mae"], "-", "-", "-"))
            continue
        print("  %-20s %8.2f %8.2f %7.1f%% %9.2f"
              % (nm, st["mae"], st["med"], 100 * st["gt5"], st.get("med_record", float("nan"))))
    if "spama_frontend_oracle" in out:
        o = out["spama_frontend_oracle"]
        print("  [前端] SpaMA raw-PPG 谱 top1 MAE %.2f / top3-oracle %.2f"
              % (o["top1"]["mae"], o["top3"]["mae"]))
    if "joss_frontend_oracle" in out:
        o = out["joss_frontend_oracle"]
        print("  [前端] JOSS 稀疏+谱减 top1 MAE %.2f / top3-oracle %.2f（bin=%.2f bpm）"
              % (o["top1"]["mae"], o["top3"]["mae"], o["df_bpm"]))
    if "ours_coverage" in out:
        print("  本方案覆盖率：", {k: round(v, 2) for k, v in out["ours_coverage"].items()})
    print("[bl] 完成 -> %s（%.1f s）" % (args.out, time.time() - t0))
    return out


if __name__ == "__main__":
    main()

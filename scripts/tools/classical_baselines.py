#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CIME 波形任务：经典/传统方法 baseline 全家桶（批次 A）。

口径与主实验完全一致：输入为 `outputs/final_10seed/features/test.npz` 的
`X = [PPG, M̂, PPG_RLS, ACC_x, Gyro_y]`（逐段标准化），真值 `y`；指标 PRD / Pearson / SNR。

方法清单（应测尽测）
--------------------
| 类别 | 方法 |
|---|---|
| 恒等 | `raw` |
| 线性滤波 | `bandpass_0.5_4`、`bandpass_0.7_3`、`movavg_0.2s`、`medfilt_0.1s`、`savgol_11_3` |
| 统计滤波 | `wiener_15` |
| 模态分解 | `emd_drop_ifm0_trend`、`eemd_drop_ifm0_trend`（PyEMD） |
| 自适应滤波 | `lms`（ACC 参考）、`nlms`（ACC+Gyro 参考） |
| 物理先验（本方案主干） | `rls_a1`、`rls_a0.4` |
| 频域 | `specsub`（用 M̂ 谱做谱减）、`rls_specsub` |
| 组合 | `rls_bandpass`、`rls_savgol` |

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/classical_baselines.py --out outputs/baselines --limit 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "modules"))
from _common import paths as P                                # noqa: E402
from _common import parallel as PAR                           # noqa: E402

FS = 100.0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- 各方法（输入 ppg/acc/gyro/mhat，输出同样长度） ----------------
def _bp(x, lo, hi, order=3):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [lo / (FS / 2), hi / (FS / 2)], btype="band")
    return filtfilt(b, a, x)


def m_raw(ppg, mhat, acc, gyro):
    return ppg.copy()


def m_bp054(ppg, mhat, acc, gyro):
    return _bp(ppg, 0.5, 4.0)


def m_bp073(ppg, mhat, acc, gyro):
    return _bp(ppg, 0.7, 3.0)


def m_movavg(ppg, mhat, acc, gyro):
    w = int(0.2 * FS)
    k = np.ones(w) / w
    return np.convolve(ppg, k, mode="same")


def m_medfilt(ppg, mhat, acc, gyro):
    from scipy.signal import medfilt
    return medfilt(ppg, kernel_size=int(0.1 * FS) | 1)


def m_savgol(ppg, mhat, acc, gyro):
    from scipy.signal import savgol_filter
    return savgol_filter(ppg, 11, 3)


def m_wiener(ppg, mhat, acc, gyro):
    from scipy.signal import wiener
    return wiener(ppg, mysize=15)


def m_emd(ppg, mhat, acc, gyro):
    from PyEMD import EMD
    imfs = EMD().emd(ppg)
    if imfs.shape[0] < 3:
        return ppg.copy()
    return imfs[1:-1].sum(axis=0)


def m_eemd(ppg, mhat, acc, gyro):
    from PyEMD import EEMD
    imfs = EEMD(trials=8).eemd(ppg)
    if imfs.shape[0] < 3:
        return ppg.copy()
    return imfs[1:-1].sum(axis=0)


def _lms(x, d, taps=16, mu=0.005):
    """NLMS 自适应抵消：用参考 d 预测 x 中的伪影分量，返回残差 x − 估计。

    标准归一化步长 `w += mu · e · buf / (‖buf‖² + ε)`；对 `mu` 上限与权重做钳制以防发散
    （首版 mu=0.05 未钳制 → PRD 826 发散）。
    """
    w = np.zeros(taps)
    buf = np.zeros(taps)
    y = np.zeros_like(x)
    for n in range(len(x)):
        buf[1:] = buf[:-1]; buf[0] = d[n]
        e = x[n] - w @ buf
        w += mu * e * buf / (float(buf @ buf) + 1e-3)
        np.clip(w, -50.0, 50.0, out=w)
        y[n] = e
    return y


def m_lms(ppg, mhat, acc, gyro):
    return _lms(ppg, acc, taps=16, mu=0.005)


def m_nlms(ppg, mhat, acc, gyro):
    # 两路参考（异质）拼接
    y1 = _lms(ppg, acc, taps=16, mu=0.005)
    return _lms(y1, gyro, taps=16, mu=0.005)


def m_rls_a1(ppg, mhat, acc, gyro):
    return ppg - mhat


def m_rls_a04(ppg, mhat, acc, gyro):
    return ppg - 0.40 * mhat


def m_specsub(ppg, mhat, acc, gyro):
    """谱减（保留含噪相位）：|S_out| = max(|S_ppg| − |S_m̂|, 0.05·|S_ppg|)，相位取原信号。"""
    from scipy.signal import stft, istft
    npseg, nov = 256, 192
    _, _, Zp = stft(ppg, FS, nperseg=npseg, noverlap=nov)
    _, _, Zn = stft(mhat, FS, nperseg=npseg, noverlap=nov)
    L = min(Zp.shape[1], Zn.shape[1])
    Zp, Zn = Zp[:, :L], Zn[:, :L]
    mag = np.maximum(np.abs(Zp) - np.abs(Zn), 0.05 * np.abs(Zp))
    Z = mag * np.exp(1j * np.angle(Zp))
    _, out = istft(Z, FS, nperseg=npseg, noverlap=nov)
    out = np.asarray(out, float)
    if len(out) < len(ppg):
        out = np.pad(out, (0, len(ppg) - len(out)))
    return out[:len(ppg)]


def m_rls_specsub(ppg, mhat, acc, gyro):
    return m_specsub(m_rls_a1(ppg, mhat, acc, gyro), mhat, acc, gyro)


def m_rls_bp(ppg, mhat, acc, gyro):
    return _bp(m_rls_a04(ppg, mhat, acc, gyro), 0.7, 3.0)


def m_rls_sg(ppg, mhat, acc, gyro):
    from scipy.signal import savgol_filter
    return savgol_filter(m_rls_a04(ppg, mhat, acc, gyro), 11, 3)


METHODS = {
    "raw": m_raw, "bandpass_0.5_4": m_bp054, "bandpass_0.7_3": m_bp073,
    "movavg_0.2s": m_movavg, "medfilt_0.1s": m_medfilt, "savgol_11_3": m_savgol,
    "wiener_15": m_wiener, "emd_drop_ifm0_trend": m_emd, "eemd_drop_ifm0_trend": m_eemd,
    "lms_acc16": m_lms, "nlms_acc_gyro": m_nlms,
    "rls_a1": m_rls_a1, "rls_a0.4": m_rls_a04, "specsub": m_specsub,
    "rls_specsub": m_rls_specsub, "rls_bandpass": m_rls_bp, "rls_savgol": m_rls_sg,
}
SLOW = {"eemd_drop_ifm0_trend"}          # EEMD 慢 → 只在子集上跑


def _worker(payload):
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    d = np.load(os.path.join(payload["feat_dir"], "test.npz"))
    X, Y = d["X"], d["y"]
    n = len(Y)
    idx = payload["idx"]
    out = []
    for name in payload["methods"]:
        f = METHODS[name]
        errs = []
        for i in idx:
            if i >= n:
                continue
            try:
                e = f(X[i, 0], X[i, 1], X[i, 3], X[i, 4])
            except Exception:
                continue
            e = np.asarray(e, float)
            if len(e) != len(Y[i]) or not np.all(np.isfinite(e)):
                continue
            errs.append((P_.prd(Y[i], e), P_.snr_db(e, Y[i]),
                         float(np.corrcoef(Y[i], e)[0, 1])))
        if errs:
            a = np.array(errs)
            # ⚠️ PAR.pmap 会拍平 dict（拆成键）→ 必须返回 list
            out.append(dict(method=name, n=len(a), PRD=float(np.nanmean(a[:, 0])),
                            SNR=float(np.nanmean(a[:, 1])),
                            Pearson=float(np.nanmean(a[:, 2]))))
    return out


def main():
    ap = argparse.ArgumentParser(description="经典 baseline 全家桶")
    ap.add_argument("--feat-dir", default=P.artifact("final_10seed/features"))
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 段（快速验证）")
    ap.add_argument("--eemd-limit", type=int, default=200, help="EEMD 只在前 N 段上跑（慢）")
    ap.add_argument("--out", default=P.artifact("baselines"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    PAR.log_env()

    d = np.load(os.path.join(args.feat_dir, "test.npz"))
    n_all = len(d["y"])

    def run(methods, idx, tag):
        n = len(idx)
        chunks = PAR.split_chunks(idx, PAR.default_workers(n, reserve=4, cap=16))
        res, dt = PAR.pmap(_worker, [dict(feat_dir=args.feat_dir, idx=c, methods=methods,
                                          limit=False) for c in chunks],
                           desc="%s（%d 段 × %d 方法）" % (tag, n, len(methods)))
        agg = {}
        for r in res:
            if isinstance(r, dict) and "method" in r:
                agg.setdefault(r["method"], []).append(r)
        out = {}
        for k, vs in agg.items():
            tot = sum(v["n"] for v in vs)
            out[k] = dict(n=tot,
                          PRD=sum(v["PRD"] * v["n"] for v in vs) / tot,
                          SNR=sum(v["SNR"] * v["n"] for v in vs) / tot,
                          Pearson=sum(v["Pearson"] * v["n"] for v in vs) / tot)
        return out, dt

    fast = [m for m in args.methods if m not in SLOW]
    idx = list(range(n_all)) if not args.limit else list(range(min(args.limit, n_all)))
    table, dt1 = run(fast, idx, "经典 baseline")
    if "eemd_drop_ifm0_trend" in args.methods:
        idx2 = list(range(min(args.eemd_limit, n_all)))
        slow, dt2 = run(["eemd_drop_ifm0_trend"], idx2, "EEMD 子集")
        # 同时在同一子集上跑 raw 作对照
        base, _ = run(["raw"], idx2, "raw 子集")
        table.update(slow)
        table["raw_on_eemd_subset"] = base["raw"]
        table["eemd_drop_ifm0_trend"]["subset"] = len(idx2)

    print("\n%-26s %6s %10s %10s %10s" % ("方法", "段数", "PRD↓", "SNR(dB)", "Pearson"))
    for k in sorted(table, key=lambda k: table[k]["PRD"]):
        v = table[k]
        print("%-26s %6d %10.2f %10.2f %10.4f" % (k, v["n"], v["PRD"], v["SNR"], v["Pearson"]))
    json.dump(dict(protocol="CIME 波形任务经典 baseline；输入/真值/指标与主实验同口径"
                            "（features/test.npz 的 X=[PPG,M̂,PPG_RLS,ACC_x,Gyro_y] 与 y）",
                   n_segments=len(idx), seconds=round(dt1, 1), table=table),
              open(os.path.join(args.out, "classical_baselines.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[base] 完成 -> %s" % args.out)
    return table


if __name__ == "__main__":
    main()

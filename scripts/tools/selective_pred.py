#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""P4 选择性预测：覆盖率–误差曲线（低置信时**弃权**而不是给错值）。

动机
----
跨域部署里"知道什么时候不可信"比"平均精度"更重要：运动域（TROIKA）在某些窗必然失效，
若能**弃权**这些窗，临床/运动场景的可用性会显著提升。本工具量化这个权衡。

置信信号（全部无标签，且尽量互相独立）
--------------------------------------
| 信号 | 定义 | 适用 |
|---|---|---|
| `snr` | 估计心率 bin 的谱峰 / 带内中位（dB） | 全部 |
| `s2` | 峰值处 脉搏/(脉搏+伪影)（需 IMU 才有伪影谱） | 有 IMU |
| `acf_agree` | −\|HR_viterbi − HR_自相关\|（两个不同估计器的一致性） | 全部 |
| `long_agree` | −\|HR_viterbi − HR_60s长窗\| | 全部 |
| `lead_agree` | −\|HR_lead1 − HR_lead2\|（TROIKA 双导路；需 ≥2 导联） | TROIKA |
| `combined` | 上述信号各自 z-score 后等权平均 | 全部 |

输出
----
每个域在覆盖率 100/90/80/70/50% 下的 MAE（弃权 = 丢掉置信最低的那些窗）。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/selective_pred.py --datasets troika wesad bidmc capnobase --out outputs/selective
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
BAND = (0.7, 3.0)
COVS = (1.0, 0.9, 0.8, 0.7, 0.5)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _viterbi(freq, Pmat, band=BAND, jump_bpm=4.0, lam=0.5, w=None):
    m = (freq >= band[0]) & (freq <= band[1])
    idx = np.where(m)[0]
    fb = freq[idx] * 60.0
    logP = np.log(Pmat[:, idx] + 1e-30)
    if w is not None:
        ww = w[:logP.shape[0], None]
        logP = ww * logP + (1 - ww) * logP.mean(axis=1, keepdims=True)
    T, K = logP.shape
    trans = -lam * ((fb[:, None] - fb[None, :]) / jump_bpm) ** 2
    dp = logP[0].copy(); bp = np.zeros((T, K), np.int32)
    for t in range(1, T):
        cand = dp[:, None] + trans
        bp[t] = np.argmax(cand, axis=0)
        dp = logP[t] + cand.max(axis=0)
    path = np.zeros(T, np.int32); path[-1] = int(np.argmax(dp))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t][path[t]]
    return fb[path]


def _acf_hr(x, n_win, fs=FS, band=BAND, win_s=8.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    xb = filtfilt(b, a, x - x.mean())
    lo, hi = int(fs / band[1]), int(fs / band[0])
    win = int(win_s * fs)
    out = np.full(n_win, np.nan)
    for t in range(n_win):
        s = int(t * 2.0 * fs); seg = xb[s:s + win]
        if len(seg) < win // 2:
            continue
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        ac /= (ac[0] + 1e-30)
        j = int(np.argmax(ac[lo:hi]))
        out[t] = 60.0 * fs / (lo + j)
    return out


def _long_hr(x, n_win, fs=FS, band=BAND, win_s=60.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="band")
    xb = filtfilt(b, a, x - x.mean())
    lo, hi = int(fs / band[1]), int(fs / band[0])
    win = int(win_s * fs)
    out = np.full(n_win, np.nan)
    for t in range(n_win):
        c = int((t * 2.0 + 4.0) * fs); s = max(0, c - win // 2); e = min(len(xb), s + win)
        seg = xb[s:e]
        if len(seg) < win // 3:
            continue
        ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
        ac /= (ac[0] + 1e-30)
        j = int(np.argmax(ac[lo:hi]))
        out[t] = 60.0 * fs / (lo + j)
    return out


def _worker(payload):
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    CDE = load_module("cde", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
    import ext_loaders as EL
    ds = payload["ds"]
    recs, has_acc = CDE.iter_dataset(EL, ds)
    out = []
    for i in payload["idx"]:
        rec = recs[i]
        use_rls = has_acc
        prep = CDE.prepare(P_, rec, use_rls)
        if prep is None:
            continue
        ppg, m_hat, accmag, ref, n_win, _ = prep
        sig = (ppg - m_hat) if use_rls else ppg
        f, Pm = P_.psd_matrix(np.asarray(sig, float), accmag, FS, n_win, win_s=8.0, hop_s=2.0, use_notch=False)
        hr = _viterbi(f, Pm, BAND, payload["jump"], payload["lam"])
        mb = (f >= BAND[0]) & (f <= BAND[1])
        fb = f[mb] * 60.0
        snr = np.zeros(n_win); s2 = np.zeros(n_win)
        Part = None
        if use_rls:
            _, Part = P_.psd_matrix(m_hat, accmag, FS, n_win, win_s=8.0, hop_s=2.0, use_notch=False)
        for t in range(n_win):
            Pv = Pm[t][mb]
            j = int(np.argmin(np.abs(fb - hr[t])))
            snr[t] = 10 * np.log10(Pv[j] / (np.median(Pv) + 1e-30) + 1e-9)
            if Part is not None:
                Pa = Part[t][mb]
                s2[t] = Pv[j] / (Pv[j] + Pa[j] + 1e-30)
        acf = _acf_hr(np.asarray(sig, float), n_win)
        lng = _long_hr(np.asarray(sig, float), n_win)
        ref = np.asarray(ref, float)
        n = min(len(hr), len(ref))
        out.append(dict(name=rec["name"], n=n,
                        hr=hr[:n].tolist(), ref=ref[:n].tolist(), snr=snr[:n].tolist(),
                        s2=s2[:n].tolist(),
                        acf=(-np.abs(hr[:n] - acf[:n])).tolist(),
                        lng=(-np.abs(hr[:n] - lng[:n])).tolist(),
                        has_imu=bool(use_rls)))
    return out


def zs(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)


def combine(rows, sigs):
    """把若干信号各自 z-score 后等权平均（跨记录一起标准化）。"""
    allv = {k: [] for k in sigs}
    for r in rows:
        for k in sigs:
            allv[k].append(np.asarray(r[k], float))
    Z = None
    for k in sigs:
        v = np.concatenate(allv[k])
        z = zs(v)
        Z = z if Z is None else Z + z
    return Z / len(sigs)


def curve(err, conf, covs=COVS):
    """按置信度排序，报告各覆盖率下的 MAE（弃权 = 丢掉置信最低的窗）。"""
    ok = np.isfinite(err) & np.isfinite(conf)
    e, c = err[ok], conf[ok]
    order = np.argsort(-c)                       # 置信从高到低
    e = e[order]
    out = {}
    for cv in covs:
        k = max(1, int(len(e) * cv))
        out[cv] = round(float(np.mean(e[:k])), 3)
    return out, len(e)


def main():
    ap = argparse.ArgumentParser(description="P4 选择性预测（覆盖率–误差曲线）")
    ap.add_argument("--datasets", nargs="+", default=["troika", "wesad", "bidmc", "capnobase"])
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--jump", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=P.artifact("selective"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    PAR.log_env()

    summary = {}
    for ds in args.datasets:
        P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
        CDE = load_module("cde", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
        import ext_loaders as EL
        if ds == "wrist":
            recs, has_acc = list(EL.iter_wrist(fs_out=FS)), True
        else:
            recs, has_acc = CDE.iter_dataset(EL, ds)
        if args.limit:
            recs = recs[:args.limit]
        chunks = PAR.split_chunks(list(range(len(recs))), PAR.default_workers(len(recs), reserve=4))
        rows, _ = PAR.pmap(_worker, [dict(ds=ds, idx=c, lam=args.lam, jump=args.jump) for c in chunks],
                           desc="%s 置信信号" % ds)
        if not rows:
            raise SystemExit("[selective] 没有可用记录（检查 --limit/数据集）")
        err = np.concatenate([np.abs(np.asarray(r["hr"]) - np.asarray(r["ref"])) for r in rows])
        sigs = ["snr"] + (["s2"] if rows and rows[0]["has_imu"] else []) + ["acf", "lng"]
        conf = {}
        for k in sigs:
            conf[k] = np.concatenate([np.asarray(r[k], float) for r in rows])
        conf["combined"] = combine(rows, sigs)
        base = float(np.nanmean(err))
        res = {"n_windows": int(np.isfinite(err).sum()), "baseline_MAE": round(base, 3), "curves": {}}
        print("\n[P4] %-10s 窗 %d，基线 MAE %.2f" % (ds, res["n_windows"], base))
        print("      %-9s %s" % ("覆盖率→", " ".join("%6.0f%%" % (100 * c) for c in COVS)))
        for k in sigs + ["combined"]:
            cur, _ = curve(err, conf[k])
            res["curves"][k] = cur
            print("      %-9s %s" % (k, " ".join("%7.2f" % cur[c] for c in COVS)))
        summary[ds] = res

    with open(os.path.join(args.out, "selective.json"), "w", encoding="utf-8") as f:
        json.dump(dict(protocol="P4 选择性预测：按无标签置信信号排序，报告覆盖率 100/90/80/70/50%% 下的 MAE",
                       results=summary), f, ensure_ascii=False, indent=2)
    print("\n[P4] 完成 -> %s" % args.out)
    return summary


if __name__ == "__main__":
    main()

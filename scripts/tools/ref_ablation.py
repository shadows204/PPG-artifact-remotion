#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考通道消融：RLS 的 IMU 参考怎么选才最优（本轮由用户提示触发）。

背景
----
本项目的跨域默认一直用 **3 轴 ACC**；但 CIME 上的消融早就显示**异质两路**更好
（仅 ACC 89.05 / 仅 Gyro 90.44 / **ACC_x+Gyro_y 两路 83.78**）。
`reports/26` 之后在 WRIST 上复测发现 **ACC_x+Gyro_y 把池化 MAE 从 21.99 降到 16.03（−27%）**，
而把 ACC+Gyro 全上（6 路）反而更差（23.05）→ 说明参考通道**不是越多越好**，存在"异质两路"的最优点。

变体
----
`acc1/acc2/acc3`、`gyro1/gyro2/gyro3`、`pair_xy`(acc_x+acc_y)、
`pair_ag`(**acc_x+gyro_y，CIME 配置**)、`acc3g1`、`acc3g3`（域内有 gyro 才测）

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/ref_ablation.py --datasets troika wesad wrist dalia --out outputs/ref_ablation
"""
from __future__ import annotations

import argparse
import glob
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _refs(pick, acc, gyro, like):
    """按变体名构造 (acc_ref, gyro_ref)，形状 (1,N,K) 或 (1,N)。"""
    zero = np.zeros_like(like)[None, :]
    if pick == "acc1":
        return acc[0:1].T[None, :], zero
    if pick == "acc2":
        return acc[:2].T[None, :], zero
    if pick == "acc3":
        return acc.T[None, :], zero
    if pick == "pair_xy":
        return acc[:2].T[None, :], zero
    if pick == "gyro1":
        return zero, gyro[0:1].T[None, :]
    if pick == "gyro3":
        return zero, gyro.T[None, :]
    if pick == "pair_ag":
        return acc[0:1].T[None, :], gyro[1:2].T[None, :]
    if pick == "acc3g1":
        return acc.T[None, :], gyro[0:1].T[None, :]
    if pick == "acc3g3":
        return acc.T[None, :], gyro.T[None, :]
    raise ValueError(pick)


def _series(P_, rec, pick, lam=0.5, jump=4.0, band=BAND):
    ppg = np.asarray(rec["ppg"], float)
    ppg = ppg.reshape(-1) if ppg.ndim == 1 else ppg[0]
    acc = np.asarray(rec["acc"], float)
    if acc.ndim == 1:
        acc = acc[None, :]
    gyro = rec["meta"].get("gyro")
    gyro = np.asarray(gyro, float) if gyro is not None else None
    if gyro is None and pick.startswith(("gyro", "pair_ag", "acc3g")):
        return None
    if gyro is None:
        gyro = np.zeros((3, len(ppg)))
    a_ref, g_ref = _refs(pick, acc, gyro, ppg)
    m_hat, _ = P_.rls_decompose(ppg[None, :], a_ref, g_ref)
    sig = ppg - m_hat[0]
    accmag = np.sqrt((np.asarray(acc, float) ** 2).sum(0))
    n = int(max(0, (len(sig) - 8 * FS) // (2 * FS)) + 1)
    if n < 10:
        return None
    f, Pm = P_.psd_matrix(sig, accmag, FS, n, win_s=8.0, hop_s=2.0, use_notch=False)
    hr = P_.sel_viterbi(f, Pm, band=band, jump_bpm=jump, lam=lam)
    ref = None
    if rec.get("hr_ref") is not None:
        ref = np.asarray(rec["hr_ref"], float)[:len(hr)]
    elif rec.get("ecg") is not None:
        CDE2 = load_module("cde2", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
        ref = CDE2.ecg_hr(np.asarray(rec["ecg"], float).reshape(-1), FS, len(hr), method="v2")
    if ref is None:
        return None
    ok = np.isfinite(ref) & np.isfinite(hr)
    if ok.sum() < 5:
        return None
    e = np.abs(hr[ok] - ref[ok])
    return dict(name=rec["name"], mae=float(e.mean()), n=int(ok.sum()),
                has_gyro=bool(rec["meta"].get("gyro") is not None))


def _ensure():
    global P_
    if "P_" not in globals():
        P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    return P_


def _worker(payload):
    P_ = _ensure()
    import ext_loaders as EL
    ds = payload["ds"]
    if ds == "wrist":
        recs = list(EL.iter_wrist(fs_out=FS)); files = None
    elif ds == "dalia":
        recs = list(EL.iter_dalia(fs_out=FS)); files = None
    elif ds == "huawei":
        root = os.path.join(EL.datasets_root(), EL.HUAWEI_DIRNAME)
        files = sorted(glob.glob(os.path.join(root, "*.zip"))); recs = None
    else:
        CDE = load_module("cde", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
        recs, _ha = CDE.iter_dataset(EL, ds); files = None
    # ⚠️ PAR.pmap 会拍平返回值（dict 会被拆成键）→ 必须返回 list
    out = []
    for i in payload["idx"]:
        if recs is None:
            if i >= len(files):
                continue
            rec = EL.load_huawei(os.path.basename(files[i]), root=None, fs_out=FS)
        else:
            if i >= len(recs):
                continue
            rec = recs[i]
        for p in payload["picks"]:
            r = _series(P_, rec, p)
            if r:
                r["pick"] = p
                out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description="RLS 参考通道消融")
    ap.add_argument("--datasets", nargs="+", default=["troika", "wesad", "wrist", "dalia"])
    ap.add_argument("--picks", nargs="+",
                    default=["acc1", "acc2", "acc3", "gyro1", "gyro3", "pair_xy", "pair_ag",
                             "acc3g1", "acc3g3"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=P.artifact("ref_ablation"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    PAR.log_env()

    summary = {}
    for ds in args.datasets:
        import ext_loaders as EL
        if ds == "wrist":
            n = len(list(EL.iter_wrist(fs_out=FS)))
        elif ds == "dalia":
            n = len(list(EL.iter_dalia(fs_out=FS)))
        elif ds == "huawei":
            n = len(glob.glob(os.path.join(EL.datasets_root(), EL.HUAWEI_DIRNAME, "*.zip")))
        else:
            CDE = load_module("cde", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
            n = len(CDE.iter_dataset(EL, ds)[0])
        if args.limit:
            n = min(n, args.limit)
        chunks = PAR.split_chunks(list(range(n)), PAR.default_workers(n, reserve=4, cap=16))
        res, dt = PAR.pmap(_worker, [dict(ds=ds, idx=c, picks=args.picks) for c in chunks],
                           desc="%s 参考消融（%d 条）" % (ds, n))
        merged = {p: [] for p in args.picks}
        for r in res:
            if isinstance(r, dict) and "pick" in r:
                merged[r["pick"]].append(r)
        rows = {}
        has_gyro = any(x.get("has_gyro") for p in merged for x in merged[p][:1])
        for p in args.picks:
            v = merged[p]
            if not v:
                continue
            e = np.array([x["mae"] for x in v]); w = np.array([x["n"] for x in v])
            rows[p] = dict(pooled=round(float((e * w).sum() / w.sum()), 3),
                           median=round(float(np.median(e)), 3),
                           mean=round(float(e.mean()), 3), n_rec=len(v), n_win=int(w.sum()))
        summary[ds] = dict(n_records=n, seconds=round(dt, 1), has_gyro=has_gyro, rows=rows)
        print("\n[ref] %s（%d 条%s）" % (ds, n, "，含 Gyro" if has_gyro else "，无 Gyro"))
        print("      %-8s %10s %10s %10s" % ("参考", "池化MAE", "记录中位", "记录均值"))
        base = rows.get("acc3", {}).get("pooled")
        for p in args.picks:
            if p in rows:
                d = rows[p]
                mark = ""
                if base and p != "acc3":
                    mark = "  (%+.0f%%)" % (100 * (d["pooled"] - base) / base)
                print("      %-8s %10.2f %10.2f %10.2f%s"
                      % (p, d["pooled"], d["median"], d["mean"], mark))
    json.dump(dict(protocol="参考通道消融：RLS 的 acc/gyro 参考组合；池化=按窗加权，"
                            "记录中位/均值为记录级；真值口径与该域既有实验一致",
                   summary=summary),
              open(os.path.join(args.out, "ref_ablation.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[ref] 完成 -> %s" % args.out)
    return summary


if __name__ == "__main__":
    main()

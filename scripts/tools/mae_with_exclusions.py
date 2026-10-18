#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E6 记录级 MAE 与"剔除指定记录后"的域均值（用于处理存疑记录）。

背景
----
`reports/17` §3.1 的四方交叉审核发现 CapnoBase 的 `0329_8min` **ECG 信号质量太差**
（R 峰仅 164 个，而同域其他记录 531–588 个）→ 以 ECG 派生 HR 作参考时该条不可信；
它单独占了该域修正后总误差的约 **49%**。本工具给出"含/不含该记录"的两个口径。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/mae_with_exclusions.py --dataset capnobase --exclude 0329_8min
    $PY scripts/tools/mae_with_exclusions.py --dataset bidmc
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


def _worker(payload):
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    CDE = load_module("cde", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
    import ext_loaders as EL
    ds = payload["ds"]
    recs, has_acc = CDE.iter_dataset(EL, ds)
    out = []
    for i in payload["idx"]:
        rec = recs[i]
        prep = CDE.prepare(P_, rec, has_acc)
        if prep is None:
            continue
        ppg, m_hat, accmag, ref, n_win, _ = prep
        sig = (ppg - m_hat) if has_acc else ppg
        f, Pm = P_.psd_matrix(np.asarray(sig, float), accmag, FS, n_win,
                              win_s=8.0, hop_s=2.0, use_notch=False)
        hr = P_.sel_viterbi(f, Pm, band=tuple(payload["band"]), jump_bpm=payload["jump"],
                            lam=payload["lam"])
        n = min(len(hr), len(ref))
        ok = np.isfinite(hr[:n]) & np.isfinite(ref[:n])
        if ok.sum() < 5:
            continue
        e = np.abs(hr[:n][ok] - ref[:n][ok])
        out.append(dict(name=rec["name"], n=int(ok.sum()), mae=float(e.mean()),
                        sum=float(e.sum())))
    return out


def main():
    ap = argparse.ArgumentParser(description="E6 记录级 MAE 与剔除后均值")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--jump", type=float, default=4.0)
    ap.add_argument("--band", default="0.7,3.0")
    ap.add_argument("--out", default=P.artifact("exclusions"))
    args = ap.parse_args()
    band = tuple(float(x) for x in args.band.split(","))
    os.makedirs(args.out, exist_ok=True)
    PAR.log_env()

    import ext_loaders as EL
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    CDE = load_module("cde", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
    recs, _ = CDE.iter_dataset(EL, args.dataset)
    chunks = PAR.split_chunks(list(range(len(recs))), PAR.default_workers(len(recs), reserve=4))
    rows, _ = PAR.pmap(_worker, [dict(ds=args.dataset, idx=c, lam=args.lam, jump=args.jump,
                                      band=band) for c in chunks],
                       desc="%s 记录级 MAE" % args.dataset)

    tot = sum(r["sum"] for r in rows); nn = sum(r["n"] for r in rows)
    keep = [r for r in rows if r["name"] not in set(args.exclude)]
    tot_k = sum(r["sum"] for r in keep); nn_k = sum(r["n"] for r in keep)
    all_mae = tot / max(nn, 1); keep_mae = tot_k / max(nn_k, 1)
    # 去掉异常记录后的稳健统计
    arr = np.array([r["mae"] for r in rows])
    print("\n[E6] %s：%d 条记录 / %d 窗" % (args.dataset, len(rows), nn))
    print("      全量         MAE **%.3f**" % all_mae)
    print("      剔除 %-18s MAE **%.3f**（窗 %d）" % (",".join(args.exclude) or "（无）", keep_mae, nn_k))
    print("      记录级 MAE：中位 %.3f | 均值 %.3f | 最差 %.2f（%s）"
          % (float(np.median(arr)), float(arr.mean()), float(arr.max()),
             max(rows, key=lambda r: r["mae"])["name"]))
    top = sorted(rows, key=lambda r: -r["mae"])[:5]
    for r in top:
        print("        最差：%-14s MAE %6.2f（%d 窗，占该域总误差 %.1f%%）"
              % (r["name"], r["mae"], r["n"], 100 * r["sum"] / max(tot, 1e-9)))
    out = dict(dataset=args.dataset, n_records=len(rows), n_windows=nn,
               mae_all=round(all_mae, 3), exclude=args.exclude, mae_excluded=round(keep_mae, 3),
               n_windows_excluded=nn_k, per_record=rows)
    with open(os.path.join(args.out, "excl_%s.json" % args.dataset), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[E6] 完成 -> %s" % args.out)
    return out


if __name__ == "__main__":
    main()

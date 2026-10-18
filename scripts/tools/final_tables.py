#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最终分层总表：按"运动强度分层"汇报 + 配对 bootstrap 置信区间（批次 2c）。

为什么要分层
------------
池化均值会被少数灾难记录主导（例：PPG-DaLiA 池化 11.66 而**记录中位仅 5.56**；
WRIST 池化 16.03 而记录中位 10.46）。只报池化值容易被误读为"整体很差"，
而真正的结构是**双峰**：低运动极好 / 中高强度很差。因此本工具同时给出：

- **池化**（按窗加权）与**记录中位**；
- **分层**：按活动/域的性质分为 `低运动`（静息/办公/驾驶/坐）与 `中高强度`（走/跑/爬楼/骑行/球类）；
- **配对 bootstrap 95% CI**（对记录重采样 5000 次，报池化值的不确定度）。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/final_tables.py --out outputs/final_tables
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
from _common import paths as P                                # noqa: E402

# 活动 → 强度分层
LOW = {"sitting", "working", "car_driving", "lunch_break", "lying", "tv", "driving"}
HIGH = {"walking", "walk", "run", "stair_climbing", "stairs", "cycling", "bike_low",
        "bike_high", "table_soccer",
        # WRIST 的骑行活动名带 resistance（历史落在「其他」，见 reports/41 审计 P2-1）
        "high_resistance_bike", "low_resistance_bike"}


DOMAIN_STRATUM = {"bidmc": "静息临床", "capnobase": "静息临床", "troika": "跑步·指夹",
                  "wesad": "腕部日常（低伪影）"}


def _stratum(ds, name):
    """活动分层：WRIST 活动在**后缀**（s1_walk），PPG-DaLiA 在**前缀**（walking_S5）。"""
    act = None
    if ds == "wrist":
        act = name.split("_", 1)[1] if "_" in name else name
    elif ds == "dalia":
        act = name.rsplit("_", 1)[0]
    if act in HIGH:
        return "中高强度"
    if act in LOW:
        return "低运动"
    return DOMAIN_STRATUM.get(ds, "其他")


def boot_ci(maes, ns, n=5000, seed=0):
    """对**记录**重采样 → 池化 MAE 的 95% CI。"""
    maes = np.asarray(maes, float); ns = np.asarray(ns, float)
    rng = np.random.default_rng(seed)
    m = len(maes)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        vals[i] = (maes[idx] * ns[idx]).sum() / ns[idx].sum()
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser(description="最终分层总表 + bootstrap CI")
    ap.add_argument("--dir", default=P.artifact("exclusions"))
    ap.add_argument("--out", default=P.artifact("final_tables"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.dir, "excl_*.json"))) + \
        sorted(glob.glob(P.artifact("dalia_eval/excl_*.json")))
    table = {}
    print("%-12s %-10s %6s %8s %10s %10s %20s" %
          ("域", "分层", "条数", "窗数", "池化MAE", "记录中位", "95% CI（池化）"))
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        ds = d["dataset"]
        rows = [r for r in d["per_record"] if isinstance(r, dict)]
        table[ds] = {}
        strata = ["全部"] + sorted({_stratum(ds, r["name"]) for r in rows})
        for stratum in strata:
            if stratum == "全部":
                sub = rows
            else:
                sub = [r for r in rows if _stratum(ds, r["name"]) == stratum]
            if not sub:
                continue
            m = np.array([r["mae"] for r in sub]); n = np.array([r["n"] for r in sub])
            pooled = float((m * n).sum() / n.sum())
            lo, hi = boot_ci(m, n)
            table[ds][stratum] = dict(n_rec=len(sub), n_win=int(n.sum()),
                                      pooled=round(pooled, 3),
                                      median=round(float(np.median(m)), 3),
                                      ci95=[round(lo, 3), round(hi, 3)])
            print("%-12s %-10s %6d %8d %10.2f %10.2f   [%.2f, %.2f]"
                  % (ds, stratum, len(sub), int(n.sum()), pooled, np.median(m), lo, hi))
    json.dump(dict(protocol="分层口径：静息临床（无活动标签域）/低运动/中高强度；"
                            "池化按窗加权；CI=对记录重采样 5000 次的 2.5–97.5 分位",
                   strata=dict(low=sorted(LOW), high=sorted(HIGH)), table=table),
              open(os.path.join(args.out, "final_tables.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[tables] 完成 -> %s" % args.out)
    return table


if __name__ == "__main__":
    main()

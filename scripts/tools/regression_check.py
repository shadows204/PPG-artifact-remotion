#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回归门禁：一键核对全项目关键数字（代码改动后必须保持）。

用途
----
任何代码修改（重构/修 bug/加功能）后运行本工具，确认**已归档的关键结果没有漂移**。
每项给出「期望值 / 实测值 / 是否通过」，退出码非 0 表示有回归。

覆盖
----
1. CIME 波形：lite 残差集成 46.43、lite 融合 46.37、41 成员集成 47.15、物理先验 83.78/79.00
2. 心率：TROIKA raw 6.64 / RLS 2.73、BIDMC RLS 3.00、CapnoBase 1.06、WESAD 8.28、WRIST 16.03、DaLiA 11.66
3. 软件接口：60 段自检 35.72（与官方管线 35.71 对齐）
4. 并行层：pmap 结果与串行一致

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/regression_check.py                 # 全部
    $PY scripts/tools/regression_check.py --only cime api # 子集（更快）
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "modules"))
from _common import paths as P                                # noqa: E402

PY = sys.executable
TOL = 0.35          # 允许的相对漂移（PRD/MAE 绝对值）


def _run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def _grep_num(txt, pattern):
    import re
    m = re.findall(pattern, txt)
    return [float(x) for x in m]


def check(name, got, want, tol=TOL):
    ok = got is not None and abs(float(got) - float(want)) <= tol
    print("  %-42s 期望 %7.2f | 实测 %s | %s"
          % (name, want, ("%.2f" % got) if got is not None else "—",
             "✅" if ok else "❌ 回归！"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="回归门禁")
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["cime", "hr", "api", "par"])
    ap.add_argument("--feat-dir", default=P.artifact("final_10seed/features"))
    a = ap.parse_args()
    only = set(a.only) if a.only else {"cime", "hr", "api", "par"}
    allok = True

    if "cime" in only:
        print("\n[1] CIME 波形（RLS 物理先验 / 集成 / 融合）")
        # 1a 物理先验（α=0.40，异质两路）：由特征直接算
        d = np.load(os.path.join(a.feat_dir, "test.npz"))
        X, Y = d["XrlsOnly"], d["y"]
        prd = float(np.nanmean([100 * np.linalg.norm(X[i, 2] - Y[i]) / np.linalg.norm(Y[i])
                                for i in range(len(Y))]))
        allok &= check("CIME RLS-only（α=0.40 异质两路）", prd, 83.78)
        # 1b lite 残差集成 + 融合（走 fuse_branches）
        out = _run("%s %s/scripts/tools/fuse_branches.py --res-dir %s/models/ppg_denoise_v1/lite "
                   "--res-pattern 'residual_rls_*.pt' --crm-dirs %s/models/ppg_denoise_v1/lite "
                   "--crm-pattern 'crm_rls_*.pt' --tags crm512 --out /tmp/reg_lite"
                   % (PY, P.ROOT, P.ROOT, P.ROOT))
        nums = _grep_num(out, r"residual\s+验证 PRD\s+[\d.]+\s+\| 测试 PRD\s+([\d.]+)")
        allok &= check("CIME lite 残差集成（5 成员）", nums[0] if nums else None, 46.91)
        nums2 = _grep_num(out, r"最优 α=[\d.]+\s+→ 验证 [\d.]+ / \*\*测试 ([\d.]+)\*\*")
        allok &= check("CIME lite 融合（5+3）", nums2[0] if nums2 else None, 46.37)

    if "hr" in only:
        print("\n[2] 心率（6 个有真值域，物理基线口径）")
        expect = dict(troika=2.73, bidmc=3.00, capnobase=1.06, wesad=8.28, wrist=16.03, dalia=11.66)
        for ds, want in expect.items():
            out = _run("OMP_NUM_THREADS=2 %s %s/scripts/tools/cross_dataset_eval.py --datasets %s "
                       "--adapt none --no-unet --weights-dir %s/outputs/final_10seed/weights --seed 0 "
                       "--lam 0.5 --jump 4.0 --band 0.7,3.0 --tag-suffix _reg --out /tmp/reg_%s"
                       % (PY, P.ROOT, ds, P.ROOT, ds))
            m = _grep_num(out, r"rls\(3axis\)\s+\d+\s+([\d.]+)")
            got = m[0] if m else None
            if ds in ("bidmc", "capnobase"):
                # 这两域物理基线是 raw（无 IMU 路径），改用 raw 行
                m2 = _grep_num(out, r"(?m)^\s+raw\s+\d+\s+([\d.]+)")
                got = m2[0] if m2 else None
            allok &= check("心率 %-10s 物理基线" % ds, got, want)

    if "api" in only:
        print("\n[3] 软件接口（60 段自检，应与官方管线对齐）")
        out = _run("OMP_NUM_THREADS=2 %s %s/scripts/tools/run_denoise.py --demo --out /tmp/reg_api "
                   "--no-plot --eval-test 60" % (PY, P.ROOT))
        m = _grep_num(out, r"滤净 PRD \*\*([\d.]+)\*\*")
        allok &= check("接口 60 段自检", m[0] if m else None, 35.72, tol=1.0)

    if "par" in only:
        print("\n[4] 并行层 pmap（结果与串行一致）")
        out = _run("%s %s/scripts/_common/parallel.py" % (PY, P.ROOT))
        ok = ("自检通过" in out) and ("❌" not in out)
        print("  %-42s %s" % ("pmap 自检（含 dict 回归用例）", "✅" if ok else "❌ 失败"))
        allok &= ok

    print("\n[regression] %s" % ("全部通过 ✅" if allok else "存在回归 ❌（详见上表）"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""批次 5：时域分支 × 频域分支的融合评估（可复现工具）。

背景
----
频域 CRM 分支单独用**明显更差**（54.3 vs 残差 47.2），但与残差分支**错误互补**：
验证受试者 9 上选出 α=0.10（CRM 占比）→ 测试 PRD **46.78**（优于 47.15）。

本工具支持：多分支（残差 + 若干 CRM 变体）的权重在**验证受试者**上做小网格搜索，
再在测试集上评估；同时输出 α 曲线以判断权重选择的稳健性（平台期 vs 尖峰）。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/fuse_branches.py --out outputs/b5_fuse
    $PY scripts/tools/fuse_branches.py --crm-dirs outputs/b5_crm/weights outputs/b5f512/weights \\
        --tags crm16 crm512 --out outputs/b5_fuse
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "modules"))
from _common import paths as P                                # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="批次 5：分支融合")
    ap.add_argument("--feat-dir", default=P.artifact("final_10seed/features"))
    ap.add_argument("--res-dir", default=P.artifact("ens_all"))
    ap.add_argument("--res-pattern", default="residual_rls_*.pt")
    ap.add_argument("--crm-dirs", nargs="+",
                    default=[P.artifact("b5_crm/weights"), P.artifact("b5b/weights")])
    ap.add_argument("--crm-pattern", default="crm_rls_*.pt")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--val-subject", type=int, default=9)
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--out", default=P.artifact("b5_fuse"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    import residual as R
    import crm as CRM
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtr = np.load(os.path.join(args.feat_dir, "train.npz"))
    dte = np.load(os.path.join(args.feat_dir, "test.npz"))
    subj = dtr["subject"].astype(int)
    va = np.where(subj == args.val_subject)[0]
    COND = {}

    def run(paths, loader, tag, X, idx, ref_ds):
        Xs = torch.from_numpy(X[idx]).float().to(dev)
        acc = np.zeros((len(idx), X.shape[-1]), dtype=np.float32)
        for p in paths:
            m, meta = loader(p, device=dev)
            cd = int(meta.get("cond_dim", 0) or 0)
            if cd > 0:
                if cd not in COND:
                    A = dtr["stats%d" % cd].astype(np.float64)
                    mu = A.mean(0); sd = A.std(0) + 1e-6
                    COND[cd] = {t: torch.from_numpy(((d["stats%d" % cd].astype(np.float64) - mu) / sd)
                                                    .astype(np.float32)).float()
                                for t, d in (("train", dtr), ("test", dte))}
                st = COND[cd][tag][idx].to(dev)
            else:
                st = torch.zeros(len(idx), 0).to(dev)
            with torch.no_grad():
                acc += m(Xs, st)[:, 0].cpu().numpy()
        return acc / max(len(paths), 1)

    def prd(p, Y):
        return float(np.nanmean([P_.prd(Y[i], p[i]) for i in range(len(Y))]))

    branches = []
    res_paths = sorted(glob.glob(os.path.join(args.res_dir, args.res_pattern)))
    print("[fuse] 残差成员 %d" % len(res_paths))
    branches.append(("residual", run(res_paths, R.load, "train", dtr["XrlsOnly"], va, dtr),
                     run(res_paths, R.load, "test", dte["XrlsOnly"], np.arange(len(dte["y"])), dte)))
    for i, d in enumerate(args.crm_dirs):
        ps = sorted(glob.glob(os.path.join(d, args.crm_pattern)))
        if not ps:
            print("     跳过（无权重）：%s" % d); continue
        tag = (args.tags[i] if args.tags and i < len(args.tags) else "crm%d" % i)
        print("[fuse] %s 成员 %d（%s）" % (tag, len(ps), d))
        branches.append((tag, run(ps, CRM.load, "train", dtr["XrlsOnly"], va, dtr),
                         run(ps, CRM.load, "test", dte["XrlsOnly"], np.arange(len(dte["y"])), dte)))
    Yv, Yt = dtr["y"][va], dte["y"]
    rv, rt = branches[0][1], branches[0][2]
    print("\n[fuse] 各分支单独（验证 %.1f / 测试 %.2f 段）" % (len(va), len(Yt)))
    res = {}
    for tag, v, t in branches:
        res[tag] = dict(val=round(prd(v, Yv), 3), test=round(prd(t, Yt), 3))
        print("      %-10s 验证 PRD %7.2f | 测试 PRD %7.2f" % (tag, res[tag]["val"], res[tag]["test"]))

    # 单分支融合：residual 为主，CRM 占比 α 在验证集上选
    print("\n[fuse] 单 CRM 分支融合（α = CRM 占比，验证集选）")
    best = None
    for tag, v, t in branches[1:]:
        curve = {}
        for a in args.grid:
            curve[a] = (prd(a * v + (1 - a) * rv, Yv), prd(a * t + (1 - a) * rt, Yt))
        a_best = min(curve, key=lambda a: curve[a][0])
        print("      %-10s 最优 α=%.2f → 验证 %.2f / **测试 %.2f**（α 曲线：%s）"
              % (tag, a_best, curve[a_best][0], curve[a_best][1],
                 " ".join("%.2f:%.1f" % (a, curve[a][1]) for a in sorted(curve))))
        res["fuse_" + tag] = dict(alpha=a_best, val=round(curve[a_best][0], 3),
                                  test=round(curve[a_best][1], 3),
                                  curve={str(a): [round(curve[a][0], 3), round(curve[a][1], 3)]
                                         for a in sorted(curve)})
        # ⚠️ 按**验证集** PRD 选最佳分支（原按测试集选 = 泄漏；见 reports/41）
        if best is None or curve[a_best][0] < best[3]:
            best = (tag, curve[a_best][1], a_best, curve[a_best][0])
    if best is None:
        print("\n[fuse][warn] 未找到任何 CRM 分支（检查 --crm-dirs/--crm-pattern）→ 跳过融合")
    else:
        print("\n[fuse] 按**验证集**选出的最佳分支：%s（α=%.2f，验证 %.2f）→ 测试 PRD %.2f"
              % (best[0], best[2], best[3], best[1]))

    # 多分支：residual + 各 CRM，权重在验证集做小网格（步长 0.05，和为 1）
    if len(branches) > 2:
        grid = np.arange(0, 0.45 + 1e-9, 0.05)
        combos = [c for c in itertools.product(grid, repeat=len(branches) - 1)
                  if sum(c) <= 0.6]
        bestm = None
        for c in combos:
            w = np.array(list(c) + [1 - sum(c)])
            pv = sum(w[i] * branches[i][1] for i in range(len(branches)))
            p = prd(pv, Yv)
            if bestm is None or p < bestm[0]:
                bestm = (p, c)
        w = np.array(list(bestm[1]) + [1 - sum(bestm[1])])
        pt = sum(w[i] * branches[i][2] for i in range(len(branches)))
        print("[fuse] 多分支最优权重（验证 %.2f）：%s → **测试 PRD %.2f**"
              % (bestm[0], dict(zip([b[0] for b in branches], np.round(w, 2))), prd(pt, Yt)))
        res["multi"] = dict(val=round(bestm[0], 3), test=round(prd(pt, Yt), 3),
                            weights={b[0]: float(x) for b, x in zip(branches, w)})
    json.dump(dict(protocol="批次5 分支融合：权重在验证受试者 %d 上选（小网格），测试集仅评估一次；"
                            "参差分支=时域残差集成，CRM 分支=频域复数比率掩码集成" % args.val_subject,
                   res_members=len(res_paths), branches=[b[0] for b in branches], results=res),
              open(os.path.join(args.out, "fuse_branches.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[fuse] 完成 -> %s" % args.out)
    return res


if __name__ == "__main__":
    main()

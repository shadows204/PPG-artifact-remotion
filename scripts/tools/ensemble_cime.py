#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E2 域内波形集成（CIME）：把同一配置的多种子残差模型在**波形层面**平均。

动机
----
`reports/21` 已证 15 种子 InstanceNorm 的单模型均值 **49.54 ± 0.86 PRD**（BN 51.62 ± 1.40），
而域内上界（用测试集自助训练）是 44.87 → 还有 ~4.7 PRD 空间。多种子波形集成是最便宜的常规增益
（**不需要任何训练**，只做前向平均）。

与训练协议严格一致
------------------
特征取自 `--feat-dir` 的 `test.npz`：`X = dte[key]`（按段标准化后的输入）、`Y = dte["y"]`（标准化参考）；
指标用 `pipeline.prd` / `snr_db` / Pearson，与 `train_residual.py` 的 test 报告口径相同。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/ensemble_cime.py --weights-dir outputs/norm_ablation/v15/weights \\
        --pattern "residual_rls_p3vin_s*.pt" --key XrlsOnly --base-index 2 --out outputs/ens_cime
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="E2 域内波形集成")
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--feat-dir", default=P.artifact("final_10seed/features"))
    ap.add_argument("--key", default="XrlsOnly")
    ap.add_argument("--base-index", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=P.artifact("ens_cime"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    import residual as R
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dte = np.load(os.path.join(args.feat_dir, "test.npz"))
    X = dte[args.key]; Y = dte["y"]
    paths = sorted(glob.glob(os.path.join(args.weights_dir, args.pattern)))
    if not paths:
        print("未找到权重：%s/%s" % (args.weights_dir, args.pattern)); return
    print("[E2] 成员 %d 个 | 测试段 %d | 输入键 %s | 先验通道 %d"
          % (len(paths), len(X), args.key, args.base_index))

    Xt = torch.from_numpy(X).float()
    zero = torch.zeros(args.batch, 0)
    # ⚠️ FiLM 条件模型（cond_dim>0）需要条件向量：用 checkpoint 里存的 mu/sd 复原（保证可复现）
    COND = {}
    outs = []
    per_model = []
    for p in paths:
        model, meta = R.load(p, device=dev)
        cd = int(meta.get("cond_dim", 0) or 0)
        S = None
        if cd > 0:
            if cd not in COND:
                A = dte["stats%d" % cd].astype(np.float64)
                if meta.get("cond_mu") is not None:
                    mu = np.asarray(meta["cond_mu"], float)
                    sd = np.asarray(meta["cond_sd"], float)
                else:   # 旧 checkpoint 未存 → 用训练集统计复算（与训练时完全一致）
                    T = np.load(os.path.join(args.feat_dir, "train.npz"))["stats%d" % cd]
                    mu = T.astype(np.float64).mean(0); sd = T.astype(np.float64).std(0) + 1e-6
                COND[cd] = torch.from_numpy(((A - mu) / sd).astype(np.float32)).float()
            S = COND[cd]
        pred = np.zeros((len(X), X.shape[-1]), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(X), args.batch):
                xb = Xt[s:s + args.batch].to(dev)
                st = (S[s:s + len(xb)].to(dev) if S is not None else zero[:len(xb)].to(dev))
                out = model(xb, st)[:, 0].cpu().numpy()
                pred[s:s + len(xb)] = out
        outs.append(pred)
        prd = float(np.nanmean([P_.prd(Y[i], pred[i]) for i in range(len(Y))]))
        pear = float(np.nanmean([np.corrcoef(Y[i], pred[i])[0, 1] for i in range(len(Y))]))
        snr = float(np.nanmean([P_.snr_db(pred[i], Y[i]) for i in range(len(Y))]))
        per_model.append(dict(name=os.path.basename(p), norm=str(meta.get("norm", "bn")),
                              PRD=round(prd, 3), Pearson=round(pear, 4), SNR=round(snr, 3)))
        print("  %-34s norm=%-3s PRD %6.2f  Pearson %.4f  SNR %5.2f"
              % (os.path.basename(p), meta.get("norm", "bn"), prd, pear, snr))

    O = np.stack(outs)                       # (M, N, L)
    res = {}
    for tag, pred in (("mean", O.mean(0)), ("median", np.median(O, 0))):
        prd = float(np.nanmean([P_.prd(Y[i], pred[i]) for i in range(len(Y))]))
        pear = float(np.nanmean([np.corrcoef(Y[i], pred[i])[0, 1] for i in range(len(Y))]))
        snr = float(np.nanmean([P_.snr_db(pred[i], Y[i]) for i in range(len(Y))]))
        res[tag] = dict(PRD=round(prd, 3), Pearson=round(pear, 4), SNR=round(snr, 3))
        print("  [集成-%s] PRD **%.2f**  Pearson %.4f  SNR %.2f dB" % (tag, prd, pear, snr))

    single = [m["PRD"] for m in per_model]
    print("\n[E2] 单模型均值 %.2f ± %.2f（n=%d）| 集成均值 %.2f | 集成中位 %.2f"
          % (float(np.mean(single)), float(np.std(single)), len(single),
             res["mean"]["PRD"], res["median"]["PRD"]))
    with open(os.path.join(args.out, "ensemble_cime.json"), "w", encoding="utf-8") as f:
        json.dump(dict(protocol="E2 域内波形集成：特征/指标与 train_residual 的 test 报告同口径；"
                                "集成为多种子输出的逐点平均/中位",
                       weights_dir=args.weights_dir, pattern=args.pattern, key=args.key,
                       n_members=len(paths), per_model=per_model,
                       single_mean=round(float(np.mean(single)), 3),
                       single_std=round(float(np.std(single)), 3), ensemble=res),
                  f, ensure_ascii=False, indent=2)
    print("[E2] 完成 -> %s" % args.out)
    return res


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E5 WESAD「RLS + 模型」波形混合的严格留出验证。

背景
----
`reports/21` §4.2 观测到 WESAD 上"RLS + 模型"波形混合可达 **6.87–7.01**（vs RLS 8.28 / raw 9.09），
但那次是在**全部 15 名受试者**上、且混合权重未经留出确认。本工具做严格版：

- **前 12 名受试者**：用于选择混合权重 w（RLS 占比 1−w）；
- **后 3 名受试者**：只用一次评估（不做选择）。

同时报告三种基线（raw / RLS / 纯模型）在同一 3 人上的成绩，避免"用全量数字误导"。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/blend_holdout.py --weights-dir outputs/norm_ablation/weights \\
        --pattern "residual_rls_p3gn_s[0-2].pt" --out outputs/blend_holdout
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


def _hr(P_, sig, accmag, n_win, lam, jump, wvec=None):
    f, Pm = P_.psd_matrix(np.asarray(sig, float), accmag, FS, n_win, win_s=8.0, hop_s=2.0,
                          use_notch=False)
    return P_.sel_viterbi(f, Pm, band=BAND, jump_bpm=jump, lam=lam)


def _worker(payload):
    """对一条 WESAD 记录：返回 raw/RLS/模型/各权重混合在不同窗上的 HR 与参考。"""
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    CDE = load_module("cde", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
    import residual as R
    import torch
    import ext_loaders as EL
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = list(EL.iter_wesad(fs_out=FS))
    nets = []
    for pth in payload["weights"]:
        m, _ = R.load(pth, device=dev)
        nets.append(m)
    out = []
    for i in payload["idx"]:
        rec = recs[i]
        prep = CDE.prepare(P_, rec, True)
        if prep is None:
            continue
        ppg, m_hat, accmag, ref, n_win, _ = prep
        chans = CDE.raw_channels(ppg, m_hat, "rls_only")
        outs = [CDE.reconstruct_batch(m, dev, chans) for m in nets]
        mdl = np.mean(outs, axis=0)
        rls = np.asarray(ppg - m_hat, float)
        ref = np.asarray(ref, float)
        row = dict(name=rec["name"], n=int(n_win), hr={}, ref=ref.tolist())
        row["hr"]["raw"] = _hr(P_, ppg, accmag, n_win, payload["lam"], payload["jump"]).tolist()
        row["hr"]["rls"] = _hr(P_, rls, accmag, n_win, payload["lam"], payload["jump"]).tolist()
        row["hr"]["mdl"] = _hr(P_, mdl, accmag, n_win, payload["lam"], payload["jump"]).tolist()
        for w in payload["weights_w"]:
            mix = (1.0 - w) * rls + w * mdl
            row["hr"]["mix%.2f" % w] = _hr(P_, mix, accmag, n_win,
                                           payload["lam"], payload["jump"]).tolist()
        out.append(row)
    return out


def mae(hr, ref):
    hr = np.asarray(hr); ref = np.asarray(ref)
    ok = np.isfinite(hr) & np.isfinite(ref)
    return float(np.mean(np.abs(hr[ok] - ref[ok]))) if ok.sum() else float("nan")


def main():
    ap = argparse.ArgumentParser(description="E5 WESAD 混合权重的严格留出验证")
    ap.add_argument("--weights-dir", default=P.artifact("norm_ablation/weights"))
    ap.add_argument("--pattern", default="residual_rls_p3gn_s[0-2].pt")
    ap.add_argument("--weights-w", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--n-select", type=int, default=12, help="前 N 人用于选权重（--folds 1 时生效）")
    ap.add_argument("--folds", type=int, default=1,
                    help="受试者 K 折交叉验证：每折在其余人上选 w，只在该折上评估（默认 1=单次划分）")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--jump", type=float, default=4.0)
    ap.add_argument("--out", default=P.artifact("blend_holdout"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    PAR.log_env()

    import ext_loaders as EL
    n_all = len(list(EL.iter_wesad(fs_out=FS)))
    weights = sorted(glob.glob(os.path.join(args.weights_dir, args.pattern)))
    print("[E5] WESAD 共 %d 人 | 模型成员 %d 个 | 选权重用前 %d 人、评估用后 %d 人"
          % (n_all, len(weights), args.n_select, n_all - args.n_select))
    sel_idx = list(range(args.n_select)); ev_idx = list(range(args.n_select, n_all))
    rows = {}
    for tag, idxs in (("select", sel_idx), ("holdout", ev_idx)):
        chunks = PAR.split_chunks(idxs, PAR.default_workers(len(idxs), reserve=4))
        r, _ = PAR.pmap(_worker, [dict(idx=c, weights=weights, weights_w=args.weights_w,
                                       lam=args.lam, jump=args.jump) for c in chunks],
                        desc="WESAD %s（%d 人）" % (tag, len(idxs)))
        rows[tag] = r
    keys = ["raw", "rls", "mdl"] + ["mix%.2f" % w for w in args.weights_w if 0 < w < 1]

    def agg(rows_sub, k):
        return float(np.nanmean([mae(r["hr"][k], r["ref"]) for r in rows_sub]))

    if args.folds > 1:
        # ---- 受试者 K 折交叉验证（每折独立选 w，仅在该折评估）----
        allrows = rows["select"] + rows["holdout"]
        order = list(range(len(allrows)))
        folds = [list(f) for f in np.array_split(order, args.folds)]
        print("\n[E5] 受试者 %d 折交叉验证（每折 %d 人评估、其余人选 w）"
              % (args.folds, len(folds[0])))
        print("      折  | 选出 w | 评估集 MAE(选中 w) | 评估集 RLS | 差值")
        per, diffs, wsell = [], [], []
        for i, ev in enumerate(folds):
            sel = [j for j in order if j not in ev]
            evr = [allrows[j] for j in ev]; slr = [allrows[j] for j in sel]
            wkey = min(keys[3:], key=lambda k: agg(slr, k))      # 在选择集上选最优混合权重
            m_sel = agg(evr, wkey); m_rls = agg(evr, "rls")
            per.append(m_sel); diffs.append(m_sel - m_rls); wsell.append(wkey)
            print("      %2d   | %-7s | %8.2f | %8.2f | %+6.2f"
                  % (i + 1, wkey, m_sel, m_rls, m_sel - m_rls))
        per = np.array(per); diffs = np.array(diffs)
        tab = {"kfold": dict(folds=args.folds, per_fold=[round(float(x), 3) for x in per],
                             selected_ws=wsell,
                             mean=round(float(per.mean()), 3),
                             std=round(float(per.std(ddof=1)), 3),
                             mean_vs_rls=round(float(diffs.mean()), 3),
                             std_vs_rls=round(float(diffs.std(ddof=1)), 3),
                             n_folds_improved=int((diffs < 0).sum()))}
        print("      —— %d 折：选中 w 的 MAE **%.2f ± %.2f**；相对 RLS 改善 **%+.2f ± %.2f**（%d/%d 折改善）"
              % (args.folds, per.mean(), per.std(ddof=1), diffs.mean(), diffs.std(ddof=1),
                 int((diffs < 0).sum()), args.folds))
    else:
        print("\n[E5] 各方案（同一划分）")
        print("      %-10s %10s %10s" % ("方案", "前%d人(选)" % args.n_select, "**后%d人(评估)**"
                                          % (n_all - args.n_select)))
        tab = {}
        for k in keys:
            a = agg(rows["select"], k); b = agg(rows["holdout"], k)
            tab[k] = dict(select=round(a, 3), holdout=round(b, 3))
            print("      %-10s %10.2f %10.2f" % (k, a, b))
        best_w = min(keys[3:], key=lambda k: tab[k]["select"])
        print("\n[E5] 前 %d 人选出的最优权重：%s（选择集 %.2f）→ **留出 %.2f**（RLS 基线 %.2f）"
              % (args.n_select, best_w, tab[best_w]["select"], tab[best_w]["holdout"],
                 tab["rls"]["holdout"]))
        tab = {"single_split": tab, "selected": best_w}
    # ⚠️ 报告口径：K 折时 headline 必须是**折均值**，不是某一折（历史 6.02 就是最后一折，
    #    导致 WESAD 被高估 1.13 bpm；见 reports/41 审计 P0-2）。
    kfold = bool(args.folds > 1)
    payload = dict(
        protocol="E5：WESAD 混合权重选择与评估%s；模型=%s（%d 成员集成）"
                 % ("（%d 折受试者 CV）" % args.folds if kfold else "（单次划分）",
                    args.pattern, len(weights)),
        headline=("kfold_mean" if kfold else "single_holdout"),
        headline_value=(float(tab["kfold"]["mean"]) if kfold else None),
        params=dict(folds=int(args.folds), n_select=int(args.n_select),
                    weights_w=list(args.weights_w), lam=float(args.lam),
                    jump=float(args.jump)),
        table=tab)
    fname = ("blend_holdout_kfold.json" if kfold else "blend_holdout_single.json")
    with open(os.path.join(args.out, fname), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if kfold:
        print("[E5] 报告口径（必须用这个）：**%d 折均值 %.2f ± %.2f**；单折 per_fold=%s"
              % (args.folds, tab["kfold"]["mean"], tab["kfold"]["std"], tab["kfold"]["per_fold"]))
    print("[E5] 完成 -> %s" % args.out)
    return tab


if __name__ == "__main__":
    main()

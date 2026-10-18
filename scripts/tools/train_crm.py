#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练「残差化 RLS+U-Net」：out = 物理先验 + Δ(x)，并在 CIME 上与跨域上报指标。

两个变体
--------
- `--residual-base rls`（主路径，需 IMU）：输入 3 通道 `[PPG, M̂, PPG_RLS]`，先验取 **PPG_RLS**（第 2 通道）；
- `--residual-base raw`（无 IMU 退化路径）：输入 1 通道 `[PPG]`，先验取 **PPG**（第 0 通道）。

损失：`MSE + 0.5·L1 + energy_w·心率带能量守恒 + delta_w·|Δ|`
- 心率带能量守恒直接针对诊断出的失败模式（域外把 0.7–4 Hz 能量削掉）；
- |Δ| 惩罚 = identity 保护，让网络"能不修就不修"。

协议：验证集 = CIME 训练集里的**受试者 9**（500 段，按受试者划分）选最优 epoch；测试集只用于报告。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/train_residual.py --residual-base rls --seeds 0 1 2 3 4 --epochs 15
    $PY scripts/tools/train_residual.py --residual-base raw --seeds 0 --epochs 15 --eval-cross

产物
----
    <out>/weights/residual_<base>_s<seed>.pt
    <out>/results/residual_train_summary_<base>[_s<seed>].json
    <out>/results/residual_crossdomain_<base>[_adabn].json    （--eval-cross 时）
"""
from __future__ import annotations

import argparse
import csv
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


def mrstft_loss(torch, pred, tgt, ffts, mode="both"):
    """多分辨率 STFT 损失（**无 log 项**，避免近静音 bin 的梯度爆炸）。

    ⚠️ 第一版用了 `log(|S|+1e-7)` 的幅度 L1，实测**训练塌缩**（|Δ|/|base| 仅 2.1%，
    PRD 82.8 ≈ RLS-only 83.78）：`log` 在 |S|→0 的 bin 上梯度 ~1e7，谱项梯度完全压过 MSE/L1，
    残差分支被推回 0。改为音频领域更稳的形式：

    - `sc`    ：谱收敛 ‖|S_t|−|S_p|‖_F / ‖|S_t|‖_F
    - `mag03` ：对幅度做 **0.3 次幂压缩**后取 L1（有界梯度、保留感知权重）
    - `both`  ：两者相加（默认）
    """
    tot = 0.0
    x = pred[:, 0] if pred.dim() == 3 else pred
    y = tgt[:, 0] if tgt.dim() == 3 else tgt
    for n_fft in ffts:
        hop = max(1, n_fft // 4)
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        sp = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        st = torch.stft(y, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
        mp, mt = sp.abs(), st.abs()
        v = 0.0
        if mode in ("sc", "both"):
            v = v + torch.norm(mt - mp) / (torch.norm(mt) + 1e-7)
        if mode in ("mag03", "both"):
            mp3 = mp.clamp_min(0) ** 0.3
            mt3 = mt.clamp_min(0) ** 0.3
            v = v + torch.mean(torch.abs(mt3 - mp3))
        tot = tot + v
    return tot / max(len(ffts), 1)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="残差化 RLS+U-Net 训练 + 跨域评估")
    ap.add_argument("--norm", choices=["bn", "in", "gn"], default="bn",
                    help="归一化层（P3 域不变实验）：bn=原实现；in=InstanceNorm；gn=GroupNorm")
    ap.add_argument("--residual-base", choices=["rls", "raw"], default="rls")
    ap.add_argument("--direct", action="store_true",
                    help="消融对照：直接输出整条波形（out = net(x)），不做 out = 先验 + Δ")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--energy-w", type=float, default=0.5, help="心率带能量守恒损失权重")
    ap.add_argument("--delta-w", type=float, default=0.02, help="|Δ| 惩罚（identity 保护）权重")
    ap.add_argument("--crm-base", type=int, default=16, help="CRM 主干基础通道数（16/32）")
    ap.add_argument("--crm-fft", type=int, default=256, help="CRM STFT 窗长（256=2.56 s / 512=5.12 s）")
    ap.add_argument("--cond-dim", type=int, choices=[0, 2, 3], default=0,
                    help="FiLM 条件注入维度（0=关闭；2=[运动强度,谱熵]；3=再+RLS 伪影占比 s2）")
    ap.add_argument("--stft-w", type=float, default=0.0,
                    help="多分辨率 STFT 损失权重（0=关闭；音频领域标准手法，直接改善波形指标）")
    ap.add_argument("--stft-ffts", type=int, nargs="+", default=[128, 256, 512],
                    help="STFT 多分辨率窗长（hop=win/4，Hann）")
    ap.add_argument("--stft-mode", choices=["sc", "mag03", "both"], default="both",
                    help="STFT 损失形式（第一版 log 幅度项会塌缩，已移除）")
    ap.add_argument("--val-subject", type=int, default=9)
    ap.add_argument("--domain-rand", action="store_true", help="用合成域扰动扩充训练集（domain randomization）")
    ap.add_argument("--dr-size", type=int, default=4500, help="每个种子合成多少条样本")
    ap.add_argument("--dr-rest-frac", type=float, default=0.25, help="合成样本中接近静息（无运动）的比例")
    ap.add_argument("--dr-real-refs", default="", choices=["", "wesad", "troika"],
                    help="用**真实运动学**（该数据集的真实 ACC 段）替代正弦合成参考")
    ap.add_argument("--dr-ref-limit", type=int, default=6, help="取多少条记录的真实参考")
    ap.add_argument("--dr-ref-seconds", type=float, default=60.0, help="每条记录取前多少秒")
    ap.add_argument("--tag", default="", help="权重/结果文件名后缀（如 _dr）")
    ap.add_argument("--feat-dir", default=P.artifact(os.path.join("final_10seed", "features")))
    ap.add_argument("--out", default=P.artifact("residual"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-cross", action="store_true", help="训练后做一次跨域零样本/AdaBN 评估")
    ap.add_argument("--eval-only", action="store_true", help="跳过训练，直接加载已存权重做跨域评估")
    ap.add_argument("--eval-cross-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lam", type=float, default=None, help="跨域评估用的 Viterbi 平滑权重（覆盖默认）")
    ap.add_argument("--jump", type=float, default=None, help="跨域评估用的跳变尺度")
    ap.add_argument("--band", default=None, help="跨域评估用的心率带，如 0.7,3.0")
    ap.add_argument("--cross-limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import residual as R
    import crm as CRM
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    CDE = load_module("cde", os.path.join(SCRIPTS, "tools", "cross_dataset_eval.py"))
    torch.manual_seed(0); np.random.seed(0)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.out, "weights"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)

    cd = int(args.cond_dim)
    if cd > 0 and bool(getattr(args, "domain_rand", False)):
        raise SystemExit("[crm] --cond-dim>0 与 DR 合成样本暂不兼容（合成样本无对应条件特征）")
    base_cfg = dict(rls=dict(xkey="XrlsOnly", in_ch=3, base_index=2),
                    raw=dict(xkey="Xppg", in_ch=1, base_index=0))[args.residual_base]
    dtr = np.load(os.path.join(args.feat_dir, "train.npz"))
    dte = np.load(os.path.join(args.feat_dir, "test.npz"))
    subj = dtr["subject"] if "subject" in dtr.files else None
    val_mask = (subj == args.val_subject) if subj is not None else \
        np.zeros(len(dtr["y"]), dtype=bool)
    if subj is None or val_mask.sum() == 0:
        val_mask[-500:] = True
    tr_idx, va_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
    X, Y = dtr[base_cfg["xkey"]], dtr["y"]
    Xte, Yte = dte[base_cfg["xkey"]], dte["y"]
    S_tr = S_te = None; cond_mu = cond_sd = None
    if cd > 0:
        A = dtr["stats%d" % cd].astype(np.float64); B = dte["stats%d" % cd].astype(np.float64)
        cond_mu = A.mean(0); cond_sd = A.std(0) + 1e-6          # 用**训练集**统计标准化
        S_tr = ((A - cond_mu) / cond_sd).astype(np.float32)
        S_te = ((B - cond_mu) / cond_sd).astype(np.float32)
        print("[crm] FiLM 条件注入 dim=%d：均值 %s 标准差 %s"
              % (cd, np.round(cond_mu, 3), np.round(cond_sd, 3)))

    def prd_np(clean, pred):
        return float(np.nanmean([P_.prd(clean[i], pred[i]) for i in range(len(clean))]))

    print("[crm] 变体=%s 输入=%s(%d 通道) 先验=第 %d 通道 | 训练 %d 段 / 验证 %d 段 / 测试 %d 段"
          % (args.residual_base, base_cfg["xkey"], base_cfg["in_ch"], base_cfg["base_index"],
             len(tr_idx), len(va_idx), len(Xte)))
    summary = {}
    model_by_seed = {}
    if args.eval_only:
        for seed in args.seeds:
            ck = os.path.join(args.out, "weights", "crm_%s%s_s%d.pt" % (args.residual_base, args.tag, seed))
            model, meta = CRM.load(ck, device=dev)
            model_by_seed[seed] = model
            with torch.no_grad():
                pte = model(torch.from_numpy(Xte).float().to(dev),
                            torch.zeros(len(Xte), 0).to(dev)).cpu().numpy()[:, 0]
            r = float(np.nanmean([np.corrcoef(Yte[i], pte[i])[0, 1] for i in range(len(Yte))]))
            snr = float(np.nanmean([P_.snr_db(pte[i], Yte[i]) for i in range(len(Yte))]))
            summary[seed] = dict(params=sum(p.numel() for p in model.parameters()),
                                 val_PRD=meta.get("val_PRD"), weight=ck, loaded=True,
                                 test_PRD=round(prd_np(Yte, pte), 3), test_Pearson=round(r, 4),
                                 test_SNR_dB=round(snr, 3))
            print("[crm] 载入 %s（val PRD %s | 域内 test PRD %.2f）"
                  % (ck, meta.get("val_PRD"), summary[seed]["test_PRD"]))
    for seed in ([] if args.eval_only else args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        model = CRM.build(base_cfg["in_ch"], base_cfg["base_index"], norm=args.norm,
                          base=args.crm_base, n_fft=args.crm_fft,
                          hop=max(16, args.crm_fft // 4)).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        mse, l1 = nn.MSELoss(), nn.L1Loss()
        Xtr_np, Ytr_np = X[tr_idx], Y[tr_idx]
        if args.domain_rand:
            import domain_rand as DR
            rng = np.random.default_rng(1000 + seed)
            dclean = np.load(os.path.join(P_.DATA_DIR, "cime_train.npz"))["clean"]
            pick = rng.integers(0, len(dclean), args.dr_size)
            if args.dr_real_refs:
                refs = DR.load_real_refs(args.dr_real_refs, limit=args.dr_ref_limit,
                                         max_seconds=args.dr_ref_seconds)
                syn_noisy, syn_ref = DR.synth_batch_real(dclean[pick].astype(np.float64), refs, rng,
                                                         rest_frac=args.dr_rest_frac)
                if seed == args.seeds[0]:
                    print("        [DR-real] 运动学来源=%s，参考 %s" % (args.dr_real_refs, refs.shape))
            else:
                syn_noisy, syn_ref = DR.synth_batch(dclean[pick].astype(np.float64), rng,
                                                    rest_frac=args.dr_rest_frac)
            Xs, Ys = DR.features_from(P_, dclean[pick].astype(np.float64), syn_noisy, syn_ref,
                                      kind=args.residual_base)
            if seed == args.seeds[0]:
                print("        [DR] 实样本 %d + 合成样本 %d（合成伪影/脉搏幅度比中位 %.2f）"
                      % (len(Xtr_np), len(Xs),
                         float(np.median((syn_noisy - dclean[pick]).std(1) / (dclean[pick].std(1) + 1e-9)))))
            Xtr_np = np.concatenate([Xtr_np, Xs]); Ytr_np = np.concatenate([Ytr_np, Ys])
        Xtr = torch.from_numpy(Xtr_np).float(); Ytr = torch.from_numpy(Ytr_np).float().unsqueeze(1)
        Xva = torch.from_numpy(X[va_idx]).float(); Yva = Y[va_idx]
        zero = torch.zeros(1, 0).to(dev)
        best = dict(vprd=float("inf"), ep=-1, state=None, delta=float("nan"))
        rows = []
        for ep in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(len(Xtr)); tot = 0.0
            for s in range(0, len(Xtr), args.batch):
                b = perm[s:s + args.batch]
                xb, yb = Xtr[b].to(dev), Ytr[b].to(dev)
                stb = (torch.from_numpy(S_tr)[b].to(dev) if cd > 0 else zero.expand(len(b), 0))
                opt.zero_grad()
                out = model(xb, stb)
                loss = mse(out, yb) + 0.5 * l1(out, yb)
                if args.stft_w:
                    loss = loss + args.stft_w * mrstft_loss(torch, out, yb, args.stft_ffts,
                                                            mode=args.stft_mode)
                if args.energy_w:
                    loss = loss + args.energy_w * R.energy_consistency_loss(out, yb)
                if args.delta_w:
                    loss = loss + args.delta_w * out.sub(xb[:, base_cfg["base_index"]:base_cfg["base_index"] + 1]).abs().mean()
                loss.backward(); opt.step()
                tot += float(loss.detach()) * len(b)
            sched.step()
            model.eval()
            with torch.no_grad():
                stva = (torch.from_numpy(S_tr)[va_idx].to(dev) if cd > 0
                        else zero.expand(len(Xva), 0))
                pv = model(Xva.to(dev), stva).cpu().numpy()[:, 0]
                dnorm = float(np.mean(np.abs(pv - X[va_idx][:, base_cfg["base_index"]])) /
                              (np.mean(np.abs(X[va_idx][:, base_cfg["base_index"]])) + 1e-12))
            vprd = prd_np(Yva, pv)
            rows.append((seed, ep, round(tot / len(Xtr), 5), round(vprd, 3), round(dnorm, 4)))
            if vprd < best["vprd"]:
                best = dict(vprd=vprd, ep=ep, delta=dnorm,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        model.load_state_dict(best["state"]); model.eval()
        model_by_seed[seed] = model
        with torch.no_grad():
            stte = (torch.from_numpy(S_te).to(dev) if cd > 0 else zero.expand(len(Xte), 0))
            pte = model(torch.from_numpy(Xte).float().to(dev), stte).cpu().numpy()[:, 0]
        r = float(np.nanmean([np.corrcoef(Yte[i], pte[i])[0, 1] for i in range(len(Yte))]))
        snr = float(np.nanmean([P_.snr_db(pte[i], Yte[i]) for i in range(len(Yte))]))
        prd = prd_np(Yte, pte)
        ck = os.path.join(args.out, "weights", "crm_%s%s_s%d.pt" % (args.residual_base, args.tag, seed))
        CRM.save(ck, model, dict(xkey=base_cfg["xkey"], seed=seed, val_PRD=best["vprd"],
                               cond_dim=cd,
                               cond_mu=(cond_mu.tolist() if cond_mu is not None else None),
                               cond_sd=(cond_sd.tolist() if cond_sd is not None else None),
                               energy_w=args.energy_w, delta_w=args.delta_w, variant=args.residual_base))
        summary[seed] = dict(params=n_par, best_epoch=best["ep"], val_PRD=round(best["vprd"], 3),
                             test_PRD=round(prd, 3), test_Pearson=round(r, 4), test_SNR_dB=round(snr, 3),
                             delta_rel=round(best["delta"], 4), weight=ck)
        print("        seed=%d  best_ep=%2d  val PRD %6.2f | test PRD %6.2f  Pearson %.4f  SNR %.2f  |Δ|/|base| %.1f%%"
              % (seed, best["ep"], best["vprd"], prd, r, snr, 100 * best["delta"]))
    slug = "crm_train_summary_%s%s%s" % (
        args.residual_base, args.tag, "" if len(args.seeds) > 1 else "_s%d" % args.seeds[0])
    with open(os.path.join(args.out, "results", slug + ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(protocol="验证集=受试者 %d（按受试者划分）；测试集只用于报告" % args.val_subject,
                       variant=args.residual_base, energy_w=args.energy_w, delta_w=args.delta_w,
                       epochs=args.epochs, seeds=args.seeds,
                       domain_rand=bool(args.domain_rand), dr_size=args.dr_size, dr_rest_frac=args.dr_rest_frac,
                       dr_real_refs=args.dr_real_refs, per_seed={str(k): v for k, v in summary.items()}),
                  f, ensure_ascii=False, indent=2)
    mean = lambda k: float(np.mean([summary[s][k] for s in summary]))                     # noqa: E731
    print("[crm] 汇总（%d 种子）：test PRD %.2f ± %.2f | Pearson %.4f | SNR %.2f dB | 参数 %.3f M"
          % (len(summary), mean("test_PRD"),
             float(np.std([summary[s]["test_PRD"] for s in summary], ddof=1)) if len(summary) > 1 else 0.0,
             mean("test_Pearson"), mean("test_SNR_dB"), mean("params") / 1e6))

    # ---------------- 可选：跨域评估 ----------------
    if args.eval_cross:
        import ext_loaders as EL
        for mode in ("none", "adabn"):
            res, integ = {}, {}
            for seed in args.eval_cross_seeds:
                if seed not in model_by_seed:
                    continue
                for ds in ("troika", "wesad", "bidmc", "capnobase"):
                    recs, has_acc = CDE.iter_dataset(EL, ds)
                    if args.cross_limit:
                        recs = recs[:args.cross_limit]
                    use_rls = has_acc and args.residual_base == "rls"
                    if args.residual_base == "rls" and not has_acc:
                        print("        跳过 %s（该数据集无 IMU，无法构造 RLS 先验）" % ds)
                        continue
                    kind = "rls_only" if use_rls else "ppg_only"
                    nets = dict(models={kind: model_by_seed[seed]}, dev=dev)
                    if mode == "adabn":
                        import copy as _copy
                        batches = []
                        for rec in recs:
                            prep = CDE.prepare(P_, rec, use_rls)
                            if prep is None:
                                continue
                            ppg, m_hat, _, _, _, _ = prep
                            st = CDE.window_starts(len(ppg), CDE.ADAPT_WIN, CDE.ADAPT_HOP)
                            Xw, _ = CDE.window_batch_scaled(CDE.raw_channels(ppg, m_hat, kind), st, CDE.ADAPT_WIN)
                            batches.append(Xw)
                            if sum(len(b) for b in batches) >= 20 * 32:
                                break
                        if batches:
                            nets["models"][kind] = CDE.adabn(_copy.deepcopy(model_by_seed[seed]),
                                                             [np.concatenate(batches)[:640]], dev, momentum=0.9)
                    acc, ig = {}, []
                    band_ovr = tuple(float(x) for x in args.band.split(",")) if args.band else None
                    for rec in recs:
                        errs, integ1 = CDE.eval_record(P_, torch, nets, rec, use_rls,
                                                       lam=args.lam, jump=args.jump, band=band_ovr)
                        if errs is None:
                            continue
                        for k, v in errs.items():
                            acc.setdefault(k, []).extend(v)
                        if integ1:
                            ig.append(integ1)
                    key = "%s_s%d" % (ds, seed)
                    res[key] = {k: CDE.stat(v) for k, v in acc.items()}
                    integ[key] = dict(n_records=len(ig))
            fn = "crm_crossdomain_%s%s%s.json" % (args.residual_base, args.tag,
                                                       "" if mode == "none" else "_adabn")
            with open(os.path.join(args.out, "results", fn), "w", encoding="utf-8") as f:
                json.dump(dict(protocol="残差模型零样本/AdaBN 跨域评估（8 s 窗 / 2 s 步进）",
                               selector=dict(lam=args.lam, jump=args.jump, band=args.band),
                               mode=mode, variant=args.residual_base, seeds=args.eval_cross_seeds,
                               results=res), f, ensure_ascii=False, indent=2)
            print("[crm] 跨域评估(%s) -> %s" % (mode, fn))
            for k, v in res.items():
                u = v.get("unet_rls_only") or v.get("unet_ppg_only") or {}
                print("        %-14s n=%5d  raw %5s  RLS %5s  **残差 U-Net %5s**"
                      % (k, (v.get("raw") or {}).get("n"),
                         (v.get("raw") or {}).get("MAE"),
                         (v.get("rls(3axis)") or {}).get("MAE"), u.get("MAE")))
    return summary


if __name__ == "__main__":
    main()

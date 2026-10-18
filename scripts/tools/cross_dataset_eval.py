#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跨数据集泛化评估（E4）+ **无监督适配（AdaBN）** + 护栏对比。

协议
----
- 全部重采样到 100 Hz；HR 用 **8 s 窗 / 2 s 步进 + 谱域 Viterbi**；
- 参考心率：有 ECG 就用**同一窗口协议的 ECG R 峰**（可暴露窗口错位）；
  没有 ECG（TROIKA 官方测试集）就用数据集自带的 `BPM0`（8 s / 2 s，逐窗对齐）；
- **零样本**：只用 CIME 训好的权重，目标域不参与任何训练/调参；
- **AdaBN（无监督）**：用目标域**无标签**数据前向若干 batch，只重算 BatchNorm 的
  running stats（不动权重、不反传）——诊断显示 BN 漂移与跨域崩溃同序（BIDMC 5.4×）。

信息条件
--------
| 数据集 | IMU | 可跑的方法 |
|---|---|---|
| TROIKA 官方测试集 | 3 轴 ACC | raw / 带通 / RLS(3轴) / ppg_only U-Net / **rls_only U-Net** |
| WESAD（腕部 BVP） | 3 轴 ACC | 同上 |
| BIDMC / CapnoBase | 无 | raw / 带通 / **ppg_only U-Net**（无 IMU 退化路径） |

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/cross_dataset_eval.py --adapt none  --datasets troika wesad bidmc capnobase
    $PY scripts/tools/cross_dataset_eval.py --adapt adabn --datasets troika wesad bidmc capnobase
    $PY scripts/tools/cross_dataset_eval.py --limit 3            # 快速

产物
----
    <out>/cross_dataset_hr[_<adapt>].{json,csv}
"""
from __future__ import annotations

import argparse
import copy
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

WIN_S, HOP_S = 8.0, 2.0
WIN, HOP = 3000, 200
FS = 100.0
BAND = (0.7, 4.0)
ADAPT_WIN = 3000          # AdaBN 用的窗长（30 s，与训练时一致）
ADAPT_HOP = 1500


def load_pipeline():
    path = os.path.join(SCRIPTS, "pipeline", "pipeline.py")
    spec = importlib.util.spec_from_file_location("ppg_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _rpeaks_v2(ecg, fs):
    """v2 R 峰检测：5–25 Hz 原始信号（**不用 |包络|**）+ 自适应阈值 + T 波幅度抑制。

    为什么换：旧口径（5–15 Hz + |包络| + 固定 0.35 s 不应期）在高心率（儿科麻醉 CapnoBase 等）下
    R 波与 T 波各产生一个峰 → **参考心率被算成 2 倍**（实测 117.6 vs 真值 66.3、133.3 vs 65.9）。
    参考错会让该记录乃至整个数据集均值失真（CapnoBase 两条记录贡献了 56% 的总误差）。
    """
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype="band")
    x = filtfilt(b, a, np.asarray(ecg, dtype=np.float64))
    cand, _ = find_peaks(x, height=np.percentile(x, 98), distance=int(0.20 * fs))
    pk = []
    for c in cand:
        if not pk:
            pk.append(c); continue
        rr = (c - pk[-1]) / fs
        if rr < 0.28:                                  # 生理不应期
            continue
        if len(pk) >= 2:                               # T 波抑制：间隔过短且幅度明显更低
            prev_rr = (pk[-1] - pk[-2]) / fs
            if rr < 0.45 * prev_rr and x[c] < 0.65 * x[pk[-1]]:
                continue
        pk.append(c)
    return np.array(pk)


ECG_REF = ["v2"]          # 由 main() 的 --ecg-ref 覆盖（供 prepare 使用）


def ecg_hr(ecg, fs, n_win, win_s=WIN_S, hop_s=HOP_S, method="v2"):
    """ECG R 峰 → 逐窗心率。

    method="v2"（默认）：`_rpeaks_v2`（修正 T 波过检，见其 docstring）；
    method="legacy"：旧口径（5–15 Hz + |包络| + 0.35 s 不应期），仅用于对照/复现已归档数字。
    """
    from scipy.signal import butter, filtfilt, find_peaks
    if method == "legacy":
        b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
        env = np.abs(filtfilt(b, a, ecg))
        pk, _ = find_peaks(env, height=env.mean() + 0.6 * env.std(), distance=int(0.35 * fs))
    else:
        pk = _rpeaks_v2(ecg, fs)
    hr = np.full(n_win, np.nan)
    win, hop = int(win_s * fs), int(hop_s * fs)
    for i in range(n_win):
        q = pk[(pk >= i * hop) & (pk < i * hop + win)]
        if len(q) >= 2:
            rr = np.diff(q) / fs
            rr = rr[(rr > 0.3) & (rr < 1.5)]
            if len(rr):
                hr[i] = 60.0 / np.median(rr)
    return hr


def band_ratio(x, fs=FS, band=BAND):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (fr >= band[0]) & (fr <= band[1])
    return float(sp[m].sum() / (sp.sum() + 1e-12))


def window_starts(n, win=WIN, hop=HOP):
    starts = list(range(0, max(1, n - win + 1), hop))
    if starts and starts[-1] + win < n:
        starts.append(n - win)
    return starts or [0]


FEAT_ALPHA = 0.40        # CIME 训练特征的收缩系数（step2 权威标定）；**模型输入必须用它**
FEAT_LAM = 0.995         # 同上（λ）


def raw_channels(ppg, m_hat, kind, alpha=FEAT_ALPHA):
    """未缩放的输入通道（与 pipeline.build_features 的通道定义一致）。

    ⚠️ `alpha` 默认必须是**训练特征口径 0.40**（历史默认 1.0 会让零样本 U-Net 的
    `ppg_rls` 通道与训练不一致：两者相差 0.6·M̂，见 `reports/41` 审计 P0）。
    物理基线行（`rls(3axis)`）仍按 `ppg − M̂`（α=1，完全抵消）单独计算。
    """
    if kind == "ppg_only":
        return [ppg]
    return [ppg, m_hat, ppg - alpha * m_hat]


def window_batch_scaled(chans, starts, win=WIN):
    """按**每窗 std** 缩放（与训练口径一致），返回 (X, 每窗缩放系数)。"""
    ppg = chans[0]
    X = np.zeros((len(starts), len(chans), win), dtype=np.float32)
    scs = np.zeros(len(starts), dtype=np.float64)
    for j, st in enumerate(starts):
        sc = ppg[st:st + win].std() + 1e-8
        scs[j] = sc
        X[j] = np.stack([c[st:st + win] / sc for c in chans])
    return X, scs


def reconstruct_batch(net, dev, chans, win=WIN, hop=HOP, bs=256, stats=None):
    """滑窗重建整段（重叠处取平均）；每窗按自身 std 缩放、输出再乘回（与训练同口径）。

    `stats`：cond/FiLM 模型的条件向量 (B,cd)（已标准化）；为 None 时对 cond_dim>0 的模型会**明确报错**
    （历史上传 (B,0) 会静默 shape 崩，见 reports/41 审计 P1-5）。
    """
    import torch
    n = len(chans[0])
    starts = window_starts(n, win, hop)
    out = np.zeros(n); wsum = np.zeros(n)
    X, scs = window_batch_scaled(chans, starts, win)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(dev)
            cd = int(getattr(net, "cond_dim", 0) or getattr(net, "cond_strength", 0) or 0)
            if cd > 0:
                if stats is None:
                    raise RuntimeError("该模型 cond_dim=%d，reconstruct_batch 必须传 stats（条件向量）" % cd)
                st = torch.from_numpy(np.asarray(stats)[i:i + len(xb)]).float().to(dev)
            else:
                st = torch.zeros(len(xb), 0).to(dev)
            yb = net(xb, st).cpu().numpy()[:, 0]
            for jj, st in enumerate(starts[i:i + bs]):
                out[st:st + win] += yb[jj] * scs[i + jj]; wsum[st:st + win] += 1
    return out / np.maximum(wsum, 1e-8)


def adabn(net, batches, dev, momentum=0.1):
    """无监督 BN 重校准：只更新 BatchNorm 的 running stats（不动权重、不反传）。"""
    import torch
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    old_m = [m.momentum for m in bns]
    for m in bns:
        m.momentum = momentum
    net.train()
    with torch.no_grad():
        for X in batches:
            xb = torch.from_numpy(X).float().to(dev)
            if len(xb) < 2:
                continue
            net(xb, torch.zeros(len(xb), 0).to(dev))
    for m, om in zip(bns, old_m):
        m.momentum = om
    net.eval()
    return net


def _peakiness(out, band=(0.7, 3.0)):
    import torch
    w = torch.hann_window(out.shape[-1], device=out.device)
    sp = torch.abs(torch.fft.rfft(out * w, dim=-1)) ** 2
    fr = torch.fft.rfftfreq(out.shape[-1], 1.0 / FS).to(out.device)
    m = (fr >= band[0]) & (fr <= band[1])
    s = sp[..., m]
    return s.max(dim=-1).values / (s.sum(dim=-1) + 1e-12)


def _band_energy(out, band=(0.7, 3.0)):
    import torch
    w = torch.hann_window(out.shape[-1], device=out.device)
    sp = torch.abs(torch.fft.rfft(out * w, dim=-1)) ** 2
    fr = torch.fft.rfftfreq(out.shape[-1], 1.0 / FS).to(out.device)
    m = (fr >= band[0]) & (fr <= band[1])
    return sp[..., m].sum(dim=-1)


def tent_adapt(net, dev, X, steps=20, lr=1e-3, bs=32, energy_w=1.0, norm_only=True, seed=0):
    """TENT：用目标域**无标签**窗，只更新归一化层仿射参数，最小化

        L = −mean(心率带内谱峰尖锐度) + energy_w·|log(E_hr(out)/E_hr(输入))|

    两项都**不含频率位置偏好**，所以不会像逐 bin 加权那样产生八度错误（`reports/11` §1.1）。
    实测：BIDMC/CapnoBase 有效（9.37→1.56 / 11.36→4.99），TROIKA 有害（`reports/13` §2）。
    """
    import torch
    if X is None or len(X) < 4:
        return net
    rng = np.random.default_rng(seed)
    params = []
    for name, p in net.named_parameters():
        if norm_only and not any(k in name for k in ("norm", "bn", "BatchNorm", "Instance")):
            continue
        p.requires_grad_(True); params.append(p)
    if not params:
        return net
    opt = torch.optim.Adam(params, lr=lr)
    net.train()
    Xa = np.asarray(X)
    for _ in range(int(steps)):
        idx = rng.integers(0, len(Xa), min(bs, len(Xa)))
        xb = torch.from_numpy(Xa[idx]).float().to(dev)
        out = net(xb, torch.zeros(len(xb), 0).to(dev))[:, 0]
        base = xb[:, 2, :] if xb.shape[1] >= 3 else xb[:, 0, :]
        loss = -_peakiness(out).mean() + energy_w * torch.abs(
            torch.log((_band_energy(out) + 1e-12) / (_band_energy(base) + 1e-12))).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    for p in params:
        p.requires_grad_(False)
    return net


def bn_shift_score(net, X, dev, bs=32):
    """任务域相对源域的 BN 漂移均值（用于 OOD 打分）。"""
    import torch
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    recs = {i: [] for i in range(len(bns))}

    def mk(j):
        def hook(mod, inp, out):
            x = inp[0]
            recs[j].append((x.mean(dim=(0, 2)).detach().cpu().numpy(),
                            x.var(dim=(0, 2), unbiased=False).detach().cpu().numpy()))
        return hook

    snap = [(m.running_mean.detach().clone(), m.running_var.detach().clone()) for m in bns]
    hs = [m.register_forward_hook(mk(j)) for j, m in enumerate(bns)]
    net.train()
    with torch.no_grad():
        for i in range(0, min(len(X), bs * 4), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().to(dev)
            if len(xb) < 2:
                continue
            net(xb, torch.zeros(len(xb), 0).to(dev))
    for h in hs:
        h.remove()
    for m, (rm, rv) in zip(bns, snap):
        m.running_mean.copy_(rm); m.running_var.copy_(rv)
    net.eval()
    vals = []
    for j, m in enumerate(bns):
        if not recs[j]:
            continue
        bm = np.mean([r[0] for r in recs[j]], axis=0)
        bv = np.mean([r[1] for r in recs[j]], axis=0)
        rm = m.running_mean.detach().cpu().numpy(); rv = m.running_var.detach().cpu().numpy()
        vals.append(float(np.mean(np.abs(bm - rm) / (np.sqrt(rv) + 1e-6)) +
                          np.mean(np.abs(np.log(bv + 1e-6) - np.log(rv + 1e-6)))))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# 单条记录
# ---------------------------------------------------------------------------
def prepare(P_, rec, use_rls):
    """返回 (ppg, m_hat, accmag, ref, n_win, ecg_ok)。多导联数据取第 0 导联（TROIKA 是 (2,N)）。"""
    ppg_arr = np.asarray(rec["ppg"], dtype=np.float64)
    ppg = ppg_arr.reshape(-1) if ppg_arr.ndim == 1 else ppg_arr[0]
    ecg = None if rec.get("ecg") is None else np.asarray(rec["ecg"], dtype=np.float64).reshape(-1)
    acc = rec.get("acc")
    n_win = int(max(0, (len(ppg) - WIN_S * FS) // (HOP_S * FS)) + 1)
    if n_win < 5:
        return None
    if ecg is not None and len(ecg) >= WIN * 2:
        ref = ecg_hr(ecg, FS, n_win, method=ECG_REF[0]); ecg_ok = True
    else:                                   # TROIKA 官方测试集：用自带 BPM0（8 s/2 s 逐窗）
        hr = rec.get("hr_ref")
        if hr is None:
            return None
        hr = np.asarray(hr, dtype=np.float64).reshape(-1)
        hfs = float(rec.get("hr_fs") or 0.0)
        hwin = float(rec.get("hr_win_s") or 0.0)
        if hfs > 0 and hwin > 0 and abs(hwin - WIN_S) < 1e-6 and abs(hfs - 1.0 / HOP_S) < 1e-6:
            ref = np.full(n_win, np.nan)
            ref[:min(n_win, len(hr))] = hr[:min(n_win, len(hr))]
        elif hfs > 0 and len(hr) >= 2:
            # 窗口 i 的中心时刻 = i*HOP_S + WIN_S/2；按参考采样率线性插值
            t_win = np.arange(n_win) * HOP_S + WIN_S / 2.0
            t_ref = np.arange(len(hr)) / hfs
            ok = np.isfinite(hr)
            ref = (np.interp(t_win, t_ref[ok], hr[ok]) if ok.sum() >= 2
                   else np.full(n_win, np.nan))
            if ok.sum() < 2 or (t_win.min() > t_ref[ok].max() or t_win.max() < t_ref[ok].min()):
                return None
        else:
            return None                       # 无法对齐的参考（宁可丢弃也不静默错位）
        ecg_ok = False
    accmag = np.zeros_like(ppg); m_hat = np.zeros_like(ppg)
    # 基线 RLS：保持历史**发布口径** λ=0.99（改它会改动已发布基线，如 TROIKA 2.73）
    _BASE_LAM = 0.99
    if use_rls and acc is not None:
        a3 = np.asarray(acc, dtype=np.float64)
        accmag = np.sqrt((a3 ** 2).sum(0))
        # 参考通道策略（reports/29）：**有陀螺仪时用异质两路 ACC_x + Gyro_y**（CIME 配置），
        # 只有加速度计时用 3 轴。实测 WRIST：21.99 → 16.03（池化，−27%）；TROIKA/WESAD 无 gyro 不受影响。
        gyro = rec.get("meta", {}).get("gyro")
        if gyro is not None and np.asarray(gyro).size and np.asarray(gyro).ndim == 2 \
                and np.asarray(gyro).shape[0] >= 2:
            g3 = np.asarray(gyro, dtype=np.float64)
            a_ref = a3[0:1].T[None, :]
            g_ref = g3[1:2].T[None, :]
        else:
            a_ref = a3.T[None, :]
            g_ref = np.zeros_like(ppg)[None, :]
        m_hat, _ = P_.rls_decompose(ppg[None, :], a_ref, g_ref, taps=4, lam=_BASE_LAM)
        m_hat = m_hat[0]
    return ppg, m_hat, accmag, ref, n_win, ecg_ok


def _model_mhat(P_, ppg, rec, m_hat, use_rls):
    """模型输入的 M̂ 必须用**训练口径 λ=0.995**（与基线 λ=0.99 不同；见 reports/41 审计 P0-1）。"""
    if FEAT_LAM == 0.99 or not use_rls:
        return m_hat
    acc = rec.get("acc")
    if acc is None:
        return m_hat
    a3 = np.asarray(acc, dtype=np.float64)
    gyro = rec.get("meta", {}).get("gyro")
    if gyro is not None and np.asarray(gyro).size and np.asarray(gyro).ndim == 2 \
            and np.asarray(gyro).shape[0] >= 2:
        g3 = np.asarray(gyro, dtype=np.float64)
        a_ref, g_ref = a3[0:1].T[None, :], g3[1:2].T[None, :]
    else:
        a_ref, g_ref = a3.T[None, :], np.zeros_like(ppg)[None, :]
    return P_.rls_decompose(ppg[None, :], a_ref, g_ref, taps=4, lam=FEAT_LAM)[0][0]


def eval_record(P_, torch, nets, rec, use_rls, ret_signals=False,
                lam=None, jump=None, band=None):
    prep = prepare(P_, rec, use_rls)
    if prep is None:
        return (None, None, None) if ret_signals else (None, None)
    ppg, m_hat, accmag, ref, n_win, ecg_ok = prep
    ppg_rls = ppg - 1.0 * m_hat

    from scipy import signal as ssig
    sos = ssig.butter(4, [0.5, 4.0], btype="bandpass", fs=FS, output="sos")
    signals = {"raw": ppg, "bandpass": ssig.sosfiltfilt(sos, ppg)}
    if use_rls:
        signals["rls(3axis)"] = ppg_rls

    def run_net(kind):
        net = nets["models"][kind]
        return reconstruct_batch(net, nets["dev"],
                                 raw_channels(ppg, _model_mhat(P_, ppg, rec, m_hat, use_rls), kind))

    if "ppg_only" in nets["models"]:
        signals["unet_ppg_only"] = run_net("ppg_only")
    if use_rls and "rls_only" in nets["models"]:
        signals["unet_rls_only"] = run_net("rls_only")

    errs = {}
    for k, sig in signals.items():
        f, Pm = P_.psd_matrix(np.asarray(sig, dtype=np.float64), accmag, FS, n_win, use_notch=False)
        kw = {}
        if lam is not None: kw["lam"] = lam
        if jump is not None: kw["jump_bpm"] = jump
        if band is not None: kw["band"] = band
        hr = P_.sel_viterbi(f, Pm, **kw)
        e = [(float(ref[i]), float(hr[i])) for i in range(min(n_win, len(hr)))
             if np.isfinite(ref[i]) and np.isfinite(hr[i])]
        if e:
            errs[k] = e

    integ = None
    if "unet_ppg_only" in signals:
        u = signals["unet_ppg_only"]
        cors, dratios = [], []
        for s in range(0, len(ppg) - WIN + 1, WIN):
            r, q = ppg[s:s + WIN], u[s:s + WIN]
            if r.std() > 0 and q.std() > 0:
                cors.append(float(np.corrcoef(r, q)[0, 1]))
            dratios.append(band_ratio(q) - band_ratio(r))
        integ = dict(corr_raw_unet=round(float(np.mean(cors)), 4) if cors else None,
                     band_ratio_delta=round(float(np.mean(dratios)), 6) if dratios else None)
    if ret_signals:
        return errs, integ, signals
    return errs, integ


def stat(pairs):
    if not pairs:
        return dict(n=0, MAE=None, RMSE=None, pct5=None)
    t = np.array([p[0] for p in pairs]); q = np.array([p[1] for p in pairs])
    err = q - t
    return dict(n=int(len(t)), MAE=round(float(np.mean(np.abs(err))), 2),
                RMSE=round(float(np.sqrt(np.mean(err ** 2))), 2),
                pct5=round(float(np.mean(np.abs(err) > 5)), 3))


# ---------------------------------------------------------------------------
def iter_dataset(EL, ds, fs_out=FS):
    if ds == "wesad":
        return list(EL.iter_wesad(fs_out=fs_out)), True
    if ds == "bidmc":
        return list(EL.iter_bidmc(fs_out=fs_out)), False
    if ds == "capnobase":
        return list(EL.iter_capnobase(fs_out=fs_out)), False
    if ds == "troika":
        return list(EL.iter_troika(split="test", fs_out=fs_out)), True
    if ds == "wrist":
        return list(EL.iter_wrist(fs_out=fs_out)), True
    if ds == "dalia":
        return list(EL.iter_dalia(fs_out=fs_out)), True
    raise ValueError(ds)


def main():
    ap = argparse.ArgumentParser(description="跨数据集泛化 + AdaBN 无监督适配")
    ap.add_argument("--datasets", nargs="+", default=["troika", "wesad", "bidmc", "capnobase"],
                    choices=["troika", "wesad", "bidmc", "capnobase", "wrist", "huawei", "dalia"])
    ap.add_argument("--adapt", choices=["none", "adabn", "adabn_per_record", "adabn_tent"], default="none",
                    help="none=零样本；adabn=重算 BN（无标签）；adabn_per_record=逐记录重算；"
                         "adabn_tent=**推荐的默认部署步骤**（AdaBN + TENT 无标签微调）")
    ap.add_argument("--tent-steps", type=int, default=20, help="TENT 步数（adabn_tent 时生效）")
    ap.add_argument("--tent-lr", type=float, default=1e-3)
    ap.add_argument("--tent-energy-w", type=float, default=1.0)
    ap.add_argument("--ecg-ref", choices=["v2", "legacy"], default="v2",
                    help="ECG 参考口径：v2=5–25 Hz 原始信号（修 T 波过检，默认）；legacy=旧口径（仅复现归档）")
    ap.add_argument("--deploy", action="store_true",
                    help="部署预设：等价于 --adapt adabn_tent --guardrail（无标签适配 + OOD 门控）")
    ap.add_argument("--adabn-batches", type=int, default=20, help="AdaBN 前向多少个 batch")
    ap.add_argument("--adabn-bs", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights-dir", default=P.artifact(os.path.join("final_10seed", "weights")))
    ap.add_argument("--out", default=P.artifact(os.path.join("final_10seed", "results")))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--guardrail", action="store_true",
                    help="开启 OOD 护栏：BN 漂移超阈值时该记录退回 RLS（无 IMU 则退回 raw）")
    ap.add_argument("--ood-threshold", type=float, default=0.20, help="BN 漂移阈值（源域约 0.08）")
    ap.add_argument("--lam", type=float, default=None, help="覆盖 Viterbi 平滑权重（默认用流水线常量）")
    ap.add_argument("--jump", type=float, default=None, help="覆盖 Viterbi 跳变尺度")
    ap.add_argument("--band", default=None, help="覆盖心率搜索带，如 0.7,3.0")
    ap.add_argument("--rls-weights", default=None, help="显式指定 rls_only 模型路径（如微调后的权重）")
    ap.add_argument("--no-unet", action="store_true",
                    help="只评物理基线，跳过网络行（回归门禁用；可省去模型 M̂ 重算）")
    ap.add_argument("--tag-suffix", default="", help="产物文件名额外后缀")
    args = ap.parse_args()

    import torch
    from gan import load_generator_checkpoint
    import ext_loaders as EL
    if args.deploy:
        args.adapt = args.adapt if args.adapt != "none" else "adabn_tent"
        args.guardrail = True
        print("[cross] --deploy 预设：adapt=%s + guardrail=on（OOD>%.2f 退回物理先验）"
              % (args.adapt, args.ood_threshold))
    ECG_REF[0] = args.ecg_ref
    P_ = load_pipeline()
    BAND_OVR = tuple(float(x) for x in args.band.split(",")) if args.band else None
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    base = {}
    for kind in ("ppg_only", "rls_only"):
        p = args.rls_weights if (kind == "rls_only" and args.rls_weights) else \
            os.path.join(args.weights_dir, "%s_s%d.pt" % (kind, args.seed))
        if os.path.exists(p):
            # ⚠️ 残差模型（out = 物理先验 + Δ）必须走 residual.load；
            # gan.load_generator_checkpoint 只加载基础网络（忽略 residual/base_index），会把输出当成 Δ 本身。
            import torch as _torch
            _ck = _torch.load(p, map_location="cpu", weights_only=True)
            if bool(_ck.get("residual", False)):
                import residual as _R
                net, meta = _R.load(p, device=dev, strict=True)
                print("        [load] %s：残差模型（norm=%s base_index=%s）→ 用 residual.load"
                      % (os.path.basename(p), _ck.get("norm", "bn"), _ck.get("base_index")))
            else:
                net, meta = load_generator_checkpoint(p, device=dev, strict=True)
            net.eval(); base[kind] = net
    print("[cross] 源域权重：%s | 适配方式：%s | 护栏：%s"
          % (list(base), args.adapt, args.guardrail))

    all_rows, out, integ_all, guard_log = [], {}, {}, {}
    for ds in args.datasets:
        recs, has_acc = iter_dataset(EL, ds)
        if args.limit:
            recs = recs[:args.limit]
        use_rls = has_acc and "rls_only" in base
        kind = "rls_only" if use_rls else "ppg_only"
        # --no-unet：只评物理基线（回归门禁/纯 RLS 诊断），跳过网络行与模型 M̂ 重算
        nets = (dict(models={}, dev=dev) if args.no_unet
                else dict(models={k: v for k, v in base.items()}, dev=dev))

        # ---- 适配：用目标域无标签数据重算 BN（复制副本，源域权重不受影响）----
        if args.adapt in ("adabn", "adabn_tent") and kind in base:
            batches = []
            for rec in recs:
                prep = prepare(P_, rec, use_rls)
                if prep is None:
                    continue
                ppg, m_hat, _, _, _, _ = prep
                starts = window_starts(len(ppg), ADAPT_WIN, ADAPT_HOP)
                Xw, _ = window_batch_scaled(raw_channels(ppg, _model_mhat(P_, ppg, rec, m_hat, use_rls), kind), starts, ADAPT_WIN)
                batches.append(Xw)
                if sum(len(b) for b in batches) >= args.adabn_batches * args.adabn_bs:
                    break
            if batches:
                X = np.concatenate(batches)[:args.adabn_batches * args.adabn_bs]
                m = adabn(copy.deepcopy(base[kind]), [X], dev,
                          momentum=0.9)     # 单次大 batch，近似精确统计
                print("        [AdaBN] 用 %d 个无标签窗重算 BN 统计（%s）" % (len(X), kind))
                if args.adapt == "adabn_tent":
                    m = tent_adapt(m, dev, X, steps=args.tent_steps, lr=args.tent_lr,
                                   energy_w=args.tent_energy_w)
                    print("        [TENT] %d 步无标签自监督微调（谱峰尖锐度 + 能量锚定）" % args.tent_steps)
                nets["models"][kind] = m

        acc = {}
        integs, ood_scores = [], []
        n_gate_total = [0, 0]
        for i, rec in enumerate(recs):
            prep = prepare(P_, rec, use_rls)
            if prep is None:
                continue
            nets_rec = nets
            if args.adapt == "adabn_per_record" and kind in base:
                ppg_, m_hat_, _, _, _, _ = prep
                st = window_starts(len(ppg_), ADAPT_WIN, ADAPT_HOP)
                Xw, _ = window_batch_scaled(raw_channels(ppg_, m_hat_, kind), st, ADAPT_WIN)
                nets_rec = dict(models=dict(nets["models"]), dev=dev)
                nets_rec["models"][kind] = adabn(copy.deepcopy(base[kind]), [Xw[:args.adabn_batches * args.adabn_bs]], dev, momentum=0.9)
            # ---- OOD 打分（护栏用）----
            ood = None
            if args.guardrail:
                ppg, m_hat, _, _, _, _ = prep
                starts = window_starts(len(ppg), ADAPT_WIN, ADAPT_HOP)
                Xw, _ = window_batch_scaled(raw_channels(ppg, _model_mhat(P_, ppg, rec, m_hat, use_rls), kind), starts, ADAPT_WIN)
                Xw = Xw[:128]
                ood = bn_shift_score(nets["models"][kind], Xw, dev)
                ood_scores.append(ood)
            errs, integ = eval_record(P_, torch, nets_rec, rec, use_rls,
                                      lam=args.lam, jump=args.jump, band=BAND_OVR)
            if errs is None:
                continue
            if args.guardrail:
                # 逐记录选择：BN 漂移 > 阈值 → 不用网络输出，退回 RLS（无 IMU 退回 raw）
                fb = "rls(3axis)" if (use_rls and "rls(3axis)" in errs) else "raw"
                um = "unet_rls_only" if "unet_rls_only" in errs else (
                    "unet_ppg_only" if "unet_ppg_only" in errs else None)
                if um is not None:
                    use_net = ood <= args.ood_threshold
                    errs["guardrail"] = errs[um] if use_net else errs[fb]
                    n_gate_total[0] += 1
                    if not use_net:
                        n_gate_total[1] += 1
            for k, v in errs.items():
                acc.setdefault(k, []).extend(v)
            if integ:
                integs.append(integ)
        table = {k: stat(v) for k, v in acc.items()}
        out[ds] = table
        entry = dict(n_records=len(integs),
                     corr_raw_unet=round(float(np.mean([x["corr_raw_unet"] for x in integs
                                                       if x["corr_raw_unet"] is not None])), 4) if integs else None,
                     band_ratio_delta=round(float(np.mean([x["band_ratio_delta"] for x in integs
                                                           if x["band_ratio_delta"] is not None])), 6) if integs else None)
        if args.guardrail and n_gate_total[0]:
            entry["gated_frac"] = round(n_gate_total[1] / n_gate_total[0], 3)
        if ood_scores:
            entry["ood_bn_shift_mean"] = round(float(np.mean(ood_scores)), 4)
            entry["ood_frac_gt_threshold"] = round(float(np.mean(np.array(ood_scores) > args.ood_threshold)), 3)
            guard_log[ds] = dict(threshold=args.ood_threshold, scores=[round(s, 4) for s in ood_scores])
        integ_all[ds] = entry
        for k, v in table.items():
            all_rows.append([ds, k, v["n"], v["MAE"], v["RMSE"], v["pct5"]])
        print("        %-18s %8s %8s %8s" % ("方法", "n", "MAE", ">5bpm"))
        for k, v in table.items():
            print("        %-18s %8d %8s %8s" % (k, v["n"], v["MAE"], v["pct5"]))

    suffix = args.tag_suffix or ("" if args.adapt == "none" and not args.guardrail else \
        "_%s%s" % (args.adapt, "_guard" if args.guardrail else ""))
    csv_p = os.path.join(args.out, "cross_dataset_hr%s.csv" % suffix)
    with open(csv_p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "n_windows", "MAE_bpm", "RMSE_bpm", "pct_err_gt5"])
        w.writerows(all_rows)
    meta = dict(protocol="8 s 窗 / 2 s 步进；参考心率优先用同步 ECG（同协议），无 ECG 用数据集 BPM0；"
                        "零样本=只用 CIME 权重；AdaBN=目标域无标签重算 BN 统计；"
                        "护栏=BN 漂移>阈值时退回 RLS（无 IMU 退回 raw）",
                adapt=args.adapt, guardrail=args.guardrail, ood_threshold=args.ood_threshold,
                selector=dict(lam=args.lam, jump=args.jump, band=args.band),
                rls_weights=args.rls_weights,
                fs=FS, limit=args.limit or None, seed=args.seed, weights_dir=args.weights_dir,
                zero_shot_models=list(base), hr=out, resting_no_degradation=integ_all, guardrail_log=guard_log)
    with open(os.path.join(args.out, "cross_dataset_hr%s.json" % suffix), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[cross] 完成 -> cross_dataset_hr%s.{json,csv}" % suffix)
    return meta


if __name__ == "__main__":
    main()

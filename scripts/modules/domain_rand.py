#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""域随机化（domain randomization）：用 CIME 的 clean PPG 合成"别的域"的训练对。

为什么
------
诊断显示模型只在 CIME（指夹、中等手指运动、100 Hz）这个域里成立；跨到
腕部/临床/剧烈运动域就崩。**零样本**要变好，只能让训练分布覆盖得更宽——
而目标域数据不能用来训练，所以用**合成**的域变化来扩充源域。

合成的四类域变化（都对应诊断里观察到的真实差异）
------------------------------------------------
1. **运动强度与基频**：合成运动参考（基频 1.2–3.5 Hz + 谐波 + 幅度调制 + 频率漂移），
   随机强度（含 0 = 静息段），直接针对“运动带与心率带重叠”的 TROIKA 失效；
2. **低频基线漂移**：腕部/接触压力引起的 <0.6 Hz 随机游走（临床与腕部数据里很显著）；
3. **器件响应**：随机低通截止（4–12 Hz）+ 随机高通（0.3–0.8 Hz），模拟不同传感器/前端；
4. **幅度尺度**：整段随机增益（不影响按段标准化后的形状，但影响 RLS 的等效步长）。

接口
----
    synth_batch(clean, rng, fs) -> (noisy, ref)      # ref: (2, B, N)，对应 CIME 的 [ACC_x, Gyro_y]
    features_from(clean, noisy, ref, P_, alpha, kind) -> (X, y)   # 与 pipeline.build_features 同口径
"""
from __future__ import annotations

import os
import sys

import numpy as np

FS = 100.0


def _lowpass(x, fs, fc, order=2):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=-1)


def _highpass(x, fs, fc, order=2):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, fc / (fs / 2), btype="high")
    return filtfilt(b, a, x, axis=-1)


def synth_reference(b, n, fs, rng):
    """合成运动参考（2 通道），模仿"某个佩戴位置 + 某种运动"的参考信号。"""
    t = np.arange(n) / fs
    f0 = rng.uniform(1.2, 3.5)                       # 步频/挥动基频（走 1.5~2，跑 2.5~3.5）
    n_h = int(rng.integers(1, 4))
    ref = np.zeros((b, 2, n), dtype=np.float64)
    drift = rng.uniform(0.02, 0.25)                  # 频率漂移率（真实运动不是严格周期）
    for h in range(1, n_h + 1):
        amp = rng.uniform(0.3, 1.0) / h
        ph = rng.uniform(0, 2 * np.pi)
        ph2 = rng.uniform(0, 2 * np.pi)
        f = f0 * h * (1.0 + drift * np.sin(2 * np.pi * rng.uniform(0.05, 0.4) * t + ph2))
        ref[:, 0] += amp * np.sin(2 * np.pi * f * t + ph)
        ref[:, 1] += rng.uniform(0.2, 1.0) * amp * np.cos(2 * np.pi * f * t + ph)
    # 幅度调制（握持松紧/接触变化）
    env = 1.0 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 1.0) * t + rng.uniform(0, 6.28))
    ref *= env[None, None, :]
    # 每个样本一个随机强度（含接近 0 = 静息）
    strength = rng.uniform(0.0, 3.0, size=(b, 1, 1))
    return ref * strength


def synth_batch(clean, rng, fs=FS, rest_frac=0.25, wander_scale=(0.0, 1.2)):
    """clean (B,N) → (noisy (B,N), ref (2,B,N))。`rest_frac` 比例的样本接近静息。"""
    b, n = clean.shape
    ref = synth_reference(b, n, fs, rng)
    # 伪影 = 随机线性混合（+轻微非线性），再叠低频漂移
    c1 = rng.uniform(0.3, 1.5, size=(b, 1))
    c2 = rng.uniform(0.3, 1.5, size=(b, 1))
    m = c1 * ref[:, 0] + c2 * ref[:, 1] + 0.1 * rng.uniform(0, 1) * ref[:, 0] ** 2
    m *= clean.std(axis=1, keepdims=True) / (m.std(axis=1, keepdims=True) + 1e-9)     # 对齐到脉搏尺度
    m *= rng.uniform(0.0, 1.5, size=(b, 1))                                          # 伪影强度
    # 低频基线漂移（腕部/接触压力）
    w = np.cumsum(rng.normal(size=(b, n)), axis=1)
    w = _lowpass(w, fs, 0.6)
    lo, hi = wander_scale
    w *= rng.uniform(lo, hi, size=(b, 1)) * clean.std(axis=1, keepdims=True) / (w.std(axis=1, keepdims=True) + 1e-9)
    noisy = clean + m + w
    # 随机器件响应 + 随机增益
    fc_lp = rng.uniform(4.0, 12.0, size=b)
    fc_hp = rng.uniform(0.3, 0.8, size=b)
    for i in range(b):
        noisy[i] = _lowpass(noisy[i], fs, fc_lp[i]) if fc_lp[i] < fs / 2 * 0.95 else noisy[i]
        noisy[i] = _highpass(noisy[i], fs, fc_hp[i])
    noisy *= rng.uniform(0.5, 2.0, size=(b, 1))
    if rest_frac > 0:                                  # 一部分样本退化为"接近静息"（模拟临床/静息域）
        k = rng.random(b) < rest_frac
        noisy[k] = clean[k] + w[k]
    return noisy, np.transpose(ref, (1, 0, 2))


def load_real_refs(source="wesad", limit=None, fs=FS, max_seconds=60.0):
    """取**真实**运动参考段（只取运动通道，不用其脉搏），用于域随机化的"运动学重放"。

    source="wesad" ：腕部日常活动 ACC（15 受试者；**不是** TROIKA，所以对 TROIKA 留出集仍是零样本）
    source="troika"：TROIKA **训练记录**的 ACC（属于目标域的无标签数据 → 只能算无监督域适配）

    返回 (K, 2, N) 的参考数组（取前两轴，模拟 CIME 的 [ACC_x, Gyro_y] 两通道结构）。
    """
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import ext_loaders as EL
    refs = []
    cap = int(max_seconds * fs)
    if source == "wesad":
        for i, rec in enumerate(EL.iter_wesad(fs_out=fs)):
            a = np.asarray(rec["acc"], dtype=np.float64)
            refs.append(a[:2, :cap])
            if limit and len(refs) >= limit:
                break
    elif source == "troika":
        for i, rec in enumerate(EL.iter_troika(split="train", fs_out=fs)):
            a = np.asarray(rec["acc"], dtype=np.float64)
            refs.append(a[:2, :cap])
            if limit and len(refs) >= limit:
                break
    else:
        raise ValueError(source)
    if not refs:
        raise SystemExit("没有取到真实参考（source=%s）" % source)
    return np.stack(refs)


def synth_batch_real(clean, refs, rng, fs=FS, rest_frac=0.25, wander_scale=(0.0, 1.2)):
    """用**真实运动参考**合成：artifact = f(真实 ACC 段)，脉搏用 CIME clean。

    与 `synth_batch` 的区别：运动参考不再是正弦合成，而是真实腕部/跑步 ACC 的一段，
    因此运动带内的谱结构、幅度调制、非平稳性都更接近真实域。
    """
    b, n = clean.shape
    k = refs.shape[0]
    seg = np.zeros((b, 2, n))
    for i in range(b):
        j = int(rng.integers(0, k))
        src = refs[j]
        st = int(rng.integers(0, max(1, src.shape[1] - n)))
        seg[i] = src[:, st:st + n]
    seg = seg - seg.mean(axis=2, keepdims=True)
    seg = seg / (seg.std(axis=2, keepdims=True) + 1e-9)
    c1 = rng.uniform(0.3, 1.5, size=(b, 1)); c2 = rng.uniform(0.3, 1.5, size=(b, 1))
    m = c1 * seg[:, 0] + c2 * seg[:, 1] + 0.15 * rng.uniform(0, 1) * seg[:, 0] ** 2
    m *= clean.std(axis=1, keepdims=True) / (m.std(axis=1, keepdims=True) + 1e-9)
    m *= rng.uniform(0.0, 1.5, size=(b, 1))
    w = _lowpass(np.cumsum(rng.normal(size=(b, n)), axis=1), fs, 0.6)
    lo, hi = wander_scale
    w *= rng.uniform(lo, hi, size=(b, 1)) * clean.std(axis=1, keepdims=True) / (w.std(axis=1, keepdims=True) + 1e-9)
    noisy = clean + m + w
    fc_lp = rng.uniform(4.0, 12.0, size=b); fc_hp = rng.uniform(0.3, 0.8, size=b)
    for i in range(b):
        if fc_lp[i] < fs / 2 * 0.95:
            noisy[i] = _lowpass(noisy[i], fs, fc_lp[i])
        noisy[i] = _highpass(noisy[i], fs, fc_hp[i])
    noisy *= rng.uniform(0.5, 2.0, size=(b, 1))
    if rest_frac > 0:
        kk = rng.random(b) < rest_frac
        noisy[kk] = clean[kk] + w[kk]
    # 参考通道：真实 ACC 前两轴（正是 RLS 要用的参考）；转成 (2, B, N) 与 synth_batch 一致
    return noisy, np.transpose(seg * rng.uniform(0.5, 3.0, size=(b, 1, 1)), (1, 0, 2))


def features_from(P_, clean, noisy, ref, alpha=None, kind="rls", taps=None, lam=None):
    """按 `pipeline.build_features` 的口径构造 (X, y)：按**含噪 PPG 的段内 std** 缩放。"""
    alpha = P_.RLS_ALPHA_ACTIVE if alpha is None else alpha
    taps = P_.RLS_TAPS if taps is None else taps
    # λ 必须与主特征一致：优先用 active（step2 标定），否则退回常量
    lam = getattr(P_, "RLS_LAMBDA_ACTIVE", P_.RLS_LAMBDA) if lam is None else lam
    scale = noisy.std(axis=1, keepdims=True) + 1e-8
    if kind == "raw":
        return (noisy / scale)[:, None, :].astype(np.float32), (clean / scale).astype(np.float32)
    m_hat, _ = P_.rls_decompose(noisy, ref[0], ref[1], taps=taps, lam=lam)
    ppg_rls = noisy - alpha * m_hat
    X = np.stack([noisy / scale, m_hat / scale, ppg_rls / scale], axis=1)
    return X.astype(np.float32), (clean / scale).astype(np.float32)


if __name__ == "__main__":     # 自检
    rng = np.random.default_rng(0)
    b, n = 8, 3000
    t = np.arange(n) / FS
    clean = np.stack([np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t) for _ in range(b)])
    clean *= 500
    noisy, ref = synth_batch(clean, rng)
    print("clean %s  noisy %s  ref %s" % (clean.shape, noisy.shape, ref.shape))
    print("  伪影/脉搏 幅度比中位 %.2f | 噪声段 std 中位 %.1f | 静息样本数 %d"
          % (np.median((noisy - clean).std(1) / clean.std(1)), np.median(noisy.std(1)),
             int((np.abs(noisy - clean).std(1) / clean.std(1) < 0.1).sum())))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pl", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline", "pipeline.py"))
    P_ = importlib.util.module_from_spec(spec); spec.loader.exec_module(P_)
    for kind in ("rls", "raw"):
        X, y = features_from(P_, clean, noisy, ref, kind=kind)
        print("  kind=%-4s X%s y%s  X 有限值=%s" % (kind, X.shape, y.shape, bool(np.isfinite(X).all())))
    print("[自检] 通过 ✅")

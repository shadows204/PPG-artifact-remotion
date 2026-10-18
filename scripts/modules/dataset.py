"""数据集与特征构造：把 CIME 的 (noisy, clean) 变成 U-Net 需要的张量。

输出契约（与 uEDA-Net 对齐后扩展）：
    X     : (N, C, L)  C = [PPG_raw, M̂, PPG_RLS, ACC_x, Gyro_y]
    y     : (N, L)     干净 PPG（与 PPG_raw 用同一尺度归一化）
    stats : (N, 2)     [运动强度, PPG 谱熵]，训练集标准化
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                            # 同目录模块（rls.py）
sys.path.insert(0, os.path.dirname(HERE))           # <工作区根>/scripts → _common

from _common import paths as P                      # noqa: E402
from rls import rls_decompose                       # noqa: E402

# 数据与产物统一放在 <工作区根>/outputs/ppg_ma/ 下
DEFAULT_DATA_DIR = P.artifact(os.path.join("ppg_ma", "data"))

__all__ = ["load_npz", "build_features", "CMIEDataset", "spectral_entropy", "motion_intensity"]


def load_npz(path):
    d = np.load(path)
    return d["noisy"], d["clean"]


def spectral_entropy(sig, fs=100.0, band=(0.5, 8.0), n_bins=32):
    """归一化谱熵：运动伪影越强/越乱，熵越高。"""
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    n = x.shape[-1]
    spec = np.abs(np.fft.rfft(x * np.hanning(n), axis=-1)) ** 2
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    m = (freq >= band[0]) & (freq <= band[1])
    s = spec[..., m]
    s = s / (s.sum(axis=-1, keepdims=True) + 1e-12)
    ent = -(s * np.log(s + 1e-12)).sum(axis=-1)
    return ent / np.log(s.shape[-1] + 1e-12)


def motion_intensity(acc):
    """log(1 + RMS(acc))，多轴时先求合成 RMS。"""
    a = np.asarray(acc, dtype=np.float64)
    if a.ndim == 3:
        a = np.sqrt((a ** 2).sum(axis=2))
    rms = np.sqrt(np.mean(a ** 2, axis=-1))
    return np.log1p(rms)


def build_features(noisy, clean, fs=100.0, taps=2, forgetting=0.999, eps=0.7,
                   alpha=0.25, verbose=True, chunk=1000):
    """noisy (N, 3, L) = [PPG, ACC_x, Gyro_y]；clean (N, L) -> X, y, stats, scale

    alpha : 收缩系数，PPG_RLS = PPG_raw − alpha · M̂。
            由 scripts/modules/tune_rls.py 在 CIME 测试集上扫出来的最优值是 0.25：
            alpha=1（标准自适应抵消）会把真实脉搏一起减掉，SNR 反而下降，
            所以这里默认只减掉 1/4 的估计伪影。

    归一化：以每段 PPG_raw 的标准差为尺度，PPG/M̂/PPG_RLS 共用同一尺度，
    以保留三者之间的相对幅度（对波形恢复很关键）；ACC/Gyro 各自全局标准化。
    """
    noisy = np.asarray(noisy, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    N, C, L = noisy.shape
    assert C >= 3, "CIME noisy 期望 3 通道 [PPG, ACC_x, Gyro_y]，实际 %d" % C

    ppg = noisy[:, 0, :]
    acc = noisy[:, 1, :]
    gyro = noisy[:, 2, :]

    # ---- RLS：分批跑，避免一次性占太多内存 ----
    m_hat = np.empty_like(ppg)
    ppg_rls = np.empty_like(ppg)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        mh, err, _ = rls_decompose(ppg[s:e], acc[s:e], gyro[s:e],
                                   taps=taps, forgetting=forgetting, eps=eps)
        m_hat[s:e] = mh
        ppg_rls[s:e] = ppg[s:e] - alpha * mh
        if verbose:
            print("  RLS %d/%d" % (e, N), end="\r")
    if verbose:
        print(" " * 30, end="\r")

    # ---- 归一化 ----
    scale = ppg.std(axis=1, keepdims=True) + 1e-8            # (N,1)
    acc_scale = acc.reshape(-1).std() + 1e-8
    gyro_scale = gyro.reshape(-1).std() + 1e-8

    X = np.stack([
        ppg / scale,
        m_hat / scale,
        ppg_rls / scale,
        acc / acc_scale,
        gyro / gyro_scale,
    ], axis=1).astype(np.float32)                            # (N, 5, L)
    y = (clean / scale).astype(np.float32)                   # (N, L)

    # ---- 条件向量 ----
    mi = motion_intensity(noisy[:, 1:, :])                   # (N,)
    se = spectral_entropy(ppg, fs=fs)                        # (N,)
    stats = np.stack([mi, se], axis=1).astype(np.float32)

    return X, y, stats, scale.astype(np.float32)


class CMIEDataset(Dataset):
    """把已构造好的特征包装成 torch Dataset；stats 做训练集标准化。"""

    def __init__(self, X, y, stats, stats_mean=None, stats_std=None):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(1)
        if stats_mean is None:
            stats_mean = stats.mean(axis=0, keepdims=True)
        if stats_std is None:
            stats_std = stats.std(axis=0, keepdims=True) + 1e-8
        self.stats_mean = stats_mean.astype(np.float32)
        self.stats_std = stats_std.astype(np.float32)
        self.stats = torch.from_numpy(
            ((stats - stats_mean) / stats_std).astype(np.float32)
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.stats[i], self.y[i]

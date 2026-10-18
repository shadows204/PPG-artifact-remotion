"""批量化 RLS 自适应滤波器：用 IMU 作为参考，估计 PPG 中的运动伪影。

参数约定沿用 ninja3697/Kernel-Adaptive-Filtering-in-Python 中 padasip 的用法
（FilterRLS(n=..., mu=遗忘因子, w="random", eps=初始化正则)），但本文件是自实现，
不依赖 padasip，并且是"整批窗口并行"的矢量化版本。

语义（与自适应噪声抵消一致）：
    u(n)  参考向量 = [1, acc(n), gyro(n), acc(n-1), gyro(n-1), ...]
    d(n)  期望信号 = 带噪 PPG
    M̂(n) = wᵀu(n)            运动伪影估计（先验，用更新前的 w）
    e(n)  = d(n) − M̂(n)      初步去伪信号 PPG_RLS
"""
from __future__ import annotations

import numpy as np

__all__ = ["build_reference", "BatchedRLS", "rls_decompose"]


def build_reference(acc, gyro, taps: int = 2, extra=None):
    """由 IMU 构造参考矩阵。

    acc, gyro : (B, N) 或 (B, N, K)
    返回 (B, N, d)，第 0 列恒为 1（等效直流偏置项），其后为当前/滞后若干拍的 acc 与 gyro。
    """
    acc = np.asarray(acc, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    if acc.ndim == 2:
        acc = acc[:, :, None]
    if gyro.ndim == 2:
        gyro = gyro[:, :, None]
    B, N, Ka = acc.shape
    Kg = gyro.shape[2]

    cols = [np.ones((B, N, 1), dtype=np.float32)]
    for lag in range(taps):
        shift = np.zeros_like(acc)
        if lag == 0:
            shift = acc
        else:
            shift[:, lag:, :] = acc[:, :-lag, :]
        cols.append(shift)
        shift_g = np.zeros_like(gyro)
        if lag == 0:
            shift_g = gyro
        else:
            shift_g[:, lag:, :] = gyro[:, :-lag, :]
        cols.append(shift_g)
    if extra is not None:                      # 可选：额外通道 (B, N, K)
        cols.append(np.asarray(extra, dtype=np.float32))
    return np.concatenate(cols, axis=2)


class BatchedRLS:
    """整批窗口并行的 RLS。

    forgetting : 遗忘因子 λ（越接近 1 越稳；KAF 演示里用的 0.9966 是短序列场景，
                 我们一段 3000 点，默认取 0.999）
    eps        : 初始化正则，P0 = I / eps
    symmetrize_every : 每隔多少步强制 P 对称，抑制数值漂移
    """

    def __init__(self, forgetting: float = 0.999, eps: float = 0.7,
                 symmetrize_every: int = 200, reset_every: int = 500):
        if not (0.0 < forgetting <= 1.0):
            raise ValueError("forgetting must be in (0, 1]")
        self.lam = float(forgetting)
        self.eps = float(eps)
        self.sym_every = int(symmetrize_every)
        # 长序列下 P 会按 1/λ^n 增长（λ=0.99、3000 点即放大 1e13 倍），
        # 周期性重置 P 是 RLS 的标准做法：保留已学到的 w，只重置协方差。
        self.reset_every = int(reset_every)
        self.n_bad = 0          # 触发过数值保护的窗口数（供诊断）

    def run(self, ref: np.ndarray, desired: np.ndarray):
        """ref (B, N, d), desired (B, N) -> (err (B,N), w_final (B,d))"""
        ref = np.asarray(ref, dtype=np.float32)
        desired = np.asarray(desired, dtype=np.float64)
        B, N, d = ref.shape
        if desired.shape != (B, N):
            raise ValueError("desired shape %s != (%d, %d)" % (desired.shape, B, N))

        # 状态用 float64：原始 ADC 量级可达 1e3，且 P 会放大到 1e13 量级，float32 会溢出
        w = np.zeros((B, d), dtype=np.float64)
        P0 = np.eye(d, dtype=np.float64) / self.eps
        P = np.broadcast_to(P0, (B, d, d)).copy()
        err = np.empty((B, N), dtype=np.float64)
        bad_any = np.zeros(B, dtype=bool)

        for n in range(N):
            u = ref[:, n, :].astype(np.float64)          # (B, d)
            Pu = P @ u[:, :, None]                       # (B, d, 1)
            uPu = np.einsum("bd,bd->b", u, Pu[:, :, 0])  # (B,)
            denom = np.maximum(self.lam + np.maximum(uPu, 0.0), 1e-12)
            gain = Pu[:, :, 0] / denom[:, None]          # (B, d)
            y = np.einsum("bd,bd->b", w, u)              # 先验输出 = M̂
            e = desired[:, n] - y
            err[:, n] = e
            w = w + gain * e[:, None]
            uP = u[:, None, :] @ P                       # (B, 1, d)
            P = (P - gain[:, :, None] * uP) / self.lam

            if self.reset_every and (n + 1) % self.reset_every == 0:
                P = np.broadcast_to(P0, (B, d, d)).copy()   # 保留 w，重置协方差
            if self.sym_every and (n + 1) % self.sym_every == 0:
                P = 0.5 * (P + np.swapaxes(P, 1, 2))
                bad = ~np.isfinite(P).all(axis=(1, 2)) | ~np.isfinite(w).all(axis=1)
                if bad.any():
                    bad_any |= bad
                    P[bad] = P0
                    w[bad] = 0.0
        self.n_bad = int(bad_any.sum())
        return err.astype(np.float32), w


def rls_decompose(ppg: np.ndarray, acc: np.ndarray, gyro: np.ndarray,
                  taps: int = 2, forgetting: float = 0.999, eps: float = 0.7):
    """一步到位：PPG + IMU -> (M̂, PPG_RLS)

    ppg  : (B, N)
    acc  : (B, N) 或 (B, N, K)
    gyro : (B, N) 或 (B, N, K)
    """
    # 入口 NaN 检查：历史上 1 个 NaN 会污染约 100 点并把 w 静默清零（reports/41 审计 P1-11）
    if not (np.isfinite(np.asarray(ppg)).all() and np.isfinite(np.asarray(acc)).all()
            and np.isfinite(np.asarray(gyro)).all()):
        raise ValueError("RLS 输入含 NaN/Inf；请在加载层插值/剔除（禁止静默传播）")
    ref = build_reference(acc, gyro, taps=taps)
    rls = BatchedRLS(forgetting=forgetting, eps=eps)
    err, w = rls.run(ref, ppg)
    m_hat = ppg - err
    return m_hat.astype(np.float32), err.astype(np.float32), w

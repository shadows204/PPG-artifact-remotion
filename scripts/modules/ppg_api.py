#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPG 运动伪影去除 · 软件接口（供软件同学直接调用/封装）。

设计目标
--------
给软件一个**最小、稳定、与训练细节解耦**的入口：

    from ppg_api import PPGDenoiser
    d = PPGDenoiser(preset="lite")                  # 加载模型（约 1 s）
    r = d.denoise(ppg, fs=100, acc=acc, gyro=gyro)  # 输入任意长度
    r.ppg_raw      # 1) 原波形（含伪影）
    r.artifact     # ② 估计的伪影波形 M̂
    r.ppg_clean    # 3) 滤净波形（最终输出）
    d.plot(r, "waveforms.png")

要点
----
1. **任意长度 / 任意采样率**：内部重采样到 100 Hz，30 s 窗 / 15 s 步进、Hann 重叠相加；
2. **三种输入模式**：`imu`（PPG+ACC+GYRO，最佳）、`imu_acc`（只有 ACC）、`no_imu`（只有 PPG）；
3. **静息旁路**：伪影能量占比过低时直接返回原波形，避免"无伪影却过度处理"（实测 κ=0 时过度处理本底 ≈15 PRD）；
4. **确定性**：推理无随机性；`torch.set_grad_enabled(False)`；
5. 不依赖训练代码/数据集，只依赖 `models/ppg_denoise_v1/` 下的权重与 numpy/torch/scipy。
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_PIPELINE = None


def _pipeline():
    """按文件路径加载 scripts/pipeline/pipeline.py（避免与同名目录冲突）。"""
    global _PIPELINE
    if _PIPELINE is None:
        import importlib.util
        # 冻结（PyInstaller）模式下 server 通过环境变量给出 pipeline.py 的实际位置
        path = os.environ.get("PPG_PIPELINE_PY") or os.path.join(SCRIPTS, "pipeline", "pipeline.py")
        spec = importlib.util.spec_from_file_location("ppg_pipeline", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PIPELINE = mod
    return _PIPELINE


FS_MODEL = 100.0          # 模型原生采样率
WIN_S = 30.0              # 模型段长
HOP_S = 15.0              # 推理步进（50% 重叠）
HR_BAND = (0.7, 3.0)      # 心率带 42–180 bpm


# --------------------------------------------------------------------------
# 结果容器
# --------------------------------------------------------------------------
@dataclass
class DenoiseResult:
    t: np.ndarray                  # 时间轴 (s)
    ppg_raw: np.ndarray            # 1) 原波形（已重采样到 100 Hz）
    artifact: np.ndarray           # 2) 伪影波形 M̂（无 IMU 时全 0）
    ppg_prior: np.ndarray          # 物理先验 = ppg_raw − α·M̂
    ppg_clean: np.ndarray          # 3) 滤净波形（最终输出）
    hr_bpm: np.ndarray | None = None
    hr_median: float | None = None
    quality: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = dict(meta=self.meta, quality=self.quality,
                 hr_median=self.hr_median,
                 hr_bpm=(None if self.hr_bpm is None else np.asarray(self.hr_bpm).round(2).tolist()),
                 n_samples=int(len(self.t)),
                 duration_s=float((len(self.t) - 1) / FS_MODEL) if len(self.t) else 0.0)
        return d

    def to_npz(self, path: str) -> str:
        np.savez_compressed(path, t=self.t, ppg_raw=self.ppg_raw, artifact=self.artifact,
                            ppg_prior=self.ppg_prior, ppg_clean=self.ppg_clean,
                            hr_bpm=(self.hr_bpm if self.hr_bpm is not None
                                    else np.zeros(0)))
        return path


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        # (n_lead, N) 或 (N, n_lead)：取"更长"的那一维作时间维，多导联取方差最大的一路
        if a.shape[0] <= 4 and a.shape[0] < a.shape[1]:
            a = a[np.argmax(np.nanvar(a, axis=1))]
        elif a.shape[1] <= 4 and a.shape[1] < a.shape[0]:
            a = a[:, np.argmax(np.nanvar(a, axis=0))]
        else:
            a = a.reshape(-1)
    return a.reshape(-1)


def _as_3ch(x, n: int):
    """(3,N)/(N,3)/(N,) → (3,N)；None → None。"""
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[0] == n:
        a = a.T
    if a.shape[0] > 3:
        a = a[:3]
    elif a.shape[0] < 3:
        a = np.vstack([a, np.zeros((3 - a.shape[0], a.shape[1]))])
    return a


def _norm_imu(x, n: int | None = None):
    """IMU → (n_ch, N)，逐通道 NaN 插值，通道数补齐到 3。None → None。

    ⚠️ 不能用 `_as_1d`：那会把 (3,N) 压成"方差最大的一轴"，丢掉另外两轴
    （早期版本即有此 bug：传 3 轴 ACC 时实际只用 1 轴）。
    """
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    if a.shape[0] > 3 and a.shape[1] <= 3:      # (N, ch) → (ch, N)
        a = a.T
    if a.shape[0] > 3:
        a = a[:3]
    a = np.stack([_fill_nan(row) for row in a])
    if a.shape[0] < 3:
        a = np.vstack([a, np.zeros((3 - a.shape[0], a.shape[1]))])
    return a


def _fill_nan(x: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(x)
    if bad.any():
        idx = np.arange(len(x))
        good = ~bad
        if good.sum() >= 2:
            x = x.copy()
            x[bad] = np.interp(idx[bad], idx[good], x[good])
        else:
            x = np.zeros_like(x)
    return x


def _resample(x: np.ndarray, fs_in: float, fs_out: float = FS_MODEL) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from scipy.signal import resample_poly
    fr = Fraction(fs_out / fs_in).limit_denominator(200)
    return resample_poly(x, fr.numerator, fr.denominator, axis=-1)


def _win_matrix(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = max(1, int(np.ceil((len(x) - win) / hop)) + 1)
    need = (n - 1) * hop + win
    if need > len(x):
        x = np.pad(x, (0, need - len(x)), mode="reflect")
    return np.stack([x[i * hop:i * hop + win] for i in range(n)], axis=0), need


# --------------------------------------------------------------------------
# 主类
# --------------------------------------------------------------------------
class PPGDenoiser:
    """运动伪影去除器。默认 `preset="lite"`（5 残差 + 3 频域 CRM，16 MB，实测 PRD 46.37）。"""

    PRESETS = {
        "lite": dict(residual="residual_rls_*.pt", crm="crm_rls_*.pt", prd_test=46.37),
        "full": dict(residual="residual_rls_*.pt", crm="crm_rls_*.pt", prd_test=46.57),
    }

    def __init__(self, model_dir: str | None = None, preset: str = "lite",
                 device: str = "cpu", alpha: float = 0.40, crm_weight: float = 0.15,
                 bypass_ratio: float = 0.05, taps: int = 4, lam: float = 0.995,
                 verbose: bool = False):
        self.model_dir = model_dir or os.path.join(SCRIPTS, os.pardir, "models",
                                                  "ppg_denoise_v1", preset)
        self.model_dir = os.path.abspath(self.model_dir)
        self.preset = preset
        self.device = device
        self.alpha = float(alpha)
        self.taps = int(taps)          # CIME 标定值（step2_best_rls.json）
        self.lam = float(lam)          # CIME 标定值 0.995（不是模块兜底 0.99）
        self.crm_weight = float(crm_weight)
        self.bypass_ratio = float(bypass_ratio)
        self.verbose = verbose
        self._loaded = False
        self._t_load = 0.0

    # ---- 加载 ----
    def load(self):
        if self._loaded:
            return self
        import glob
        import torch
        import residual as R
        import crm as CRM
        self.torch = torch
        self._R, self._CRM = R, CRM
        # 注意：不要在此全局 set_grad_enabled(False)——那是进程级状态，会让同进程后续训练静默失效；
        # 推理处已逐次使用 torch.no_grad()（见 _forward_ensemble）。
        dev = torch.device(self.device)
        self.dev = dev
        self.res_models, self.crm_models = [], []
        for p in sorted(glob.glob(os.path.join(self.model_dir, "residual_rls_*.pt"))):
            m, meta = R.load(p, device=dev)
            self.res_models.append((m, meta))
        for p in sorted(glob.glob(os.path.join(self.model_dir, "crm_rls_*.pt"))):
            m, meta = CRM.load(p, device=dev)
            self.crm_models.append((m, meta))
        if not self.res_models and not self.crm_models:
            # 回落到无 IMU 预设（models/ppg_denoise_v1/no_imu/ppg_only_*.pt，1 通道）
            alt = os.path.join(os.path.dirname(self.model_dir), "no_imu")
            for p in sorted(glob.glob(os.path.join(alt, "*.pt"))):
                m, meta = R.load(p, device=dev)
                self.res_models.append((m, meta))
            self._no_imu_only = True
            if not self.res_models:
                raise FileNotFoundError("未在 %s（或 %s）找到权重" % (self.model_dir, alt))
        # FiLM 条件标准化参数：优先读 checkpoint 的 cond_mu/cond_sd；否则用训练集统计的常量
        self._cond_mu, self._cond_sd = None, None
        for m, meta in self.res_models:
            if int(meta.get("cond_dim", 0) or 0) > 0 and meta.get("cond_mu") is not None:
                self._cond_mu = np.asarray(meta["cond_mu"], dtype=np.float32)
                self._cond_sd = np.asarray(meta["cond_sd"], dtype=np.float32)
                break
        if self._cond_mu is None:
            self._cond_mu = np.array([7.759, 0.521, 0.525], dtype=np.float32)
            self._cond_sd = np.array([0.998, 0.101, 0.131], dtype=np.float32)
        # 特征溯源：优先用 checkpoint 里记录的 (α,taps,λ)；否则保持构造参数默认值
        for _, meta in self.res_models:
            if meta.get("feat_alpha") is not None:
                self.alpha = float(meta["feat_alpha"])
                self.taps = int(meta["feat_taps"])
                self.lam = float(meta["feat_lam"])
                self._prov = dict(feat_json=meta.get("feat_json"))
                break
        else:
            self._prov = None
        # cond 统计量：优先从 step3_feature_stats.json 读取（不要硬编码）
        _cd_models = [m for m, _ in self.res_models]
        self._cond_dim = int(self.res_models[0][1].get("cond_dim", 0) or 0) if self.res_models else 0
        self._loaded = True
        self._t_load = time.time()
        if self.verbose:
            print("[api] 已加载 %s：残差 %d + CRM %d，device=%s"
                  % (self.preset, len(self.res_models), len(self.crm_models), self.device))
        return self

    # ---- 信息 ----
    def info(self) -> dict:
        return dict(version="ppg_denoise_v1", preset=self.preset, model_dir=self.model_dir,
                    n_residual=len(getattr(self, "res_models", [])),
                    n_crm=len(getattr(self, "crm_models", [])),
                    prd_test=self.PRESETS.get(self.preset, {}).get("prd_test"),
                    fs_model=FS_MODEL, win_s=WIN_S, hop_s=HOP_S, alpha=self.alpha,
                    crm_weight=self.crm_weight, bypass_ratio=self.bypass_ratio,
                    device=self.device, feat_provenance=self._prov)

    # ---- 物理先验 ----
    def _rls(self, ppg, acc, gyro):
        """返回 (m_hat, mode)。mode ∈ {imu, imu_acc, no_imu}。"""
        pl = _pipeline()
        if acc is None:
            return np.zeros_like(ppg), "no_imu"
        if gyro is not None and np.any(gyro):
            # 训练口径是"ACC_x + Gyro_y"：gyro 有 ≥2 轴时取第 1 轴，只有 1 轴时取第 0 轴
            gi = 1 if (gyro.shape[0] >= 2 and np.any(gyro[1])) else 0
            m, _ = pl.rls_decompose(ppg[None, :], acc[0:1].T[None, :],
                                    gyro[gi:gi + 1].T[None, :],
                                    taps=self.taps, lam=self.lam)
            return m[0], "imu"
        m, _ = pl.rls_decompose(ppg[None, :], acc.T[None, :], np.zeros_like(ppg)[None, :],
                                taps=self.taps, lam=self.lam)
        return m[0], "imu_acc"

    # ---- 条件特征（FiLM 模型用）----
    @staticmethod
    def _motion_intensity(acc_w, gyro_w):
        """运动强度：log1p(rms(acc) + rms(gyro))，与 pipeline.motion_intensity 一致。"""
        ra = np.sqrt(np.mean(np.asarray(acc_w, float) ** 2, axis=-1))
        rg = np.sqrt(np.mean(np.asarray(gyro_w, float) ** 2, axis=-1))
        return np.log1p(ra + rg)

    @staticmethod
    def _spectral_entropy(sig_w, fs=FS_MODEL, band=(0.5, 8.0)):
        """与 pipeline.spectral_entropy 一致（Hann、0.5–8 Hz、归一化熵）。"""
        x = np.asarray(sig_w, float)
        x = x - x.mean(axis=-1, keepdims=True)
        spec = np.abs(np.fft.rfft(x * np.hanning(x.shape[-1]), axis=-1)) ** 2
        freq = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
        m = (freq >= band[0]) & (freq <= band[1])
        s_ = spec[..., m]
        s_ = s_ / (s_.sum(axis=-1, keepdims=True) + 1e-12)
        return -(s_ * np.log(s_ + 1e-12)).sum(-1) / np.log(s_.shape[-1] + 1e-12)

    def _cond_batch(self, Wp, Wa, Wacc, Wgyro):
        """**逐窗**条件特征 (B,3) = [运动强度, 谱熵, 伪影占比]（与 build_features 完全同口径）。"""
        mi = self._motion_intensity(Wacc, Wgyro)
        ent = self._spectral_entropy(Wp)
        ratio = np.abs(Wa).mean(axis=1) / (np.abs(Wp).mean(axis=1) + 1e-8)
        return np.stack([mi, ent, ratio], axis=1).astype(np.float32)

    def _forward_ensemble(self, feats, cond):
        """feats: (B,3,L) 标准化后；cond: (B,C) 或 None。按**每个模型自己的** in_channels/cond_dim 取输入。"""
        torch = self.torch
        x = torch.from_numpy(feats).float().to(self.dev)
        n = len(feats)

        def _prep(meta):
            """按模型结构裁输入通道与条件维度（1 通道 no_imu 模型 / cond_dim=2 模型都能正确跑）。"""
            ci = int(meta.get("in_channels", feats.shape[1]))
            xi = x if ci == x.shape[1] else x[:, :ci]
            cd = int(meta.get("cond_dim", 0) or 0)
            if cd > 0:
                if cond is None:
                    raise RuntimeError("模型 cond_dim=%d 但未提供条件向量" % cd)
                st = torch.from_numpy(np.ascontiguousarray(cond[:, :cd])).float().to(self.dev)
            else:
                st = torch.zeros(n, 0, device=self.dev)
            return xi, st

        acc = None
        for m, meta in self.res_models:
            xi, st = _prep(meta)
            with torch.no_grad():
                out = m(xi, st)[:, 0].cpu().numpy()
            acc = out if acc is None else acc + out
        res = acc / max(len(self.res_models), 1) if acc is not None else None
        accm = None
        for m, meta in self.crm_models:
            xi, st = _prep(meta)
            with torch.no_grad():
                out = m(xi, st)[:, 0].cpu().numpy()
            accm = out if accm is None else accm + out
        crm = accm / len(self.crm_models) if accm is not None else None
        if res is None:
            return crm
        if crm is None:
            return res
        w = min(max(self.crm_weight, 0.0), 1.0)
        return (1 - w) * res + w * crm

    # ---- 主接口 ----
    def denoise(self, ppg, fs: float, acc=None, gyro=None, want_hr: bool = True) -> DenoiseResult:
        """输入任意长度的单路（或多导联）PPG 与可选 IMU，返回三种波形。

        参数
        ----
        ppg : (N,) 或 (n_lead, N)  —— 原始 PPG（任意单位；只需波形，不需校准）
        fs  : 采样率 (Hz)
        acc : (3,N)/(N,3)/(N,)      —— 可选，加速度（推荐与 PPG 同侧）
        gyro: 同上                  —— 可选，陀螺仪（有则用"异质两路"参考，实测 WRIST −27%）
        want_hr : 是否同时估计心率（默认 True）

        返回
        ----
        DenoiseResult（字段见类定义）：`ppg_raw` / `artifact` / `ppg_clean` 即三条曲线
        """
        t0 = time.time()
        if not self._loaded:
            self.load()
        ppg = _fill_nan(_as_1d(ppg))
        n_in = len(ppg)
        if n_in < int(fs):                      # 少于 1 s 直接拒绝
            raise ValueError("输入过短：%d 采样（至少 1 s）" % n_in)
        acc = _norm_imu(acc, n_in)
        gyro = _norm_imu(gyro, n_in)

        # 1) 重采样到 100 Hz
        ppg = _resample(ppg, fs)
        if acc is not None:
            acc = np.stack([_resample(a, fs) for a in acc])
        if gyro is not None:
            gyro = np.stack([_resample(g, fs) for g in gyro])
        n = len(ppg)

        # 2) 物理先验（RLS）
        m_hat, mode = self._rls(ppg, acc, gyro)
        prior = ppg - self.alpha * m_hat

        # 3) 滑窗 + 集成前向 + Hann 重叠相加
        win, hop = int(WIN_S * FS_MODEL), int(HOP_S * FS_MODEL)
        if n < win:                                  # 太短 → 反射填充到一个窗
            pad = win - n
            ppg_p = np.pad(ppg, (0, pad), mode="reflect")
            art_p = np.pad(m_hat, (0, pad), mode="reflect")
            pri_p = np.pad(prior, (0, pad), mode="reflect")
            W, need = _win_matrix(ppg_p, win, hop)
            Wa, _ = _win_matrix(art_p, win, hop)
            Wp, _ = _win_matrix(pri_p, win, hop)
        else:
            W, need = _win_matrix(ppg, win, hop)
            Wa, _ = _win_matrix(m_hat, win, hop)
            Wp, _ = _win_matrix(prior, win, hop)
        sd = W.std(axis=1, keepdims=True) + 1e-8
        feats = np.stack([W, Wa, Wp], axis=1) / sd[:, :, None]          # (B,3,L) 逐段标准化
        cond = None
        if any(int(meta.get("cond_dim", 0) or 0) > 0 for _, meta in self.res_models):
            mi = np.empty(len(W)); ent = np.empty(len(W)); rat = np.empty(len(W))
            for i in range(len(W)):
                s0 = i * hop
                pw = W[i]
                aw = Wa[i]
                a_ch = (np.asarray(acc)[:, s0:s0 + win] if acc is not None
                        else np.zeros((1, len(pw))))
                g_ch = (np.asarray(gyro)[:, s0:s0 + win]
                        if (gyro is not None and np.any(gyro)) else np.zeros((1, len(pw))))
                mi[i] = float(np.log1p(np.sqrt(np.mean(a_ch ** 2))
                                       + np.sqrt(np.mean(g_ch ** 2))))
                ent[i] = float(self._spectral_entropy(pw[None, :])[0])
                rat[i] = float(np.abs(aw).mean() / (np.abs(pw).mean() + 1e-8))
            base_c = np.stack([mi, ent, rat], axis=1).astype(np.float32)
            cond = ((base_c - self._cond_mu) / self._cond_sd).astype(np.float32)
        out_w = self._forward_ensemble(feats.astype(np.float32), cond) * sd  # (B,L)
        # Hann 重叠相加
        win_h = np.hanning(win + 2)[1:-1]
        num = np.zeros(need)
        den = np.zeros(need)
        for i in range(len(out_w)):
            s = i * hop
            num[s:s + win] += out_w[i] * win_h
            den[s:s + win] += win_h
        clean = (num / np.maximum(den, 1e-8))[:n]

        # 4) 静息旁路（避免无伪影时过度处理）
        art_ratio = float(np.std(m_hat) / (np.std(ppg) + 1e-12))
        # no_imu 模式下 M̂≡0 → 旁路判据恒成立，会让 1 通道模型永远不生效（历史 bug，reports/41 P0）
        bypass = bool(mode != "no_imu" and art_ratio < self.bypass_ratio)
        if bypass:
            clean = ppg.copy()

        # 5) 心率（可选）
        hr, hr_med = None, None
        if want_hr:
            try:
                pl = _pipeline()
                n_win = int(max(0, (n - 8 * FS_MODEL) // (2 * FS_MODEL)) + 1)
                if n_win >= 3:
                    accmag = (np.sqrt((acc ** 2).sum(0)) if acc is not None
                              else np.zeros(n))
                    # HR 从**最终输出**估计（原用 ppg−M̂ = α=1，与输出/先验的 α=0.40 不一致）
                    f, P_ = pl.psd_matrix(clean, accmag, FS_MODEL, n_win,
                                          win_s=8.0, hop_s=2.0, use_notch=False)
                    hr = pl.sel_viterbi(f, P_, band=HR_BAND, jump_bpm=4.0, lam=0.5)
                    ok = np.isfinite(hr)
                    hr_med = float(np.nanmedian(hr[ok])) if ok.any() else None
            except Exception:
                hr, hr_med = None, None

        # 6) 质量指标
        try:
            _resid = float(np.var(ppg - clean))
            # 标准"输出信噪比"；旁路/残差≈0 时不应给出 +171 dB 的荒唐值 → 返回 None
            band_snr = (None if (_resid <= 1e-12 or bypass)
                        else float(10 * np.log10(float(np.var(clean)) / _resid)))
        except Exception:
            band_snr = None
        quality = dict(artifact_ratio=round(art_ratio, 4),
                       bypass_applied=bypass,
                       snr_improve_db=round(band_snr, 2) if band_snr is not None else None,
                       input_std=round(float(np.std(ppg)), 6))
        meta = dict(version="ppg_denoise_v1", preset=self.preset, mode=mode,
                    fs_in=float(fs), fs_model=FS_MODEL, n_in=int(n_in), n_out=int(n),
                    alpha=self.alpha, crm_weight=self.crm_weight,
                    n_residual=len(self.res_models), n_crm=len(self.crm_models),
                    elapsed_s=round(time.time() - t0, 3))
        return DenoiseResult(t=np.arange(n) / FS_MODEL, ppg_raw=ppg, artifact=m_hat,
                             ppg_prior=prior, ppg_clean=clean, hr_bpm=hr,
                             hr_median=hr_med, quality=quality, meta=meta)

    # ---- 画图（软件可直接复用）----
    def plot(self, result: DenoiseResult, path: str | None = None,
             reference=None, title: str | None = None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        # 中文字体（系统缺失时自动回退，不影响出图）
        for fam in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
                    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "SimHei"):
            try:
                font_manager.findfont(fam, fallback_to_default=False)
                plt.rcParams["font.sans-serif"] = [fam]
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        t = result.t
        axes[0].plot(t, result.ppg_raw, color="#1f77b4", lw=0.9)
        axes[0].set_ylabel("1) 原波形\n(含伪影)")
        axes[0].set_title(title or "PPG 运动伪影去除结果")
        axes[1].plot(t, result.artifact, color="#d62728", lw=0.9)
        axes[1].set_ylabel("2) 伪影波形 M̂")
        axes[2].plot(t, result.ppg_clean, color="#2ca02c", lw=0.9, label="滤净输出")
        if reference is not None:
            ref = np.asarray(reference, float).reshape(-1)
            m = min(len(ref), len(t))
            axes[2].plot(t[:m], ref[:m], color="#7f7f7f", lw=0.8, ls="--",
                         label="参考（如有）")
            axes[2].legend(loc="upper right", fontsize=8)
        axes[2].set_ylabel("3) 滤净波形")
        axes[2].set_xlabel("时间 (s)")
        info = "HR 中位 %s bpm ｜ 伪影占比 %.3f ｜ 旁路 %s ｜ 模式 %s" % (
            ("—" if result.hr_median is None else "%.1f" % result.hr_median),
            result.quality.get("artifact_ratio", 0), result.quality.get("bypass_applied"),
            result.meta.get("mode"))
        fig.text(0.5, 0.005, info, ha="center", fontsize=9, color="#333333")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.tight_layout(rect=(0, 0.025, 1, 1))
        if path:
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path
        return fig

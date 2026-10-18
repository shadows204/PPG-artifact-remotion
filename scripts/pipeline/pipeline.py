#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPG 运动伪影抑制流水线：RLS 自适应滤波 + 1D Attention/条件 U-Net（单文件实现）

设计原则：**每一步都有落盘产出**，上一步的输出就是下一步的输入，可单独重跑。

产物根目录（可用环境变量 PPG_ARTIFACT_DIR 覆盖）：
    <工作区根>/outputs/rls_unet/
        ├── data/       step1 产出的 float32 npz
        ├── features/   step3 产出的网络输入张量
        ├── weights/    step4 产出的模型权重
        └── results/    全部 json / csv / png / md（日志里简写为 results/xxx）

    step1_prepare   数据准备与体检      -> data/*.npz                    + results/step1_data_report.json
    step2_tune_rls  RLS 参数标定        -> results/step2_rls_tuning.csv  + results/step2_best_rls.json
    step3_features  构造网络输入特征     -> features/train.npz, test.npz
    step4_train     训练消融模型        -> weights/*.pt                 + results/step4_train_log.csv
    step5_eval      CIME 波形恢复评估    -> results/step5_metrics.json   + results/step5_ablation.csv
    step6_hr        IEEEPPG 心率评估     -> results/step6_hr_metrics.json
    step7_report    汇总成报告           -> results/REPORT.md
    step8_verdict   多种子统计判决        -> results/VERDICT.md, results/verdict.json
    step9_hr2       TROIKA 训练集心率消融  -> results/step9_hr_table.csv
    step10_hr_test  TROIKA 官方留出集验证  -> results/step10_hr_test.csv
    step11_strata   运动强度分层实验       -> results/step11_motion_strata.csv/.json/.png

用法（在**工作区根**目录下执行）：
    python scripts/pipeline/pipeline.py all             # 一键全流程
    python scripts/pipeline/pipeline.py step2_tune_rls  # 只跑某一步
    python scripts/pipeline/pipeline.py all --quick     # 小样本快速验证（几分钟）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# 0. 路径与常量
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# 统一路径解析：<工作区根>/scripts/_common/paths.py
sys.path.insert(0, os.path.dirname(HERE))          # <工作区根>/scripts
from _common import paths as P                     # noqa: E402

# 产物根：默认 <工作区根>/outputs/rls_unet（data / features / weights / results）
PROJ = os.environ.get("PPG_ARTIFACT_DIR") or P.artifact("rls_unet")
DATA_ROOT = P.datasets_root()                      # 环境变量 PPG_DATASETS 可覆盖
CIME_MAT_DIR = os.path.join(DATA_ROOT, "CIME_PPG_dataset")
TROIKA_DIR = os.path.join(DATA_ROOT, "IEEEPPG_TROIKA")

DATA_DIR = os.path.join(PROJ, "data")
FEAT_DIR = os.path.join(PROJ, "features")
W_DIR = os.path.join(PROJ, "weights")
OUT_DIR = os.environ.get("PPG_RESULT_DIR") or os.path.join(PROJ, "results")
for d in (DATA_DIR, FEAT_DIR, OUT_DIR, W_DIR):
    os.makedirs(d, exist_ok=True)

FS_CIME = 100.0        # CIME 发布版采样率
FS_TROIKA = 125.0      # IEEE SP Cup 2015
SEG_LEN = 3000         # 30 s @100 Hz，与 CIME 官方 cell 长度一致
UNET_MULT = 32         # 5 级 stride=2
RLS_TAPS = 4           # step2 标出的滞后拍数
RLS_LAMBDA = 0.99      # 兜底遗忘因子（step2 未跑时使用）
RLS_ALPHA = 0.25       # 兜底收缩系数（step2 未跑时使用）
RLS_RESET = 500        # P 协方差重置周期（数值稳定）
HR_BAND = (0.6, 3.3)   # 心率搜索带 36~198 bpm
VIT_LAM = 4.0          # Viterbi 平滑权重（在 TROIKA 训练集上扫出）
VIT_JUMP = 8.0         # Viterbi 跳变尺度 (bpm)
NOTCH_W = 0.08         # 运动谐波陷波半宽 (Hz)：宽陷波会误删真峰
# 参考维度受控实验（step12）用的两套 α：2 通道 / 3 轴
RLS_ALPHA_2CH = 0.25
RLS_ALPHA_3AX = 1.00


# CIME 权威标定（outputs/final_10seed/results/step2_best_rls.json，训练集 5000 段上标定）
# ⚠️ 历史坑：`outputs/rls_unet/results/step2_best_rls.json` 是**早期 200 段快标定**（α=0.25/λ=0.99），
#    若 OUT_DIR 指向它，α/λ 会与已归档特征（α=0.40/λ=0.995）不一致且**静默**发生。现已显式告警。
RLS_LAMBDA_TUNED = 0.995
RLS_ALPHA_TUNED = 0.40


def _step2_json_path():
    return os.environ.get("PPG_STEP2_JSON") or os.path.join(OUT_DIR, "step2_best_rls.json")


def _tuned_rls(verbose=True):
    """读 step2 标定（taps/lam/alpha 的**唯一真源**），与权威常量不一致时**显式告警**。"""
    path = _step2_json_path()
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        taps, lam, alpha = int(d["taps"]), float(d["lam"]), float(d["alpha"])
        if verbose and (taps, lam, alpha) != (RLS_TAPS, RLS_LAMBDA_TUNED, RLS_ALPHA_TUNED):
            print("[rls][warn] 当前标定与 CIME 权威标定不同：来源 %s（n_segments=%s）→ "
                  "taps=%d λ=%.4f α=%.2f；权威为 taps=%d λ=%.4f α=%.2f。"
                  "若要用权威标定，设 PPG_STEP2_JSON=outputs/final_10seed/results/step2_best_rls.json"
                  % (path, d.get("n_segments"), taps, lam, alpha,
                     RLS_TAPS, RLS_LAMBDA_TUNED, RLS_ALPHA_TUNED))
        return taps, lam, alpha
    except Exception as e:
        if verbose:
            print("[rls][warn] 读取标定失败（%s：%s）→ 退回权威标定常量 taps=%d λ=%.4f α=%.2f"
                  % (path, type(e).__name__, RLS_TAPS, RLS_LAMBDA_TUNED, RLS_ALPHA_TUNED))
        return RLS_TAPS, RLS_LAMBDA_TUNED, RLS_ALPHA_TUNED


def _best_alpha():
    """收缩系数 α（兼容旧接口）。"""
    return _tuned_rls()[2]


_RLS_TUNED = _tuned_rls()


def _rls_kw():
    """调用时读取标定（避免 import 时快照）：返回 rls_decompose 的 taps/lam。"""
    taps, lam, _ = _tuned_rls(verbose=False)
    return dict(taps=taps, lam=lam)


def _active_alpha():
    """调用时读取收缩系数 α（step2 重写 json 后立即生效）。"""
    return _tuned_rls(verbose=False)[2]
RLS_ALPHA_ACTIVE = _RLS_TUNED[2]        # 与特征口径相关的消费者统一用它
RLS_LAMBDA_ACTIVE = _RLS_TUNED[1]        # 同上（λ 也必须统一，否则 M̂ 口径不一致）

# 消融阶梯：每一格只多给网络一种信息，用来分离 IMU 与 RLS 各自的贡献
VARIANTS = {
    "ppg_only": dict(xkey="Xppg",     cond="none",   desc="仅 PPG（无 IMU、无 RLS）"),
    "rls_only": dict(xkey="XrlsOnly", cond="none",   desc="PPG + M̂ + PPG_RLS（无 IMU 输入）"),
    "no_rls":   dict(xkey="XnoRLS",   cond="stats2", desc="PPG + IMU（无 RLS）"),
    "rls_cond": dict(xkey="XnoRLS",   cond="stats3", desc="PPG + IMU，M̂ 作条件调制（不拼接）"),
    "full":     dict(xkey="X",        cond="stats2", desc="PPG + M̂ + PPG_RLS + IMU"),
    # E2 补格：① FiLM 消融（同输入、关掉运动条件注入）；② 最优通道集 + 运动条件
    "full_nofilm": dict(xkey="X",        cond="none",   desc="PPG + M̂ + PPG_RLS + IMU，关 FiLM（FiLM 消融）"),
    "rls_film":    dict(xkey="XrlsOnly", cond="stats2", desc="PPG + M̂ + PPG_RLS + 运动条件（原始 IMU 不进网络）"),
}
COND_DIM = {"none": 0, "stats2": 2, "stats3": 3}

# 带通基线参数（E1 传统滤波对照）
BP_BAND = (0.5, 4.0)   # Hz，覆盖 30~240 bpm
BP_ORDER = 4


def log(msg):
    print(msg, flush=True)


def jdump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def jload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def wcsv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ----------------------------------------------------------------------------
# 1. 数据准备与体检
# ----------------------------------------------------------------------------
def step1_prepare(args):
    """CIME .mat -> float32 npz，并输出数据体检报告。"""
    t0 = time.time()
    import scipy.io as sio

    report = {"source": CIME_MAT_DIR, "splits": {}}
    for mat_name, npz_name, key in [
        ("train_all_noise.mat", "cime_train.npz", "train"),
        ("test_all_noise.mat", "cime_test.npz", "test"),
    ]:
        mat_path = os.path.join(CIME_MAT_DIR, mat_name)
        npz_path = os.path.join(DATA_DIR, npz_name)
        # 优先用已转换的 npz（本工作区可能没有原始 .mat）
        _npz_ok = os.path.exists(npz_path) and not args.force
        if not _npz_ok and not os.path.exists(mat_path):
            raise SystemExit("既没有 %s，也没有 %s，请确认数据集已下载/转换" % (npz_path, mat_path))
        # 官方说明：列按受试者分块。train = 10 受试者 × 10 组 × 10 sequence = 100 列/人；
        # test = 38 受试者 × 1 组 × 10 sequence = 10 列/人。列内 5 行是同 sequence 的 5 个 30 s 段。
        per_subject = 100 if key == "train" else 10
        if os.path.exists(npz_path) and not args.force:
            d = np.load(npz_path)
            noisy, clean = d["noisy"], d["clean"]
            log("[step1] %-18s 已存在，跳过转换  noisy=%s clean=%s" % (npz_name, noisy.shape, clean.shape))
        else:
            d = sio.loadmat(mat_path, variable_names=["noise_input_all", "noise_output_all"])
            ni, no = d["noise_input_all"], d["noise_output_all"]
            rows, cols = ni.shape
            n, c, L = rows * cols, ni[0, 0].shape[0], ni[0, 0].shape[1]
            noisy = np.empty((n, c, L), dtype=np.float32)
            clean = np.empty((n, L), dtype=np.float32)
            subj = np.empty(n, dtype=np.int32)
            k = 0
            for i in range(rows):
                for j in range(cols):
                    noisy[k] = ni[i, j].astype(np.float32)
                    clean[k] = no[i, j].astype(np.float32).reshape(-1)
                    subj[k] = j // per_subject
                    k += 1
            np.savez_compressed(npz_path, noisy=noisy, clean=clean, subject=subj)
            log("[step1] %-18s -> %s  noisy=%s (%.1f MB, %d 名受试者)"
                % (mat_name, npz_name, noisy.shape, os.path.getsize(npz_path) / 1024 / 1024,
                   len(np.unique(subj))))

        report["splits"][key] = {
            "file": npz_name,
            "n_segments": int(noisy.shape[0]),
            "n_channels": int(noisy.shape[1]),
            "seg_len": int(noisy.shape[2]),
            "fs": FS_CIME,
            "duration_s": round(noisy.shape[2] / FS_CIME, 1),
            "channels": ["PPG_noisy", "ACC_x", "Gyro_y"],
            "clean_available": True,
            "amp_std": {
                "ppg_noisy": float(np.std(noisy[:, 0])),
                "acc_x": float(np.std(noisy[:, 1])),
                "gyro_y": float(np.std(noisy[:, 2])),
                "ppg_clean": float(np.std(clean)),
            },
        }
    report["elapsed_s"] = round(time.time() - t0, 1)
    jdump(os.path.join(OUT_DIR, "step1_data_report.json"), report)
    log("[step1] 完成 -> results/step1_data_report.json")
    for k, v in report["splits"].items():
        log("        %-5s %d 段 x %d 通道 x %d 点 (%.1f s)" % (k, v["n_segments"], v["n_channels"], v["seg_len"], v["duration_s"]))
    return report


# ----------------------------------------------------------------------------
# 2. RLS 实现 + 参数标定
# ----------------------------------------------------------------------------
def build_reference(acc, gyro, taps):
    """(B,N[,K]) -> (B,N,d)，第 0 列为常数项，其后是当前与滞后的 acc/gyro。"""
    acc = np.asarray(acc, dtype=np.float32)
    gyro = np.asarray(gyro, dtype=np.float32)
    if acc.ndim == 2:
        acc = acc[:, :, None]
    if gyro.ndim == 2:
        gyro = gyro[:, :, None]
    B, N, _ = acc.shape
    cols = [np.ones((B, N, 1), dtype=np.float32)]
    for lag in range(taps):
        a = np.zeros_like(acc); g = np.zeros_like(gyro)
        if lag == 0:
            a, g = acc, gyro
        else:
            a[:, lag:, :] = acc[:, :-lag, :]
            g[:, lag:, :] = gyro[:, :-lag, :]
        cols += [a, g]
    return np.concatenate(cols, axis=2)


def rls_run(ref, desired, lam=RLS_LAMBDA, eps=0.7, reset_every=RLS_RESET):
    """批量 RLS。状态用 float64：P 会按 1/λⁿ 增长（λ=0.99、3000 点约 1e13 倍），float32 会溢出。

    返回 (err, w)；M̂ = desired - err
    """
    ref = np.asarray(ref, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float64)
    B, N, d = ref.shape
    w = np.zeros((B, d), dtype=np.float64)
    P0 = np.eye(d, dtype=np.float64) / eps
    P = np.broadcast_to(P0, (B, d, d)).copy()
    err = np.empty((B, N), dtype=np.float64)
    n_bad = 0
    for n in range(N):
        u = ref[:, n, :].astype(np.float64)
        Pu = P @ u[:, :, None]
        uPu = np.einsum("bd,bd->b", u, Pu[:, :, 0])
        denom = np.maximum(lam + np.maximum(uPu, 0.0), 1e-12)
        gain = Pu[:, :, 0] / denom[:, None]
        e = desired[:, n] - np.einsum("bd,bd->b", w, u)
        err[:, n] = e
        w = w + gain * e[:, None]
        P = (P - gain[:, :, None] * (u[:, None, :] @ P)) / lam
        if reset_every and (n + 1) % reset_every == 0:
            P = np.broadcast_to(P0, (B, d, d)).copy()
        if (n + 1) % 200 == 0:
            P = 0.5 * (P + np.swapaxes(P, 1, 2))
            bad = ~np.isfinite(P).all(axis=(1, 2)) | ~np.isfinite(w).all(axis=1)
            if bad.any():
                n_bad += int(bad.sum()); P[bad] = P0; w[bad] = 0.0
    return err.astype(np.float32), w, n_bad


def _seg_std(x):
    """按段求标准差，保持可广播形状：(B, ...) -> (B, 1, ...)"""
    B = x.shape[0]
    s = x.reshape(B, -1).std(axis=-1) + 1e-12
    return s.reshape((B,) + (1,) * (x.ndim - 1))


def rls_decompose(ppg, acc, gyro, taps=RLS_TAPS, lam=RLS_LAMBDA, eps=0.7, normalize=True):
    """RLS 运动伪影抵消。

    normalize=True 时先把 PPG 与 IMU 各自按段标准化再自适应：
    CIME 的 ACC 是原始 ADC 量级（std~2e2）而 PPG std~1e3，TROIKA 的 ACC 单位是 g
    （std~0.7）而 PPG std~1.5e2 —— 不归一化的话同一个 λ/eps 在两个数据集上
    的等效步长完全不同，跨数据集直接失效。归一化让超参可迁移。
    """
    p = np.asarray(ppg, dtype=np.float64)
    a = np.asarray(acc, dtype=np.float64)
    g = np.asarray(gyro, dtype=np.float64)
    if normalize:
        ps, as_, gs = _seg_std(p), _seg_std(a), _seg_std(g)
        err_n, _, n_bad = rls_run(build_reference(a / as_, g / gs, taps), p / ps, lam, eps)
        err = err_n * ps
    else:
        err, _, n_bad = rls_run(build_reference(a, g, taps), p, lam, eps)
    return p - err, err          # (M̂, PPG_RLS@α=1)


def snr_db(signal, clean):
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    c = np.asarray(clean, dtype=np.float64).reshape(-1)
    pn = np.mean((s - c) ** 2)
    return float(10 * np.log10(np.mean(c ** 2) / pn)) if pn > 0 else float("inf")


def prd(clean, pred):
    c = np.asarray(clean, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    return float(100 * np.linalg.norm(p - c) / (np.linalg.norm(c) + 1e-12))


def step2_tune_rls(args):
    """在 CIME **训练集**上扫 λ / taps / α，选出真正能改善 SNR 的配置。

    ⚠️ 口径修正（2026-10-14）：此前版本默认在**测试集**前 200 段上标定超参，
    属于调参泄漏；且前 200 段比整体容易（SNR 4.70 vs 全测试集 2.10 dB），
    标出的 α=0.25 在完整测试集上并非最优（全量最优约 0.50）。
    现默认改为在训练集上标定，并用**等间隔抽样**覆盖所有受试者。
    `--tune-on test` 可复现旧的泄漏口径，仅用于对比说明。
    """
    t0 = time.time()
    split = getattr(args, "tune_on", "train")
    npz = "cime_train.npz" if split == "train" else "cime_test.npz"
    d = np.load(os.path.join(DATA_DIR, npz))
    total = d["noisy"].shape[0]
    n = min(args.tune_n, total)
    idx = np.linspace(0, total - 1, n).astype(int) if n < total else np.arange(total)
    noisy, clean = d["noisy"][idx], d["clean"][idx]
    ppg, acc, gyro = noisy[:, 0], noisy[:, 1], noisy[:, 2]

    base_snr = float(np.nanmean([snr_db(ppg[i], clean[i]) for i in range(n)]))
    base_prd = float(np.nanmean([prd(clean[i], ppg[i]) for i in range(n)]))
    log("[step2] 标定集 = CIME %s（%d/%d 段，等间隔抽样），基线 PPG_raw: SNR %.2f dB, PRD %.2f%%"
        % (split, n, total, base_snr, base_prd))

    rows, best = [], None
    for lam in args.lams:
        for taps in args.taps_list:
            m_hat, _ = rls_decompose(ppg, acc, gyro, taps=taps, lam=lam)
            b = None
            for a in args.alphas:
                out = ppg - a * m_hat
                s = float(np.nanmean([snr_db(out[i], clean[i]) for i in range(n)]))
                p = float(np.nanmean([prd(clean[i], out[i]) for i in range(n)]))
                if b is None or s > b[1]:
                    b = (a, s, p)
            rows.append([lam, taps, b[0], round(b[1], 3), round(b[1] - base_snr, 3), round(b[2], 3),
                         round(base_prd - b[2], 3)])
            log("        λ=%.4f taps=%d -> 最优α=%.2f  SNR %.2f dB (Δ%+.2f)  PRD %.2f%%"
                % (lam, taps, b[0], b[1], b[1] - base_snr, b[2]))
            # 赛题指标是 PRD → 按 PRD 选（历史按 SNR 选得到 α=0.25，与 PRD 最优 0.40 不一致）
            if best is None or b[2] < best["prd"]:
                best = dict(lam=lam, taps=taps, alpha=b[0], snr=b[1],
                            delta_snr=b[1] - base_snr, prd=b[2])

    wcsv(os.path.join(OUT_DIR, "step2_rls_tuning.csv"), rows,
         ["lambda", "taps", "best_alpha", "snr_db", "delta_snr_db", "prd_pct"])
    best.update(base_snr=base_snr, base_prd=base_prd, n_segments=n, tune_split=split,
                elapsed_s=round(time.time() - t0, 1))
    jdump(os.path.join(OUT_DIR, "step2_best_rls.json"), best)
    log("[step2] 最佳: λ=%.4f taps=%d α=%.2f  ΔSNR=%+.2f dB" % (best["lam"], best["taps"], best["alpha"], best["delta_snr"]))
    log("[step2] 完成 -> results/step2_rls_tuning.csv, results/step2_best_rls.json")
    if best["delta_snr"] <= 1.0:
        log("[step2] 注意：RLS 单独增益很小，且最优 α<1 说明存在『信号抵消』——")
        log("        RLS 应定位为运动伪影估计/先验，最终恢复交给 U-Net。")
    return best


# ----------------------------------------------------------------------------
# 3. 特征构造
# ----------------------------------------------------------------------------
def spectral_entropy(sig, fs=FS_CIME, band=(0.5, 8.0)):
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.shape[-1]), axis=-1)) ** 2
    freq = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    m = (freq >= band[0]) & (freq <= band[1])
    s = spec[..., m]
    s = s / (s.sum(axis=-1, keepdims=True) + 1e-12)
    return -(s * np.log(s + 1e-12)).sum(-1) / np.log(s.shape[-1] + 1e-12)


def motion_intensity(acc, gyro):
    """运动强度：ACC 与 Gyro 的合成 RMS 取 log。"""
    a = np.asarray(acc, dtype=np.float64)
    g = np.asarray(gyro, dtype=np.float64)
    ra = np.sqrt(np.mean(a ** 2, axis=-1) if a.ndim == 3 else np.mean(a ** 2, axis=-1))
    rg = np.sqrt(np.mean(g ** 2, axis=-1) if g.ndim == 3 else np.mean(g ** 2, axis=-1))
    return np.log1p(ra + rg)


def build_features(noisy, clean, alpha, taps, lam, verbose=True):
    """一次算出全部消融阶梯需要的输入分组。

    返回 dict：X(5ch) / XnoRLS(3ch) / Xppg(1ch) / XrlsOnly(3ch) / y / stats2 / stats3 / scale
    """
    noisy = np.asarray(noisy, dtype=np.float32)
    clean = np.asarray(clean, dtype=np.float32)
    ppg, acc, gyro = noisy[:, 0], noisy[:, 1], noisy[:, 2]

    m_hat, _ = rls_decompose(ppg, acc, gyro, taps=taps, lam=lam)
    ppg_rls = ppg - alpha * m_hat

    scale = ppg.std(axis=1, keepdims=True) + 1e-8
    acc_s = acc.reshape(-1).std() + 1e-8
    gyro_s = gyro.reshape(-1).std() + 1e-8

    f = lambda *ch: np.stack(ch, axis=1).astype(np.float32)
    X = f(ppg / scale, m_hat / scale, ppg_rls / scale, acc / acc_s, gyro / gyro_s)
    XnoRLS = f(ppg / scale, acc / acc_s, gyro / gyro_s)          # 不给 RLS 任何信息
    Xppg = f(ppg / scale)                                        # 连 IMU 都不给
    XrlsOnly = f(ppg / scale, m_hat / scale, ppg_rls / scale)     # 只给 RLS，不给 IMU

    y = (clean / scale).astype(np.float32)
    stats2 = np.stack([motion_intensity(acc, gyro), spectral_entropy(ppg)], axis=1).astype(np.float32)
    # stats3：把 RLS 的产物"伪影占比"作为第三个条件，用于 rls_cond 变体（条件调制而非拼接）
    artifact_ratio = (np.abs(m_hat).mean(axis=1) / (np.abs(ppg).mean(axis=1) + 1e-8)).astype(np.float32)
    stats3 = np.stack([stats2[:, 0], stats2[:, 1], artifact_ratio], axis=1).astype(np.float32)

    if verbose:
        log("        X=%s XnoRLS=%s Xppg=%s XrlsOnly=%s" % (X.shape, XnoRLS.shape, Xppg.shape, XrlsOnly.shape))
    return dict(X=X, XnoRLS=XnoRLS, Xppg=Xppg, XrlsOnly=XrlsOnly, y=y,
                stats2=stats2, stats3=stats3, scale=scale.astype(np.float32))


def subject_ids(npz, split):
    """取受试者编号；旧 npz 没有 subject 字段时按官方列布局回推。"""
    if "subject" in npz.files:
        return npz["subject"]
    n = npz["noisy"].shape[0]
    cols = 1000 if split == "train" else 380
    per = 100 if split == "train" else 10
    # ⚠️ 这是按 CIME 官方列布局**假设**回推的（10 受试者/块）；若真实布局不同会导致验证集串人。
    # 当前所有归档 features/*.npz 都带 subject 字段，不会走到这里；仅在从原始 npz 重建特征时告警。
    print("[warn] features npz 缺少 subject 字段 → 按假设布局回推（train: %d 段/块, test: %d 段/块）；"
          "若与真实布局不符，验证集划分会出错，请改用带 subject 的特征文件。" % (per, per))
    return ((np.arange(n) % cols) // per).astype(np.int32)


def step3_features(args):
    t0 = time.time()
    best = jload(os.path.join(OUT_DIR, "step2_best_rls.json"))
    tr = np.load(os.path.join(DATA_DIR, "cime_train.npz"))
    te = np.load(os.path.join(DATA_DIR, "cime_test.npz"))
    if args.quick:
        tr_n, te_n = 512, 256
    else:
        tr_n, te_n = tr["noisy"].shape[0], te["noisy"].shape[0]
    log("[step3] 用 λ=%.4f taps=%d α=%.2f 构造特征" % (best["lam"], best["taps"], best["alpha"]))

    out = {}
    for name, d, n in [("train", tr, tr_n), ("test", te, te_n)]:
        noisy, clean = d["noisy"][:n], d["clean"][:n]
        feats = build_features(noisy, clean, best["alpha"], best["taps"], best["lam"], verbose=True)
        feats["subject"] = subject_ids(d, name)[:n]
        np.savez_compressed(os.path.join(FEAT_DIR, "%s.npz" % name), **feats)
        out[name] = dict(X=list(feats["X"].shape), y=list(feats["y"].shape),
                         n_subjects=int(len(np.unique(feats["subject"]))),
                         path="features/%s.npz" % name)
        log("[step3] %-5s -> features/%s.npz  X=%s  %d 名受试者"
            % (name, name, feats["X"].shape, out[name]["n_subjects"]))

    # 条件向量标准化参数（用训练集统计）
    d = np.load(os.path.join(FEAT_DIR, "train.npz"))
    jdump(os.path.join(OUT_DIR, "step3_feature_stats.json"),
          dict(stats2_mean=d["stats2"].mean(0).tolist(), stats2_std=d["stats2"].std(0).tolist(),
               stats3_mean=d["stats3"].mean(0).tolist(), stats3_std=d["stats3"].std(0).tolist(),
               splits=out, elapsed_s=round(time.time() - t0, 1)))
    log("[step3] 完成 -> features/*.npz, results/step3_feature_stats.json")
    return out


# ----------------------------------------------------------------------------
# 4. 模型与训练
# ----------------------------------------------------------------------------
def torch_import():
    import torch
    import torch.nn as nn
    return torch, nn


def make_model(torch, nn, in_channels, cond_dim, norm="bn"):
    """cond_dim = 0 表示不做条件注入（消融用）；2 = [运动强度, 谱熵]；3 = 再加 RLS 伪影占比。

    norm：归一化层（`reports/21`/`24`）——`bn`=BatchNorm（1 通道路径必须用它，AdaBN 依赖）；
          `in`=InstanceNorm（3 通道路径域内 PRD 更优，−2.08）。
    """
    use_film = cond_dim > 0
    try:
        from models import make_norm as _mk_norm          # scripts/modules 在 sys.path
    except Exception:                                     # 兜底：等价实现
        def _mk_norm(kind, ch):
            kind = (kind or "bn").lower()
            if kind == "in":
                return nn.InstanceNorm1d(ch, affine=True, track_running_stats=False)
            if kind == "gn":
                g = 8 if ch % 8 == 0 else (4 if ch % 4 == 0 else 1)
                return nn.GroupNorm(g, ch)
            return nn.BatchNorm1d(ch)

    class FiLM(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.g = nn.Linear(cond_dim, ch); self.b = nn.Linear(cond_dim, ch)
            nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)
            nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)

        def forward(self, x, s):
            return x * (self.g(s).unsqueeze(-1) + 1.0) + self.b(s).unsqueeze(-1)

    class ConvB(nn.Module):
        def __init__(self, i, o, k, p, s):
            super().__init__()
            self.c = nn.Conv1d(i, o, k, padding=p, stride=s)
            self.bn = _mk_norm(norm, o)
            self.act = nn.LeakyReLU(0.2)
            self.film = FiLM(o) if use_film else None

        def forward(self, x, st):
            x = self.act(self.bn(self.c(x)))
            return self.film(x, st) if self.film is not None else x

    class DeconvB(nn.Module):
        def __init__(self, i, o, k, p, op, s):
            super().__init__()
            self.c = nn.ConvTranspose1d(i, o, k, padding=p, output_padding=op, stride=s)
            self.bn = _mk_norm(norm, o)
            self.act = nn.LeakyReLU(0.2)
            self.film = FiLM(o) if use_film else None

        def forward(self, x, st):
            x = self.act(self.bn(self.c(x)))
            return self.film(x, st) if self.film is not None else x

    class Net(nn.Module):
        """1D U-Net，架构蓝本 yongbin98/uEDA-Net::denoising_model（加法跳连）。"""

        def __init__(self):
            super().__init__()
            self.e1 = ConvB(in_channels, 16, 65, 32, 2)
            self.e2 = ConvB(16, 32, 31, 15, 2)
            self.e3 = ConvB(32, 64, 15, 7, 2)
            self.e4 = ConvB(64, 128, 7, 3, 2)
            self.e5 = ConvB(128, 256, 7, 3, 2)
            self.d1 = DeconvB(256, 128, 7, 3, 1, 2)
            self.d2 = DeconvB(128, 64, 7, 3, 1, 2)
            self.d3 = DeconvB(64, 32, 15, 7, 1, 2)
            self.d4 = DeconvB(32, 16, 31, 15, 1, 2)
            self.d5 = nn.ConvTranspose1d(16, 1, 7, padding=3, output_padding=1, stride=2)

        def forward(self, x, st):
            L = x.shape[-1]
            pad = (-L) % UNET_MULT                     # 5 级 stride=2 要求 L 是 32 的倍数
            if pad:
                x = nn.functional.pad(x, (0, pad), mode="replicate")
            x1 = self.e1(x, st); x2 = self.e2(x1, st); x3 = self.e3(x2, st)
            x4 = self.e4(x3, st); x5 = self.e5(x4, st)
            y = self.d1(x5, st)
            y = self.d2(y + x4, st)
            y = self.d3(y + x3, st)
            y = self.d4(y + x2, st)
            y = self.d5(y + x1)
            return y[..., :L]

    return Net()


def step4_train(args):
    torch, nn = torch_import()
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    dtr = np.load(os.path.join(FEAT_DIR, "train.npz"))
    dte = np.load(os.path.join(FEAT_DIR, "test.npz"))
    norm = {}
    for sk in ("stats2", "stats3"):
        norm[sk] = (dtr[sk].mean(0, keepdims=True), dtr[sk].std(0, keepdims=True) + 1e-8)

    dev = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    log("[step4] 设备 %s | 训练 %d 段 / 测试 %d 段 | %d 个消融变体"
        % (dev, dtr["X"].shape[0], dte["X"].shape[0], len(VARIANTS)))

    log_rows, results = [], {}
    for name, cfg in VARIANTS.items():
        xk, ck_ = cfg["xkey"], cfg["cond"]
        cd = COND_DIM[ck_]
        X = torch.from_numpy(dtr[xk]).float()
        Xv = torch.from_numpy(dte[xk]).float()
        Y = torch.from_numpy(dtr["y"]).float().unsqueeze(1)
        Yv = torch.from_numpy(dte["y"]).float().unsqueeze(1)
        if cd:
            m, s = norm[ck_]
            S = torch.from_numpy(((dtr[ck_] - m) / s).astype(np.float32))
            Sv = torch.from_numpy(((dte[ck_] - m) / s).astype(np.float32))
        else:
            S = torch.zeros(len(X), 0); Sv = torch.zeros(len(Xv), 0)

        # ---- P2 修正：epoch 选择所用验证集 ----
        #   val_mode=test    ：沿用历史口径（每 epoch 用**测试集** PRD 选最优 → 绝对指标偏乐观）
        #   val_mode=subject ：从**训练集**留出一个受试者做验证（推荐；测试集只用于报告）
        use_subj_val = (args.val_mode == "subject")
        if use_subj_val and "subject" not in dtr.files:
            raise SystemExit("[step4] --val-mode=subject 但训练特征缺少 subject 字段；"
                             "请用带 subject 的特征（禁止退回用测试集选 epoch）")
        if args.val_mode == "test":
            print("[step4][warn] --val-mode=test 会用**测试集** PRD 选 epoch（历史泄漏口径，仅用于复现旧归档）")
        if use_subj_val:
            vm = (dtr["subject"] == args.val_subject)
            X = torch.from_numpy(dtr[xk][~vm]).float()
            Xv = torch.from_numpy(dtr[xk][vm]).float()
            Y = torch.from_numpy(dtr["y"][~vm]).float().unsqueeze(1)
            Yv = torch.from_numpy(dtr["y"][vm]).float().unsqueeze(1)
            if cd:
                m, s = norm[ck_]
                S = torch.from_numpy(((dtr[ck_][~vm] - m) / s).astype(np.float32))
                Sv = torch.from_numpy(((dtr[ck_][vm] - m) / s).astype(np.float32))
            else:
                S = torch.zeros(len(X), 0); Sv = torch.zeros(len(Xv), 0)

        model = make_model(torch, nn, X.shape[1], cd, getattr(args, "norm", "bn")).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        mse, l1 = nn.MSELoss(), nn.L1Loss()
        n_par = sum(p.numel() for p in model.parameters())
        best_prd, best_state = float("inf"), None

        for ep in range(1, args.epochs + 1):
            model.train()
            perm = torch.randperm(X.shape[0])
            tot = 0.0
            for s in range(0, X.shape[0], args.batch):
                idx = perm[s:s + args.batch]
                xb, sb, yb = X[idx].to(dev), S[idx].to(dev), Y[idx].to(dev)
                opt.zero_grad()
                out = model(xb, sb)
                loss = mse(out, yb) + 0.5 * l1(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += loss.item() * len(idx)
            sched.step()
            tot /= X.shape[0]
            model.eval()
            with torch.no_grad():
                pv = model(Xv.to(dev), Sv.to(dev)).cpu().numpy()[:, 0]
            yv = Yv.numpy()[:, 0]
            m_prd = float(np.nanmean([prd(yv[i], pv[i]) for i in range(len(pv))]))
            log_rows.append([name, ep, round(tot, 6), round(m_prd, 4)])
            if m_prd < best_prd:
                best_prd, best_state = m_prd, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ep == 1 or ep % max(1, args.epochs // 5) == 0 or ep == args.epochs:
                log("        %-8s epoch %2d/%d  loss %.4f  test PRD %.2f%%" % (name, ep, args.epochs, tot, m_prd))

        test_prd_report = best_prd
        if use_subj_val:
            model.load_state_dict(best_state); model.eval()
            with torch.no_grad():
                Xt = torch.from_numpy(dte[xk]).float().to(dev)
                St = (torch.from_numpy(((dte[ck_] - norm[ck_][0]) / norm[ck_][1]).astype(np.float32)).to(dev)
                      if cd else torch.zeros(len(dte[xk]), 0).to(dev))
                pt = model(Xt, St).cpu().numpy()[:, 0]
            test_prd_report = float(np.nanmean([prd(dte["y"][i], pt[i]) for i in range(len(pt))]))
            log("        %-8s [subject-val] 验证最优 PRD %.2f%%（epoch 选择用）| 测试集 PRD %.2f%%（仅报告）"
                % (name, best_prd, test_prd_report))
        torch.save({"model": best_state, "in_channels": int(X.shape[1]),
                    "cond_dim": cd, "xkey": xk, "cond": ck_,
                    "stats2_mean": torch.from_numpy(norm["stats2"][0].astype(np.float32)),
                    "stats2_std": torch.from_numpy(norm["stats2"][1].astype(np.float32)),
                    "stats3_mean": torch.from_numpy(norm["stats3"][0].astype(np.float32)),
                    "stats3_std": torch.from_numpy(norm["stats3"][1].astype(np.float32)),
                    "rls": dict(**_rls_kw(), alpha=_active_alpha()),
                    "norm": str(getattr(args, "norm", "bn")),
                    "feat_alpha": _active_alpha(),
                    "feat_taps": _rls_kw()["taps"], "feat_lam": _rls_kw()["lam"],
                    "seg_len": SEG_LEN, "fs": FS_CIME},
                   os.path.join(W_DIR, "%s%s.pt" % (name, args.tag)))
        results[name] = dict(params=int(n_par), best_prd=best_prd, test_prd_report=test_prd_report,
                             val_mode=args.val_mode, desc=cfg["desc"],
                             xkey=xk, cond=ck_, weight="weights/%s%s.pt" % (name, args.tag))
        log("        %-9s [%s] 参数量 %.2f M -> weights/%s%s.pt  (PRD %.2f%%)"
            % (name, cfg["desc"], n_par / 1e6, name, args.tag, best_prd))

    wcsv(os.path.join(OUT_DIR, "step4_train_log%s.csv" % args.tag), log_rows, ["variant", "epoch", "loss", "test_prd"])
    jdump(os.path.join(OUT_DIR, "step4_train_summary%s.json" % args.tag),
          dict(variants=results, epochs=args.epochs, batch=args.batch, lr=args.lr,
               seed=args.seed, device=str(dev), val_mode=args.val_mode,
               val_subject=(args.val_subject if args.val_mode == "subject" else None),
               elapsed_s=round(time.time() - t0, 1)))
    log("[step4] 完成 -> weights/*.pt, results/step4_train_log%s.csv, results/step4_train_summary%s.json"
        % (args.tag, args.tag))
    return results


# ----------------------------------------------------------------------------
# 5. CIME 波形恢复评估（消融）
# ----------------------------------------------------------------------------
def step5_eval(args):
    torch, nn = torch_import()
    from scipy import stats as sstats
    t0 = time.time()
    dte = np.load(os.path.join(FEAT_DIR, "test.npz"))
    summ = jload(os.path.join(OUT_DIR, "step4_train_summary%s.json" % args.tag))

    y = dte["y"]; ppg_raw = dte["X"][:, 0]; ppg_rls = dte["X"][:, 2]
    subj = subject_ids(dte, "test")
    n = len(y)
    subjects = np.unique(subj)

    def seg_metrics(pred):
        """逐段算，返回数组，便于做受试者层面的配对统计。"""
        p = np.array([prd(y[i], pred[i]) for i in range(n)])
        r = np.array([np.corrcoef(y[i], pred[i])[0, 1] if pred[i].std() > 0 else np.nan for i in range(n)])
        c = np.array([np.dot(y[i], pred[i]) / (np.linalg.norm(y[i]) * np.linalg.norm(pred[i]) + 1e-12) for i in range(n)])
        s = np.array([snr_db(pred[i], y[i]) for i in range(n)])
        return dict(prd=p, r=r, cos=c, snr=s)

    def subj_mean(x):
        return np.array([np.nanmean(x[subj == s]) for s in subjects])

    per_seg, per_subj = {}, {}
    rows = []

    def add_row(label, m, key=None):
        key = key or label
        per_seg[key] = m
        per_subj[key] = {k: subj_mean(v) for k, v in m.items()}
        rows.append([label, round(float(np.nanmean(m["prd"])), 3), round(float(np.nanmean(m["r"])), 4),
                     round(float(np.nanmean(m["cos"])), 4), round(float(np.nanmean(m["snr"])), 3)])

    add_row("Raw PPG", seg_metrics(ppg_raw), key="Raw PPG")
    add_row("RLS only (α=%.2f)" % _active_alpha(), seg_metrics(ppg_rls), key="RLS_only_out")

    # ---------- 带通基线（E1 传统滤波对照，零相位 Butterworth） ----------
    try:
        from scipy import signal as ssig
        sos = ssig.butter(BP_ORDER, BP_BAND, btype="bandpass", fs=FS_CIME, output="sos")
        bp = ssig.sosfiltfilt(sos, ppg_raw, axis=-1).astype(np.float32)
        add_row("带通 %.1f–%.1f Hz（%d 阶 Butterworth）" % (BP_BAND[0], BP_BAND[1], BP_ORDER),
                seg_metrics(bp), key="Bandpass")
    except Exception as e:  # 基线失败不应阻断主实验
        log("[step5] 带通基线跳过：%s" % e)

    preds = {}
    for name, info in summ["variants"].items():
        ck = torch.load(os.path.join(W_DIR, "%s%s.pt" % (name, args.tag)), map_location="cpu")
        model = make_model(torch, nn, ck["in_channels"], ck["cond_dim"], ck.get("norm", "bn"))
        model.load_state_dict(ck["model"]); model.eval()
        X = dte[ck["xkey"]]
        if ck["cond_dim"]:
            m = ck["%s_mean" % ck["cond"]].numpy()
            s = ck["%s_std" % ck["cond"]].numpy()
            S = ((dte[ck["cond"]] - m) / s).astype(np.float32)
        else:
            S = np.zeros((len(X), 0), dtype=np.float32)
        with torch.no_grad():
            pred = model(torch.from_numpy(X).float(), torch.from_numpy(S)).numpy()[:, 0]
        preds[name] = pred
        add_row("%s: %s" % (name, info["desc"]), seg_metrics(pred), key=name)

    # ---------- 受试者层面的配对显著性检验 ----------
    # 统计单位必须是「受试者」而不是「片段」：同一受试者的片段高度相关，
    # 按片段做配对检验会把自由度从 38 抬到 1900，显著性严重虚高。
    def paired(a_key, b_key, metric, better="lower"):
        a = per_subj[a_key][metric]; b = per_subj[b_key][metric]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if len(a) < 3:
            return None
        diff = a - b
        t, p_t = sstats.ttest_rel(a, b)
        try:
            _, p_w = sstats.wilcoxon(a, b)
        except Exception:
            p_w = float("nan")
        dz = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
        win = float(np.mean(diff < 0)) if better == "lower" else float(np.mean(diff > 0))
        return dict(metric=metric, n_subjects=int(len(a)),
                    mean_a=float(np.nanmean(a)), mean_b=float(np.nanmean(b)),
                    delta=float(np.nanmean(diff)), p_ttest=float(p_t), p_wilcoxon=float(p_w),
                    cohens_dz=dz, win_rate=win)

    comparisons = [
        ("ppg_only", "rls_only", "RLS 相对纯 PPG 的独立价值"),
        ("ppg_only", "no_rls", "原始 IMU 相对纯 PPG 的价值"),
        ("rls_only", "no_rls", "RLS 蒸馏 vs 原始 IMU 输入"),
        ("rls_only", "full", "再塞入原始 IMU 是否有害"),
        ("Raw PPG", "rls_only", "整体方案 vs 不做处理"),
    ]
    sig = []
    for a, b, note in comparisons:
        if a not in per_subj or b not in per_subj:
            continue
        rec = dict(comparison="%s vs %s" % (a, b), note=note)
        for metric, better in (("prd", "lower"), ("r", "higher")):
            res = paired(a, b, metric, better)
            if res:
                rec[metric] = res
        sig.append(rec)

    wcsv(os.path.join(OUT_DIR, "step5_ablation%s.csv" % args.tag), rows,
         ["method", "PRD_pct", "Pearson", "Cosine", "SNR_dB"])
    metrics = {"n_segments": n, "n_subjects": int(len(subjects)),
               "baseline_raw": {"PRD": rows[0][1], "Pearson": rows[0][2], "Cosine": rows[0][3], "SNR_dB": rows[0][4]},
               "methods": [{"method": r[0], "PRD": r[1], "Pearson": r[2], "Cosine": r[3], "SNR_dB": r[4]} for r in rows],
               "elapsed_s": round(time.time() - t0, 1)}
    jdump(os.path.join(OUT_DIR, "step5_metrics%s.json" % args.tag), metrics)
    jdump(os.path.join(OUT_DIR, "step5_significance%s.json" % args.tag),
          {"n_subjects": int(len(subjects)), "unit": "subject (paired)", "comparisons": sig})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
            try:
                matplotlib.rcParams["font.sans-serif"] = [fam]
                matplotlib.rcParams["axes.unicode_minus"] = False
                break
            except Exception:
                continue
        i = int(np.argmax([np.std(ppg_raw[k]) for k in range(min(200, n))]))
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        t = np.arange(SEG_LEN) / FS_CIME
        ax[0].plot(t, y[i], "k", lw=1.2, label="clean (参考)")
        ax[0].plot(t, ppg_raw[i], "r", lw=0.8, alpha=0.7, label="PPG_raw")
        ax[0].legend(fontsize=8); ax[0].set_ylabel("幅值"); ax[0].set_title("CIME 测试段 #%d：原始 vs 干净" % i)
        ax[1].plot(t, y[i], "k", lw=1.2, label="clean")
        for k_, col in (("full", "b"), ("rls_only", "g"), ("ppg_only", "m")):
            if k_ in preds:
                ax[1].plot(t, preds[k_][i], col, lw=0.9, alpha=0.8, label=k_)
        ax[1].legend(fontsize=8); ax[1].set_xlabel("时间 (s)"); ax[1].set_ylabel("幅值")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "step5_waveform_compare%s.png" % args.tag), dpi=130)
        plt.close(fig)
        log("[step5] 图 -> results/step5_waveform_compare%s.png" % args.tag)
    except Exception as e:                                   # 画图失败不影响主流程
        log("[step5] 画图跳过：%s" % e)

    log("[step5] 消融结果（%d 段 / %d 名受试者）：" % (n, len(subjects)))
    for r in rows:
        log("        %-46s PRD %7.2f%%  r %.4f  SNR %6.2f dB" % (r[0][:46], r[1], r[2], r[4]))
    log("[step5] 受试者层面配对检验（Wilcoxon / 配对 t 检验）：")
    for rec in sig:
        pr = rec.get("prd"); rr = rec.get("r")
        log("        %-28s | %s" % (rec["comparison"], rec["note"]))
        if pr:
            log("            PRD : %.2f vs %.2f  Δ=%+.2f  p_w=%.2e  dz=%+.2f  胜率=%.2f"
                % (pr["mean_a"], pr["mean_b"], pr["delta"], pr["p_wilcoxon"], pr["cohens_dz"], pr["win_rate"]))
        if rr:
            log("            r   : %.4f vs %.4f  Δ=%+.4f  p_w=%.2e  dz=%+.2f  胜率=%.2f"
                % (rr["mean_a"], rr["mean_b"], rr["delta"], rr["p_wilcoxon"], rr["cohens_dz"], rr["win_rate"]))
    log("[step5] 完成 -> results/step5_ablation%s.csv, results/step5_significance%s.json" % (args.tag, args.tag))
    return metrics


# ----------------------------------------------------------------------------
# 6. IEEEPPG（TROIKA）心率评估
# ----------------------------------------------------------------------------
def _psd(x, fs):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    return np.fft.rfftfreq(len(x), d=1.0 / fs), spec


def estimate_hr(seg, fs, band=HR_BAND, acc=None, notch=0.18, n_harm=3):
    """频域峰值法估心率（bpm），带抛物线插值。

    关键点：剧烈运动时跑步步频（约 1.3~2 Hz）及其谐波的功率往往**高于**真实心率峰，
    直接用 argmax 会被运动峰骗走（实测 TROIKA raw PPG MAE 高达 38 bpm）。
    这里用 IMU 谱峰作为运动先验，对 PPG 谱在步频及其谐波附近做陷波，再取峰——
    这正是"IMU 提供运动先验"的可解释用法。
    """
    freq, spec = _psd(seg, fs)
    if acc is not None:
        _, aspec = _psd(acc, fs)
        band_m = (freq >= 0.8) & (freq <= 4.0)          # 步频通常落在 0.8~4 Hz
        if band_m.any():
            f_motion = freq[band_m][int(np.argmax(aspec[band_m]))]
            if f_motion > 0.3:
                for h in range(1, n_harm + 1):
                    f0 = f_motion * h
                    if f0 > freq[-1]:
                        break
                    spec[np.abs(freq - f0) <= notch] = 0.0
    m = (freq >= band[0]) & (freq <= band[1])
    if not m.any() or spec[m].max() <= 0:
        return float("nan")
    idx = int(np.argmax(np.where(m, spec, -1)))
    if 0 < idx < len(spec) - 1:
        a, b, c = spec[idx - 1], spec[idx], spec[idx + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        f = freq[idx] + d * (freq[1] - freq[0])
    else:
        f = freq[idx]
    return float(f * 60.0)


def step6_hr(args):
    """在 IEEEPPG/TROIKA 上比较 raw / RLS / U-Net 的心率误差（跨数据集零样本）。"""
    import scipy.io as sio
    import scipy.signal as ssig
    torch, nn = torch_import()
    t0 = time.time()

    tr_dir = os.path.join(TROIKA_DIR, "Training_data")
    if not os.path.isdir(tr_dir):
        log("[step6] 跳过：未找到 %s" % tr_dir)
        return None

    _ck_name = "full%s.pt" % args.tag
    _ck_path = os.path.join(W_DIR, _ck_name)
    if not os.path.exists(_ck_path):
        _ck_path = os.path.join(W_DIR, "full.pt")
        log("[step6][warn] 未找到 full%s.pt，回退读取无 tag 的 full.pt（可能不是本次训练产物）"
            % args.tag)
    ck = torch.load(_ck_path, map_location="cpu")
    model = make_model(torch, nn, ck["in_channels"], ck["cond_dim"], ck.get("norm", "bn"))
    model.load_state_dict(ck["model"]); model.eval()
    sm, ss = ck["stats2_mean"].numpy(), ck["stats2_std"].numpy()

    files = sorted(f for f in os.listdir(tr_dir) if f.endswith(".mat") and "BPMtrace" not in f)
    if args.quick:
        files = files[:4]

    res = {"raw": [], "rls": [], "unet": []}
    for fn in files:
        base = fn[:-4]
        mt = os.path.join(tr_dir, base + "_BPMtrace.mat")
        if not os.path.exists(mt):
            continue
        d = sio.loadmat(os.path.join(tr_dir, fn))
        bpm0 = sio.loadmat(mt)["BPM0"].reshape(-1)
        sig = np.asarray(d["sig"], dtype=np.float64)
        # 官方 Readme 的行序（已用频带能量实证）：0=ECG, 1=PPG1, 2=PPG2, 3~5=ACC(x,y,z)
        # 注意：数据集的 row0 是 ECG，不是 PPG！
        ppg = sig[1]
        acc = sig[3:6]                                        # 3 轴加速度
        # 125 Hz -> 100 Hz（模型在 CIME 100 Hz 上训练）
        ppg100 = ssig.resample_poly(ppg, 4, 5)
        acc100 = ssig.resample_poly(acc, 4, 5, axis=1)
        N = min(len(ppg100), int(len(bpm0) * 2 * FS_CIME) + 8 * int(FS_CIME))
        ppg100, acc100 = ppg100[:N], acc100[:, :N]

        n_seg = N // SEG_LEN
        for k in range(n_seg):
            s = k * SEG_LEN
            seg = ppg100[s:s + SEG_LEN]
            a = acc100[:, s:s + SEG_LEN]
            # BPM0[i] 对应原始记录中 [2i, 2i+8) 秒的 8 秒窗（步进 2 s，重叠 6 s）。
            # 本段覆盖 [k*30, (k+1)*30) 秒，取完全落在段内的 8 秒窗做参考值。
            t0s, t1s = k * 30, (k + 1) * 30
            sel = [i for i in range(len(bpm0)) if 2 * i >= t0s and 2 * i + 8 <= t1s]
            if not sel:
                continue
            ref_bpm = float(np.median(bpm0[sel]))
            m_hat, _ = rls_decompose(seg[None, :], a[0][None, :], a.mean(0)[None, :], **_rls_kw())
            rls_out = seg - _active_alpha() * m_hat[0]
            # U-Net 输入
            scale = seg.std() + 1e-8
            chans = [seg / scale, m_hat[0] / scale, rls_out / scale,
                     a[0] / (acc100.std() + 1e-8), a.mean(0) / (acc100.std() + 1e-8)]
            X = np.stack(chans)[None].astype(np.float32)
            st = np.array([[np.log1p(np.sqrt(np.mean(a ** 2))),
                            float(np.atleast_1d(spectral_entropy(seg))[0])]], dtype=np.float32)
            stn = ((st - sm) / ss).astype(np.float32)
            with torch.no_grad():
                out = model(torch.from_numpy(X).float(), torch.from_numpy(stn)).numpy()[0, 0]
            res["raw"].append((ref_bpm, estimate_hr(seg, FS_CIME, acc=a.mean(0))))
            res["rls"].append((ref_bpm, estimate_hr(rls_out, FS_CIME, acc=a.mean(0))))
            res["unet"].append((ref_bpm, estimate_hr(out, FS_CIME, acc=a.mean(0))))

    def stats(pairs):
        if not pairs:
            return {"n": 0, "MAE": None, "RMSE": None, "pct_err_gt5": None}
        t = np.array([p[0] for p in pairs]); q = np.array([p[1] for p in pairs])
        ok = np.isfinite(t) & np.isfinite(q)
        t, q = t[ok], q[ok]
        if len(t) == 0:
            return {"n": 0, "MAE": None, "RMSE": None, "pct_err_gt5": None}
        err = q - t
        return {"n": int(len(t)), "MAE": round(float(np.mean(np.abs(err))), 3),
                "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 3),
                "pct_err_gt5": round(float(np.mean(np.abs(err) > 5)), 4)}

    out = {k: stats(v) for k, v in res.items()}
    out["fs_model"] = FS_CIME
    out["fs_source"] = FS_TROIKA
    out["seg_len_s"] = SEG_LEN / FS_CIME
    out["elapsed_s"] = round(time.time() - t0, 1)
    jdump(os.path.join(OUT_DIR, "step6_hr_metrics.json"), out)
    log("[step6] IEEEPPG 零样本心率评估（%d 段）" % out["raw"]["n"])
    for k in ("raw", "rls", "unet"):
        v = out[k]
        log("        %-6s MAE %s  RMSE %s  >5bpm %s" % (k, v["MAE"], v["RMSE"], v["pct_err_gt5"]))
    log("[step6] 完成 -> results/step6_hr_metrics.json")
    return out


# ----------------------------------------------------------------------------
# 7. 汇总报告
# ----------------------------------------------------------------------------
def step7_report(args):
    r1 = jload(os.path.join(OUT_DIR, "step1_data_report.json"))
    r2 = jload(os.path.join(OUT_DIR, "step2_best_rls.json"))
    r4 = jload(os.path.join(OUT_DIR, "step4_train_summary.json"))
    r5 = jload(os.path.join(OUT_DIR, "step5_metrics.json"))
    p6 = os.path.join(OUT_DIR, "step6_hr_metrics.json")
    r6 = jload(p6) if os.path.exists(p6) else None

    L = []
    L.append("# PPG 运动伪影抑制流水线 — 运行报告\n")
    L.append("> 自动生成，全部数字来自本地实跑。\n")
    L.append("## 1. 数据\n")
    for k, v in r1["splits"].items():
        L.append("- **CIME %s**：%d 段 × %d 通道 × %d 点（%.1f s @ %g Hz），通道 %s，有同步 clean 参考"
                 % (k, v["n_segments"], v["n_channels"], v["seg_len"], v["duration_s"], v["fs"], "+".join(v["channels"])))
    L.append("")
    L.append("## 2. RLS 标定（step2）\n")
    L.append("- 基线 PPG_raw：SNR %.2f dB，PRD %.2f%%" % (r2["base_snr"], r2["base_prd"]))
    L.append("- 最佳：λ=%.4f，taps=%d，**收缩系数 α=%.2f**，ΔSNR=%+.2f dB，PRD %.2f%%"
             % (r2["lam"], r2["taps"], r2["alpha"], r2["delta_snr"], r2["prd"]))
    L.append("- 结论：RLS 单独增益有限，且最优 α<1 说明存在**信号抵消**；RLS 定位为伪影估计与运动先验。\n")
    L.append("## 3. 训练（step4）\n")
    for k, v in r4["variants"].items():
        L.append("- `%s`：参数量 %.2f M，最优测试 PRD %.2f%%" % (k, v["params"] / 1e6, v["best_prd"]))
    L.append("- 设备 %s，%d epoch，耗时 %.0f s\n" % (r4["device"], r4["epochs"], r4["elapsed_s"]))
    L.append("## 4. CIME 波形恢复消融（step5）\n")
    L.append("| 方法 | PRD (%) ↓ | Pearson ↑ | Cosine ↑ | SNR (dB) ↑ |")
    L.append("|---|---|---|---|---|")
    for m in r5["methods"]:
        L.append("| %s | %.2f | %.4f | %.4f | %.2f |" % (m["method"], m["PRD"], m["Pearson"], m["Cosine"], m["SNR_dB"]))
    L.append("")
    if r6:
        L.append("## 5. IEEEPPG 零样本心率评估（step6）\n")
        L.append("| 输入 | 段数 | MAE (bpm) | RMSE (bpm) | 误差>5bpm 比例 |")
        L.append("|---|---|---|---|---|")
        for k in ("raw", "rls", "unet"):
            v = r6[k]
            L.append("| %s | %s | %s | %s | %s |" % (k, v["n"], v["MAE"], v["RMSE"], v["pct_err_gt5"]))
        L.append("")
        L.append("> 跨数据集零样本：模型只在 CIME（指夹、100 Hz）训练，直接用于 TROIKA（腕部、"
                 "剧烈运动、125 Hz 重采样）存在明显域差，此表反映真实迁移难度。\n")
    L.append("## 6. 图表产物\n")
    L.append("- `results/step5_waveform_compare.png`：典型片段的原始 / 干净 / 恢复波形对比")
    L.append("- `results/step5_ablation.csv`：消融实验原始数据")
    L.append("- `results/step2_rls_tuning.csv`：RLS 参数扫描原始数据")
    L.append("- `results/step4_train_log.csv`：逐 epoch 训练日志")
    L.append("")

    txt = "\n".join(L)
    with open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    log("[step7] 完成 -> results/REPORT.md")
    return txt


# ----------------------------------------------------------------------------
# 8. 多种子汇总 + 显著性判决
# ----------------------------------------------------------------------------
def _load_ablation(tag):
    p = os.path.join(OUT_DIR, "step5_ablation%s.csv" % tag)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))[1:]
    return {r[0]: dict(PRD=float(r[1]), Pearson=float(r[2]), Cosine=float(r[3]), SNR=float(r[4])) for r in rows}


def step8_verdict(args):
    """把多个随机种子的结果汇总成 mean±std，并给出「方案是否显著可行」的数据判决。"""
    seeds = args.seeds
    tables = {}
    for s in seeds:
        t = _load_ablation("_s%d" % s)
        if t:
            tables[s] = t
    if not tables:
        log("[step8] 没找到任何 step5_ablation_s*.csv，先跑多种子实验")
        return None

    methods = list(next(iter(tables.values())).keys())
    agg = {}
    for m in methods:
        agg[m] = {}
        for k in ("PRD", "Pearson", "Cosine", "SNR"):
            v = np.array([tables[s][m][k] for s in tables])
            agg[m][k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0)

    # 显著性：跨种子聚合每对比较的 p 值（取中位数）与「有多少个种子达到 p<0.05」
    sig_by_pair = {}
    for s in tables:
        p = os.path.join(OUT_DIR, "step5_significance_s%d.json" % s)
        if not os.path.exists(p):
            continue
        d = jload(p)
        for rec in d["comparisons"]:
            key = rec["comparison"]
            sig_by_pair.setdefault(key, dict(note=rec["note"], prd=[], r=[]))
            if rec.get("prd"):
                sig_by_pair[key]["prd"].append((rec["prd"]["p_wilcoxon"], rec["prd"]["delta"], rec["prd"]["win_rate"], rec["prd"]["cohens_dz"]))
            if rec.get("r"):
                sig_by_pair[key]["r"].append((rec["r"]["p_wilcoxon"], rec["r"]["delta"], rec["r"]["win_rate"], rec["r"]["cohens_dz"]))

    def summarize(lst):
        if not lst:
            return None
        a = np.array(lst, dtype=float)
        return dict(p_median=float(np.median(a[:, 0])), p_min=float(a[:, 0].min()),
                    n_sig=int((a[:, 0] < 0.05).sum()), n_total=len(lst),
                    delta_mean=float(a[:, 1].mean()), win_mean=float(a[:, 2].mean()),
                    dz_mean=float(a[:, 3].mean()))

    sig = {}
    for k, v in sig_by_pair.items():
        sig[k] = dict(note=v["note"], prd=summarize(v["prd"]), r=summarize(v["r"]))

    # ---------------- 判决规则 ----------------
    def get(m, k):
        return agg.get(m, {}).get(k, {}).get("mean", float("nan"))

    def get_std(m, k):
        return agg.get(m, {}).get(k, {}).get("std", float("nan"))

    checks = []

    # 符号约定：delta = mean_a − mean_b。
    # 下列比较中「更好的方法」是第二个参数 b，所以期望的符号随指标相反：
    #   Pearson 越高越好 -> delta 应 < 0；PRD 越低越好 -> delta 应 > 0
    rls_vs_ppg = sig.get("ppg_only vs rls_only")
    if rls_vs_ppg and rls_vs_ppg.get("r"):
        s_ = rls_vs_ppg["r"]
        ok = s_["p_median"] < 0.05 and s_["delta_mean"] < 0 and s_["n_sig"] >= max(1, s_["n_total"] - 1)
        checks.append(("C1 RLS 相对纯 PPG 有显著独立价值（Pearson↑）", ok,
                       "Δr=%+.4f, p中位=%.1e, %d/%d 个种子显著" % (s_["delta_mean"], s_["p_median"], s_["n_sig"], s_["n_total"])))
    rls_vs_ppg_p = sig.get("ppg_only vs rls_only")
    if rls_vs_ppg_p and rls_vs_ppg_p.get("prd"):
        s_ = rls_vs_ppg_p["prd"]
        ok = s_["p_median"] < 0.05 and s_["delta_mean"] > 0
        checks.append(("C2 RLS 相对纯 PPG 显著降低 PRD", ok,
                       "ΔPRD=%+.2f, p中位=%.1e, %d/%d 个种子显著" % (s_["delta_mean"], s_["p_median"], s_["n_sig"], s_["n_total"])))

    rls_vs_imu = sig.get("rls_only vs no_rls")
    if rls_vs_imu and rls_vs_imu.get("r"):
        s_ = rls_vs_imu["r"]
        ok = s_["p_median"] < 0.05 and s_["delta_mean"] > 0
        checks.append(("C3 RLS 蒸馏优于直接输入原始 IMU（Pearson↑）", ok,
                       "Δr=%+.4f, p中位=%.1e, %d/%d 个种子显著" % (s_["delta_mean"], s_["p_median"], s_["n_sig"], s_["n_total"])))

    raw_vs_best = sig.get("Raw PPG vs rls_only")
    if raw_vs_best and raw_vs_best.get("prd"):
        s_ = raw_vs_best["prd"]
        ok = s_["p_median"] < 0.05 and s_["delta_mean"] > 0
        checks.append(("C4 整体方案相对 Raw 显著改善 PRD", ok,
                       "ΔPRD=%+.2f, p中位=%.1e, %d/%d 个种子显著" % (s_["delta_mean"], s_["p_median"], s_["n_sig"], s_["n_total"])))

    stds = [get_std(m, "Pearson") for m in methods if np.isfinite(get_std(m, "Pearson"))]
    checks.append(("C5 多种子结果稳定（Pearson 种子间 std < 0.01）", max(stds) < 0.01 if stds else False,
                   "最大 std=%.4f（%d 个种子）" % (max(stds) if stds else float("nan"), len(tables))))

    passed = sum(1 for _, ok, _ in checks if ok)
    verdict = ("方案可行性得到数据支持" if passed >= 4 else
               "方案部分成立，需按未通过项调整" if passed >= 2 else
               "方案未获数据支持，需重新设计")

    # ---------------- 输出 ----------------
    L = ["# 数据判决：RLS + U-Net 方案是否显著可行\n",
         "> 种子数 %d：%s；统计单位 = 受试者（%d 名，配对检验）\n" % (len(tables), seeds, 
         jload(os.path.join(OUT_DIR, "step5_significance_s%d.json" % seeds[0]))["n_subjects"]
         if os.path.exists(os.path.join(OUT_DIR, "step5_significance_s%d.json" % seeds[0])) else -1)]
    L.append("## 1. 多种子汇总（mean ± std）\n")
    L.append("| 方法 | PRD (%) ↓ | Pearson ↑ | Cosine ↑ | SNR (dB) ↑ |")
    L.append("|---|---|---|---|---|")
    for m in methods:
        a = agg[m]
        L.append("| %s | %.2f ± %.2f | %.4f ± %.4f | %.4f ± %.4f | %.2f ± %.2f |"
                 % (m, a["PRD"]["mean"], a["PRD"]["std"], a["Pearson"]["mean"], a["Pearson"]["std"],
                    a["Cosine"]["mean"], a["Cosine"]["std"], a["SNR"]["mean"], a["SNR"]["std"]))
    L.append("")
    L.append("## 2. 受试者层面配对检验（跨种子汇总）\n")
    L.append("| 比较 | 说明 | 指标 | Δ | p(中位) | 显著种子数 | Cohen's dz | 胜率 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, v in sig.items():
        for mk, mn in (("prd", "PRD"), ("r", "Pearson")):
            s_ = v.get(mk)
            if s_:
                L.append("| %s | %s | %s | %+.4f | %.1e | %d/%d | %+.2f | %.2f |"
                         % (k, v["note"], mn, s_["delta_mean"], s_["p_median"],
                            s_["n_sig"], s_["n_total"], s_["dz_mean"], s_["win_mean"]))
    L.append("")
    L.append("## 3. 判决\n")
    for name, ok, detail in checks:
        L.append("- [%s] **%s** — %s" % ("通过" if ok else "未通过", name, detail))
    L.append("")
    L.append("### 结论：**%s**（%d/%d 项通过）\n" % (verdict, passed, len(checks)))

    txt = "\n".join(L)
    with open(os.path.join(OUT_DIR, "VERDICT.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    jdump(os.path.join(OUT_DIR, "verdict.json"),
          dict(seeds=seeds, aggregate=agg, significance=sig,
               checks=[{"name": n, "pass": bool(o), "detail": d} for n, o, d in checks],
               passed=passed, total=len(checks), verdict=verdict))

    log("[step8] 多种子汇总（%d 个种子）：" % len(tables))
    for m in methods:
        a = agg[m]
        log("        %-46s PRD %.2f±%.2f  r %.4f±%.4f  SNR %.2f±%.2f"
            % (m[:46], a["PRD"]["mean"], a["PRD"]["std"], a["Pearson"]["mean"], a["Pearson"]["std"],
               a["SNR"]["mean"], a["SNR"]["std"]))
    log("[step8] 判决：")
    for name, ok, detail in checks:
        log("        [%s] %s — %s" % ("通过" if ok else "未通过", name, detail))
    log("[step8] 结论：%s（%d/%d）" % (verdict, passed, len(checks)))
    log("[step8] -> results/VERDICT.md, results/verdict.json")
    return verdict


# ----------------------------------------------------------------------------
# 9. TROIKA 官方协议下的 HR 决策层（8 s 窗 / 2 s 步进，与 BPM0 逐点对齐）
# ----------------------------------------------------------------------------
def _psd_pad(x, fs, pad=4):
    """零填充 FFT 做 PSD，得到比 1/T 更细的峰值分辨率。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n), n=n * pad)) ** 2
    return np.fft.rfftfreq(n * pad, d=1.0 / fs), X


def _motion_notch(P, freq, acc, fs, notch=0.2, n_harm=3):
    """用 ACC 谱峰作为运动先验，对 PPG 谱在步频及其谐波处陷波（返回被压制的位置）。"""
    if acc is None:
        return P, []
    _, Pa = _psd_pad(acc, fs)
    m = (freq >= 0.8) & (freq <= 4.0)
    if not m.any():
        return P, []
    f_m = freq[m][int(np.argmax(Pa[m]))]
    if f_m <= 0.3:
        return P, []
    P = P.copy()
    killed = []
    for h in range(1, n_harm + 1):
        f0 = f_m * h
        if f0 > freq[-1]:
            break
        sel = np.abs(freq - f0) <= notch
        P[sel] = 0.0
        killed.append(float(f0))
    return P, killed


def psd_matrix(sig, acc, fs, n_win, win_s=8.0, hop_s=2.0, use_notch=True, notch_w=0.2):
    """把整段信号切成与 BPM0 对齐的窗，逐窗算 PSD，返回 (freq, P[n_win, F])。

    对齐关系：第 i 个窗覆盖 [2i, 2i+8) 秒，对应 BPM0[i]。
    """
    win, hop = int(round(win_s * fs)), int(round(hop_s * fs))
    rows, freq = [], None
    for i in range(n_win):
        s, e = i * hop, i * hop + win
        if e > len(sig):
            break
        f, P = _psd_pad(sig[s:e], fs)
        if use_notch:
            P, _ = _motion_notch(P, f, None if acc is None else acc[s:e], fs, notch=notch_w)
        freq = f
        rows.append(P)
    return freq, np.array(rows)


def _interp_peak(freq, P, j):
    """抛物线插值，把峰值定位精度从 1 个 bin 提升到亚 bin。"""
    if 0 < j < len(P) - 1:
        a, b, c = P[j - 1], P[j], P[j + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        return freq[j] + d * (freq[1] - freq[0])
    return freq[j]


def sel_argmax(freq, P, band=HR_BAND):
    """基准：带内直接取最大峰（会被跑步步频骗走）。"""
    m = (freq >= band[0]) & (freq <= band[1])
    idx = np.where(m)[0]
    out = np.full(len(P), np.nan)
    for t in range(len(P)):
        if P[t][idx].max() <= 0:
            continue
        j = int(idx[np.argmax(P[t][m])])
        out[t] = _interp_peak(freq, P[t], j) * 60.0
    return out


def sel_track(freq, P, band=HR_BAND, jump_bpm=18.0, w=1.2):
    """贪婪追踪：候选峰里挑「谱能量 − 与上一窗跳变惩罚」最大者。"""
    m = (freq >= band[0]) & (freq <= band[1])
    idx = np.where(m)[0]
    out = np.full(len(P), np.nan)
    prev = None
    for t in range(len(P)):
        row = P[t]
        cand = [j for j in idx if row[j] > 0 and (j == 0 or row[j] >= row[j - 1])
                and (j == len(row) - 1 or row[j] >= row[j + 1])]
        if not cand:
            cand = [int(idx[np.argmax(row[m])])]
        if prev is None:
            j = int(idx[np.argmax(row[m])])
        else:
            best, bs = None, -1e18
            for jj in cand:
                bpm = freq[jj] * 60.0
                sc = np.log(row[jj] + 1e-30) - w * ((bpm - prev) / jump_bpm) ** 2
                if sc > bs:
                    bs, best = sc, jj
            j = best
        v = _interp_peak(freq, row, j) * 60.0
        out[t] = v
        prev = v
    return out


def sel_viterbi(freq, P, band=HR_BAND, jump_bpm=VIT_JUMP, lam=VIT_LAM):
    """Viterbi/DP：全局最优路径 max Σ log P(f_t) − λΣ((f_t−f_{t−1})/jump)²。

    比贪婪追踪强在：单窗错峰不会把后续全部带偏（有回溯）。
    """
    m = (freq >= band[0]) & (freq <= band[1])
    idx = np.where(m)[0]
    fb = freq[idx] * 60.0
    logP = np.log(P[:, idx] + 1e-30)                       # (T, K)
    T, K = logP.shape
    if T == 0:
        return np.full(0, np.nan)
    trans = -lam * ((fb[:, None] - fb[None, :]) / jump_bpm) ** 2       # (K,K) prev->cur
    dp = logP[0].copy()
    bp = np.zeros((T, K), dtype=np.int32)
    for t in range(1, T):
        cand = dp[:, None] + trans
        bp[t] = np.argmax(cand, axis=0)
        dp = logP[t] + cand.max(axis=0)
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t][path[t]]
    return fb[path]


def sqi_series(freq, P, hr, band=HR_BAND):
    """可信度：真实谱峰能量占带内总能量的比例。"""
    m = (freq >= band[0]) & (freq <= band[1])
    out = np.full(len(P), np.nan)
    for t in range(len(P)):
        if not np.isfinite(hr[t]):
            continue
        j = int(np.argmin(np.abs(freq * 60.0 - hr[t])))
        out[t] = P[t][j] / (P[t][m].sum() + 1e-30)
    return out


def _reconstruct_unet(net, in_ch, dev, ppg, acc, gyro, fs, win_s=30.0, hop_s=2.0):
    """滑窗推理 + 汉宁窗重叠相加：把 30 s 窗的模型输出拼回连续信号。"""
    import torch
    win, hop = int(win_s * fs), int(round(hop_s * fs))
    n = len(ppg)
    out = np.zeros(n); wsum = np.zeros(n)
    hann = np.hanning(win)
    starts = list(range(0, max(1, n - win + 1), hop)) or [0]
    Xs = []
    for s0 in starts:
        seg = ppg[s0:s0 + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        a, g = acc[s0:s0 + win], gyro[s0:s0 + win]
        if len(a) < win:
            a = np.pad(a, (0, win - len(a)))
        if len(g) < win:
            g = np.pad(g, (0, win - len(g)))
        m_hat, _ = rls_decompose(seg[None, :], a[None, :], g[None, :], **_rls_kw())
        rls_out = seg - _active_alpha() * m_hat[0]
        sc = seg.std() + 1e-8
        ch = [seg / sc, m_hat[0] / sc, rls_out / sc]
        if in_ch == 5:
            ch += [a / (acc.std() + 1e-8), g / (gyro.std() + 1e-8)]
        Xs.append(np.stack(ch).astype(np.float32))
    X = np.stack(Xs)
    preds = []
    with torch.no_grad():
        for b in range(0, len(X), 64):
            xb = torch.from_numpy(X[b:b + 64]).to(dev)
            preds.append(net(xb, None).cpu().numpy()[:, 0])
    P = np.concatenate(preds, 0)
    for k, s0 in enumerate(starts):
        L = min(win, n - s0)
        out[s0:s0 + L] += P[k][:L] * hann[:L]
        wsum[s0:s0 + L] += hann[:L]
    return out / np.maximum(wsum, 1e-6)


def step9_hr2(args):
    """TROIKA 官方协议下的 HR：输入信号 × 选峰策略 两个维度的消融。"""
    import scipy.io as sio
    import scipy.signal as ssig
    from scipy.signal import butter, filtfilt, find_peaks
    torch, nn = torch_import()
    t0 = time.time()
    dev = torch.device(args.device)

    tr_dir = os.path.join(TROIKA_DIR, "Training_data")
    if not os.path.isdir(tr_dir):
        log("[step9] 跳过：未找到 %s" % tr_dir)
        return None

    ck = torch.load(os.path.join(W_DIR, "rls_only.pt"), map_location="cpu")
    net = make_model(torch, nn, ck["in_channels"], ck["cond_dim"], ck.get("norm", "bn")).to(dev).eval()
    net.load_state_dict(ck["model"])
    in_ch = int(ck["in_channels"])
    log("[step9] 用 rls_only 模型（%d 通道，设备 %s）做连续信号重构" % (in_ch, dev))

    files = sorted(f for f in os.listdir(tr_dir) if f.endswith(".mat") and "BPMtrace" not in f)
    if args.quick:
        files = files[:4]

    def hr_from_ecg(ecg, fs, n_win, win_s=8.0, hop_s=2.0):
        """协议自检：ECG R 峰测心率。若它也测不准，说明窗对齐有误。"""
        b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
        env = np.abs(filtfilt(b, a, ecg))
        pk, _ = find_peaks(env, height=env.mean() + 0.6 * env.std(), distance=int(0.35 * fs))
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

    inputs = ["raw", "rls", "unet"] if not args.no_unet else ["raw", "rls"]
    # 分离「陷波」与「Viterbi 追踪」各自的作用：viterbi 一行完全不陷波
    sels = ["argmax", "viterbi", "notch+argmax", "notch+greedy",
            "notch+viterbi", "notch0.1+viterbi", "notch+viterbi+sqi"]
    acc = {k: {s: [] for s in sels} for k in inputs}
    ecg_pairs = []

    for fn in files:
        base = fn[:-4]
        mt = os.path.join(tr_dir, base + "_BPMtrace.mat")
        if not os.path.exists(mt):
            continue
        sig = np.asarray(sio.loadmat(os.path.join(tr_dir, fn))["sig"], dtype=np.float64)
        bpm0 = sio.loadmat(mt)["BPM0"].reshape(-1)
        ppg, acc3 = sig[1], sig[3:6]        # 官方 Readme：row0=ECG, row1~2=PPG, row3~5=ACC
        ppg100 = ssig.resample_poly(ppg, 4, 5)
        acc100 = ssig.resample_poly(acc3, 4, 5, axis=1)
        acc_mag = np.sqrt((acc100 ** 2).sum(0))
        n_win = len(bpm0)

        hr_e = hr_from_ecg(ssig.resample_poly(sig[0], 4, 5), FS_CIME, n_win)
        for i in range(min(n_win, len(hr_e))):
            if np.isfinite(hr_e[i]) and np.isfinite(bpm0[i]):
                ecg_pairs.append((float(bpm0[i]), float(hr_e[i])))

        m_hat, _ = rls_decompose(ppg100[None, :], acc100[0][None, :], acc_mag[None, :], **_rls_kw())
        rls_sig = ppg100 - _active_alpha() * m_hat[0]
        sigs = {"raw": ppg100, "rls": rls_sig}
        try:
            sigs["unet"] = _reconstruct_unet(net, in_ch, dev, ppg100, acc100[0], acc_mag, FS_CIME)
        except Exception as e:
            # 不再用 raw 顶替（会把 unet 行伪装成"和 raw 一样好"）
            log("[step9][warn] %s U-Net 重构失败，跳过 unet 行：%s" % (base, e))

        for k in inputs:
            f0, P0 = psd_matrix(sigs[k], acc_mag, FS_CIME, n_win, use_notch=False)
            f1, P1 = psd_matrix(sigs[k], acc_mag, FS_CIME, n_win, use_notch=True)
            f2, P2 = psd_matrix(sigs[k], acc_mag, FS_CIME, n_win, use_notch=True, notch_w=0.08)
            res = {"argmax": sel_argmax(f0, P0),
                   "viterbi": sel_viterbi(f0, P0),
                   "notch+argmax": sel_argmax(f1, P1),
                   "notch+greedy": sel_track(f1, P1)}
            hr_v = sel_viterbi(f1, P1)
            res["notch+viterbi"] = hr_v
            res["notch0.1+viterbi"] = sel_viterbi(f2, P2)
            sq = sqi_series(f1, P1, hr_v)
            hr_s = hr_v.copy()
            thr = np.nanpercentile(sq, 20) if np.isfinite(sq).any() else 0.0
            for t in range(1, len(hr_s)):
                if np.isfinite(sq[t]) and sq[t] < thr and np.isfinite(hr_s[t - 1]):
                    hr_s[t] = hr_s[t - 1]
            res["notch+viterbi+sqi"] = hr_s
            for s in sels:
                hr = res[s]
                for i in range(min(n_win, len(hr))):
                    if np.isfinite(hr[i]) and np.isfinite(bpm0[i]):
                        acc[k][s].append((float(bpm0[i]), float(hr[i])))

    def stat(pairs):
        if not pairs:
            return dict(n=0, MAE=None, RMSE=None, pct5=None)
        t = np.array([p[0] for p in pairs]); q = np.array([p[1] for p in pairs])
        err = q - t
        return dict(n=int(len(t)), MAE=round(float(np.mean(np.abs(err))), 2),
                    RMSE=round(float(np.sqrt(np.mean(err ** 2))), 2),
                    pct5=round(float(np.mean(np.abs(err) > 5)), 3))

    table = {k: {s: stat(acc[k][s]) for s in sels} for k in inputs}
    rows = [[k, s, table[k][s]["n"], table[k][s]["MAE"], table[k][s]["RMSE"], table[k][s]["pct5"]]
            for k in inputs for s in sels]
    wcsv(os.path.join(OUT_DIR, "step9_hr_table.csv"), rows,
         ["input", "selector", "n_windows", "MAE_bpm", "RMSE_bpm", "pct_err_gt5"])
    e = stat(ecg_pairs)
    out = dict(protocol="TROIKA 官方：8 s 窗 / 2 s 步进，与 BPM0 逐点对齐",
               fs_model=FS_CIME, fs_source=FS_TROIKA, files=len(files),
               ecg_sanity=e, table=table, elapsed_s=round(time.time() - t0, 1))
    jdump(os.path.join(OUT_DIR, "step9_hr_metrics.json"), out)

    log("[step9] 协议自检（ECG R 峰 vs BPM0，%d 窗）：MAE %.2f  RMSE %.2f  >5bpm %.3f"
        % (e["n"], e["MAE"], e["RMSE"], e["pct5"]))
    log("        %s" % ("✅ 协议正确，PPG 误差来自伪影本身" if e["MAE"] <= 8 else "⚠ 协议可能有误"))
    log("[step9] HR 结果（%d 记录 / %d 窗）" % (len(files), table["raw"]["argmax"]["n"]))
    log("        %-6s %-22s %8s %8s %8s" % ("输入", "选峰策略", "MAE", "RMSE", ">5bpm"))
    for k in inputs:
        for s in sels:
            v = table[k][s]
            log("        %-6s %-22s %8.2f %8.2f %8.3f" % (k, s, v["MAE"], v["RMSE"], v["pct5"]))
    best = min(((k, s, table[k][s]["MAE"]) for k in inputs for s in sels), key=lambda x: x[2])
    log("[step9] 最佳：输入=%s 选峰=%s  MAE=%.2f bpm" % best)
    log("[step9] 完成 -> results/step9_hr_table.csv, results/step9_hr_metrics.json")
    return out

def step10_hr_test(args):
    """独立验证：在官方留出的 SP Cup 测试集（TestData + TrueBPM）上评估 HR。

    这批数据在本项目的任何调参中都没有参与，是真正的 held-out 验证。
    TestData 的 sig 为 5 行 = [PPG1, PPG2, ACCx, ACCy, ACCz]，不含 ECG。
    """
    import scipy.io as sio
    import scipy.signal as ssig
    t0 = time.time()
    te_dir = os.path.join(TROIKA_DIR, "TestData")
    tb_dir = os.path.join(TROIKA_DIR, "TrueBPM")
    if not os.path.isdir(te_dir):
        log("[step10] 跳过：未找到 %s" % te_dir)
        return None

    files = sorted(f for f in os.listdir(te_dir) if f.endswith(".mat"))
    if args.quick:
        files = files[:4]
    inputs = ["raw", "rls(1axis,a=.25)", "rls(3axis,a=1)"]
    sels = ["argmax", "viterbi", "notch+viterbi(tuned)"]
    acc = {k: {s: [] for s in sels} for k in inputs}

    for fn in files:
        base = fn[:-4]
        mt = os.path.join(tb_dir, base.replace("TEST_", "True_") + ".mat")
        if not os.path.exists(mt):
            log("[step10] 缺少真值：%s" % mt)
            continue
        sig = np.asarray(sio.loadmat(os.path.join(te_dir, fn))["sig"], dtype=np.float64)
        bpm0 = sio.loadmat(mt)["BPM0"].reshape(-1)
        ppg = sig[0]                      # 测试集：行 0/1 = PPG，行 2~4 = ACC
        acc3 = sig[2:5]
        ppg100 = ssig.resample_poly(ppg, 4, 5)
        acc100 = ssig.resample_poly(acc3, 4, 5, axis=1)
        amag = np.sqrt((acc100 ** 2).sum(0))
        n_win = len(bpm0)

        # 旧配置：只有 1 轴 ACC + 幅度（等效 ~1.5 通道）
        m1, _ = rls_decompose(ppg100[None, :], acc100[0][None, :], amag[None, :], **_rls_kw())
        # 新配置：纯 3 轴 ACC + α=1.0（受控实验确定）
        m3, _ = rls_decompose(ppg100[None, :], acc100.T[None, :],
                              np.zeros_like(acc100[0])[None, :])
        sigs = {"raw": ppg100,
                "rls(1axis,a=.25)": ppg100 - RLS_ALPHA_2CH * m1[0],
                "rls(3axis,a=1)": ppg100 - RLS_ALPHA_3AX * m3[0]}
        for k in inputs:
            f0, P0 = psd_matrix(sigs[k], amag, FS_CIME, n_win, use_notch=False)
            f1, P1 = psd_matrix(sigs[k], amag, FS_CIME, n_win, use_notch=True, notch_w=NOTCH_W)
            res = {"argmax": sel_argmax(f0, P0),
                   "viterbi": sel_viterbi(f0, P0),
                   "notch+viterbi(tuned)": sel_viterbi(f1, P1)}
            for s in sels:
                hr = res[s]
                for i in range(min(n_win, len(hr))):
                    if np.isfinite(hr[i]) and np.isfinite(bpm0[i]):
                        acc[k][s].append((float(bpm0[i]), float(hr[i])))

    def stat(pairs):
        if not pairs:
            return dict(n=0, MAE=None, RMSE=None, pct5=None)
        t = np.array([p[0] for p in pairs]); q = np.array([p[1] for p in pairs])
        err = q - t
        return dict(n=int(len(t)), MAE=round(float(np.mean(np.abs(err))), 2),
                    RMSE=round(float(np.sqrt(np.mean(err ** 2))), 2),
                    pct5=round(float(np.mean(np.abs(err) > 5)), 3))

    table = {k: {s: stat(acc[k][s]) for s in sels} for k in inputs}
    rows = [[k, s, table[k][s]["n"], table[k][s]["MAE"], table[k][s]["RMSE"], table[k][s]["pct5"]]
            for k in inputs for s in sels]
    wcsv(os.path.join(OUT_DIR, "step10_hr_test.csv"), rows,
         ["input", "selector", "n_windows", "MAE_bpm", "RMSE_bpm", "pct_err_gt5"])
    out = dict(protocol="SP Cup 官方测试集（held-out）：8 s 窗 / 2 s 步进",
               vit_lam=VIT_LAM, vit_jump=VIT_JUMP, notch_w=NOTCH_W,
               files=len(files), table=table, elapsed_s=round(time.time() - t0, 1))
    jdump(os.path.join(OUT_DIR, "step10_hr_test_metrics.json"), out)

    log("[step10] 官方留出测试集独立验证（%d 记录 / %d 窗）" % (len(files), table["raw"]["argmax"]["n"]))
    log("        %-6s %-22s %8s %8s %8s" % ("输入", "选峰策略", "MAE", "RMSE", ">5bpm"))
    for k in inputs:
        for s in sels:
            v = table[k][s]
            log("        %-6s %-22s %8.2f %8.2f %8.3f" % (k, s, v["MAE"], v["RMSE"], v["pct5"]))
    log("[step10] 完成 -> results/step10_hr_test.csv, results/step10_hr_test_metrics.json")
    return out


# ----------------------------------------------------------------------------
# 11. 运动强度分层实验（E3）
# ----------------------------------------------------------------------------
def step11_strata(args):
    """按 IMU 运动强度把测试段分 4 档（静息/轻/中/强），分层报告各方法指标。

    动机：整体均值会掩盖「强运动段更难、静息段几乎不变」的真实结构；
    方案里的 FiLM 运动自适应也需要这张表来支撑。

    运动强度定义：`stats2[:,0] = log(1+RMS(IMU))`，先用**训练集**统计量标准化，
    再按训练集分位在测试集上分档（避免用测试集信息定分位）。
    """
    torch, nn = torch_import()
    t0 = time.time()
    dtr = np.load(os.path.join(FEAT_DIR, "train.npz"))
    dte = np.load(os.path.join(FEAT_DIR, "test.npz"))
    summ = jload(os.path.join(OUT_DIR, "step4_train_summary%s.json" % args.tag))

    y = dte["y"]
    ppg_raw = dte["X"][:, 0]
    ppg_rls = dte["X"][:, 2]

    # 运动强度：训练集标准化 → 训练集分位分档 → 套到测试集
    m = dtr["stats2"].mean(0, keepdims=True)
    s = dtr["stats2"].std(0, keepdims=True) + 1e-8
    mi_tr = ((dtr["stats2"][:, :1] - m[:, :1]) / s[:, :1]).ravel()
    mi_te = ((dte["stats2"][:, :1] - m[:, :1]) / s[:, :1]).ravel()
    qs = np.quantile(mi_tr, [0.25, 0.50, 0.75])
    bins = np.digitize(mi_te, qs)                       # 0..3
    names = ["静息（最低25%）", "轻度（25–50%）", "中度（50–75%）", "强运动（最高25%）"]

    def seg_metrics(pred):
        p = np.array([prd(y[i], pred[i]) for i in range(len(y))])
        r = np.array([np.corrcoef(y[i], pred[i])[0, 1] if pred[i].std() > 0 else np.nan
                      for i in range(len(y))])
        sn = np.array([snr_db(pred[i], y[i]) for i in range(len(y))])
        return p, r, sn

    # 各方法的逐段指标（fig_labels 用纯 ASCII，避免中文在无 CJK 字体的机器上变豆腐块）
    methods, fig_labels = {}, {}
    fig_labels["Raw PPG"] = "Raw PPG"
    methods["Raw PPG"] = seg_metrics(ppg_raw)
    k_rls = "RLS only (α=%.2f)" % _active_alpha()
    fig_labels[k_rls] = "RLS only"
    methods[k_rls] = seg_metrics(ppg_rls)
    try:
        from scipy import signal as ssig
        sos = ssig.butter(BP_ORDER, BP_BAND, btype="bandpass", fs=FS_CIME, output="sos")
        k_bp = "带通 %.1f–%.1f Hz" % BP_BAND
        fig_labels[k_bp] = "Bandpass"
        methods[k_bp] = seg_metrics(ssig.sosfiltfilt(sos, ppg_raw, axis=-1).astype(np.float32))
    except Exception as e:
        log("[step11] 带通基线跳过：%s" % e)
    for name, info in summ["variants"].items():
        ck = torch.load(os.path.join(W_DIR, "%s%s.pt" % (name, args.tag)), map_location="cpu")
        model = make_model(torch, nn, ck["in_channels"], ck["cond_dim"], ck.get("norm", "bn"))
        model.load_state_dict(ck["model"]); model.eval()
        X = dte[ck["xkey"]]
        if ck["cond_dim"]:
            cm, cs = ck["%s_mean" % ck["cond"]].numpy(), ck["%s_std" % ck["cond"]].numpy()
            S = ((dte[ck["cond"]] - cm) / cs).astype(np.float32)
        else:
            S = np.zeros((len(X), 0), dtype=np.float32)
        with torch.no_grad():
            pred = model(torch.from_numpy(X).float(), torch.from_numpy(S)).numpy()[:, 0]
        methods[name] = seg_metrics(pred)
        fig_labels[name] = name

    # 逐档汇总
    rows, table = [], {}
    raw_snr_by_bin = {b: methods["Raw PPG"][2][bins == b].mean() for b in range(4)}
    for label, (p, r, sn) in methods.items():
        table[label] = {}
        for b in range(4):
            k = bins == b
            rec = dict(n=int(k.sum()),
                       PRD=round(float(np.nanmean(p[k])), 3),
                       Pearson=round(float(np.nanmean(r[k])), 4),
                       SNR_dB=round(float(np.nanmean(sn[k])), 3),
                       dSNR_vs_raw=round(float(np.nanmean(sn[k]) - raw_snr_by_bin[b]), 3))
            table[label][names[b]] = rec
            rows.append([label, names[b], rec["n"], rec["PRD"], rec["Pearson"],
                         rec["SNR_dB"], rec["dSNR_vs_raw"]])

    wcsv(os.path.join(OUT_DIR, "step11_motion_strata%s.csv" % args.tag), rows,
         ["method", "stratum", "n_segments", "PRD_pct", "Pearson", "SNR_dB", "dSNR_vs_raw_dB"])

    # 静息不退化检查（E4 的口径，在有 clean 参考的数据上直接可算）
    degrade = {}
    for label, (p, r, sn) in methods.items():
        if label == "Raw PPG":
            continue
        base_prd = float(np.nanmean(p[bins == 0]))
        raw_prd = float(np.nanmean(methods["Raw PPG"][0][bins == 0]))
        degrade[label] = round(base_prd - raw_prd, 3)   # 负值=比 Raw 更好
    out = dict(protocol="训练集分位分档（IMU 运动强度），测试集套用",
               quantile_boundaries=[round(float(q), 4) for q in qs],
               stratum_counts={names[b]: int((bins == b).sum()) for b in range(4)},
               resting_prd_minus_raw=degrade, table=table,
               elapsed_s=round(time.time() - t0, 1))
    jdump(os.path.join(OUT_DIR, "step11_motion_strata%s.json" % args.tag), out)

    # 图：四档下 PRD / SNR 改善（标签用 ASCII，避免缺 CJK 字体）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["axes.unicode_minus"] = False
        keys = [k for k in methods if k != "Raw PPG"]
        x = np.arange(4)
        xt = ["rest", "light", "moderate", "intense"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for k in keys:
            ax[0].plot(x, [table[k][names[b]]["PRD"] for b in range(4)], "o-", label=fig_labels.get(k, k))
            ax[1].plot(x, [table[k][names[b]]["dSNR_vs_raw"] for b in range(4)], "o-", label=fig_labels.get(k, k))
        ax[0].axhline(table["Raw PPG"][names[0]]["PRD"], ls="--", c="k", lw=1, label="Raw (rest)")
        ax[0].set_xticks(x); ax[0].set_xticklabels(xt)
        ax[0].set_ylabel("PRD (%) lower better"); ax[0].set_title("PRD by motion stratum")
        ax[1].axhline(0, ls="--", c="k", lw=1)
        ax[1].set_xticks(x); ax[1].set_xticklabels(xt)
        ax[1].set_ylabel("dSNR vs Raw (dB) higher better"); ax[1].set_title("SNR gain by motion stratum")
        for a in ax:
            a.grid(alpha=0.3); a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "step11_motion_strata%s.png" % args.tag), dpi=130)
        log("[step11] 图 -> results/step11_motion_strata%s.png" % args.tag)
    except Exception as e:
        log("[step11] 画图跳过：%s" % e)

    log("[step11] 运动强度分层（%d 段，分位边界 %s）" % (len(y), [round(float(q), 3) for q in qs]))
    for b in range(4):
        log("        %-16s n=%4d   Raw PRD %6.2f%%   rls_only PRD %6.2f%%   ΔSNR %+5.2f dB"
            % (names[b], table["Raw PPG"][names[b]]["n"], table["Raw PPG"][names[b]]["PRD"],
               table.get("rls_only", {}).get(names[b], {}).get("PRD", float("nan")),
               table.get("rls_only", {}).get(names[b], {}).get("dSNR_vs_raw", float("nan"))))
    log("[step11] 完成 -> results/step11_motion_strata%s.csv/.json/.png" % args.tag)
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
STEPS = ["step1_prepare", "step2_tune_rls", "step3_features",
         "step4_train", "step5_eval", "step6_hr", "step7_report",
         "step8_verdict", "step9_hr2", "step10_hr_test", "step11_strata"]


def parse_args():
    ap = argparse.ArgumentParser(description="PPG 运动伪影抑制流水线（RLS + 1D U-Net）")
    ap.add_argument("step", nargs="?", default="all", choices=STEPS + ["all"])
    ap.add_argument("--quick", action="store_true", help="小样本快速跑通（验证用）")
    ap.add_argument("--force", action="store_true", help="step1 强制重新转换 .mat")
    ap.add_argument("--tune-n", type=int, default=400, help="step2 用多少段做扫描（等间隔抽样）")
    ap.add_argument("--tune-on", choices=["train", "test"], default="train",
                    help="step2 在哪个划分上标定超参（默认 train；test 仅为复现旧的泄漏口径）")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.99, 0.995, 0.999, 1.0])
    ap.add_argument("--taps-list", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.4, 0.5, 0.75, 1.0])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 训练")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（多种子重复用）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="step8 汇总哪些种子")
    ap.add_argument("--tag", default="", help="输出文件名后缀，用于多种子实验互不覆盖")
    ap.add_argument("--val-mode", choices=["test", "subject"], default="subject",
                    help="epoch 选择的验证集：test=历史口径（用测试集，偏乐观）；subject=从训练集留一个受试者（推荐）")
    ap.add_argument("--val-subject", type=int, default=9, help="val-mode=subject 时留出的受试者编号")
    ap.add_argument("--norm", choices=["bn", "in", "gn"], default="bn",
                    help="主干归一化层：bn=BatchNorm（原实现/1 通道路径）；in=InstanceNorm（3 通道推荐，域内 PRD 更优）")
    ap.add_argument("--no-unet", action="store_true", help="step9 跳过 U-Net 重构（调参用）")
    ap.add_argument("--device", default=("cuda" if __import__("torch").cuda.is_available() else "cpu"),
                    help="step9 滑窗推理设备")
    return ap.parse_args()


def main():
    args = parse_args()
    todo = STEPS if args.step == "all" else [args.step]
    if args.quick:
        args.tune_n = min(args.tune_n, 60)
        args.epochs = min(args.epochs, 5)
    t0 = time.time()
    for s in todo:
        log("\n" + "=" * 74)
        log(">>> %s" % s)
        log("=" * 74)
        globals()[s](args)
        if s == "step2_tune_rls":
            global RLS_ALPHA_ACTIVE, RLS_LAMBDA_ACTIVE, _RLS_TUNED
            _RLS_TUNED = _tuned_rls()
            RLS_ALPHA_ACTIVE, RLS_LAMBDA_ACTIVE = _RLS_TUNED[2], _RLS_TUNED[1]
            log("[step2] 已刷新 α=%.2f λ=%.4f（后续 step 立即生效）"
                % (RLS_ALPHA_ACTIVE, RLS_LAMBDA_ACTIVE))
    log("\n全部完成，总耗时 %.1f s" % (time.time() - t0))
    if "step7_report" in todo:
        log("报告：%s" % os.path.join(OUT_DIR, "REPORT.md"))
    else:
        log("产物目录：%s" % OUT_DIR)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""伪影强度鲁棒性曲线：在 CIME 测试集上人工注入**可控强度**的运动伪影，
测量经典方法与本方案最终模型在各强度档下的 PRD / Pearson / SNR 退化曲线。

为什么需要它
------------
答辩里"鲁棒性"是本方案的核心卖点之一。前面所有结论（47.15 集成、46.57 融合）
都只在 CIME 测试集这一个**固有伪影强度分布**上成立；本工具把"伪影强度"变成
**自变量**，回答"当运动伪影比现在还强 2×/4×/8×/16× 时，各方法还剩多少"。

口径（与主实验严格对齐，逐条核对过）
------------------------------------
1. 干净参考：`outputs/rls_unet/data/cime_test.npz::clean`（1900×3000，与特征同一批）；
2. **注入在未标准化的 clean 上做**，再按主实验口径逐段标准化；
3. 特征与主实验逐位对齐（已数值核对，见 reports/36 §方法）：
   - `scale = noisy.std(1) + 1e-8`（float32）
   - `M̂ = rls_decompose(noisy, acc_ref, gyro_ref, taps=4, lam=0.995)` 的**第一返回值**
   - `PPG_RLS = noisy − 0.40·M̂`，`α=0.40` / `λ=0.995` 来自 `outputs/final_10seed/results/step2_best_rls.json`
     （⚠️ pipeline.py 里的兜底常量是 0.25/0.99，模块级 `_best_alpha()` 只在本工具的运行目录下
      找不到 step2 结果时会退回 0.25 —— 所以这里**显式传参**，不用默认值）
   - `XrlsOnly = [noisy/scale, M̂/scale, PPG_RLS/scale]`，`y = clean/scale`
   - `stats3 = [log1p(rms(acc)+rms(gyro)), 谱熵(ppg), mean|M̂|/mean|ppg|]`（FiLM 条件用）
   核对结果：真实测试集上 `max|X[:,2]−(X[:,0]−0.40·X[:,1])| = 4.8e-7`、`y == clean/scale` 逐位相等。

伪影模型（可控强度）
--------------------
每个受试段 i 生成**同一份**运动基（跨档位固定，保证"配对比较"）：

    m_i   = 单位方差步频运动（1.2–3.5 Hz + 谐波 + 幅度调制 + 频率漂移，复用
            `scripts/modules/domain_rand.synth_reference`），线性混合 + 轻微非线性
    n_i   = 单位方差白噪声（器件噪声）
    w_i   = 单位方差 <0.6 Hz 漂移（随机游走低通）

    artifact_i(κ) = std(clean_i) · [ κ·m_i + 1{本底}·(0.05·n_i + 0.15·w_i) ]
    noisy_i(κ)    = clean_i + artifact_i(κ)

κ 即"步频运动伪影/脉搏幅度比"（幅度比，不是 SNR）。档位：

| tag | κ | 说明 |
|---|---|---|
| `0`  | 0 | **无注入对照**（noisy = clean，参考通道全 0） |
| `f`  | 0 | 常数本底对照：只有 5% 白噪声 + 15% 漂移，无步频运动（参考通道 0） |
| `0.5`…`16` | 0.5/1/2/4/8/16 | 步频运动 + 同一份常数本底 |

参考通道（ACC_x / Gyro_y）与注入的运动**同源**：把合成运动基按 κ × 该段真实
CIME ACC/Gyro 的 RMS 缩放，因此
(a) `stats2[0]=log1p(rms)` 落在真实分布内（κ=1 时与真实段同量级），
(b) RLS/本方案原则上能完全抵消注入伪影 —— 这样测的是"对强度的鲁棒性"，
    而不是"参考通道失真带来的鲁棒性"（后者是另一个实验）。

方法（全部在同一批注入信号上）
------------------------------
`raw` / `bandpass_0.5_4` / `rls`(=ppg−0.40·M̂) / `specsub`（均复用
`scripts/tools/classical_baselines.py` 的既有实现）+ 本方案最终模型
`0.85×(41 成员残差集成) + 0.15×(3 成员 CRM 集成)`（复现 `outputs/b5_fuse` 的 46.57）。
另外附带 `residual_ens` / `crm_ens` 两个单分支行，用于归因。

真实集自检
----------
`tag=real` 用 `features/test.npz` 的 `XrlsOnly/y/stats3`（全部 1900 段）跑同一套前向+融合，
应当复现 **PRD 46.57**（`outputs/b5_fuse/fuse_branches.json` 的 `fuse_crm512`，α=0.15）。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    OMP_NUM_THREADS=2 $PY scripts/tools/robustness_sweep.py --out outputs/robustness
    # 快速冒烟（少量段 + 少量模型）
    OMP_NUM_THREADS=2 $PY scripts/tools/robustness_sweep.py --out outputs/tmp_smoke \\
        --n-seg 32 --models-limit 3 --skip-real
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "modules"))
from _common import paths as P                                # noqa: E402
from _common import parallel as PAR                           # noqa: E402

FS = 100.0
ALPHA = 0.40            # 与 features/test.npz 逐位核对过的收缩系数
TAPS = 4
LAM = 0.995
FLOOR_NOISE = 0.05      # 常数本底：白噪声（相对 clean std）
FLOOR_WANDER = 0.15     # 常数本底：<0.6 Hz 漂移
MOTION_LEVELS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
LEVEL_TAGS = ["0", "f"] + ["%g" % k for k in MOTION_LEVELS]
CLASSICAL = ["raw", "bandpass_0.5_4", "rls_a0.4", "specsub"]
# 报告里用的中文方法名
METHOD_LABEL = {"raw": "raw（恒等）", "bandpass_0.5_4": "bandpass 0.5–4 Hz",
                "rls_a0.4": "rls（ppg−0.40·M̂）", "specsub": "specsub（谱减）",
                "ours": "本方案 0.85·残差+0.15·CRM", "residual_ens": "残差集成（41 成员）",
                "crm_ens": "CRM 集成（3 成员）"}
W_RES, W_CRM = 0.85, 0.15


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prd(clean, pred):
    c = np.asarray(clean, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    return float(100 * np.linalg.norm(p - c) / (np.linalg.norm(c) + 1e-12))


def snr_db(signal, clean):
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    c = np.asarray(clean, dtype=np.float64).reshape(-1)
    pn = np.mean((s - c) ** 2)
    return float(10 * np.log10(np.mean(c ** 2) / pn)) if pn > 0 else float("inf")


def metrics(pred, Y):
    """pred/Y: (M, L) → PRD/Pearson/SNR 的段均值。

    与 ensemble_cime 的逐段 `pipeline.prd` / `snr_db` / `np.corrcoef` 逐段平均等价
    （只是向量化；数值差异 <1e-9，见 reports/36 §自检）。
    """
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    m = len(y)
    num = np.linalg.norm(p - y, axis=1)
    den = np.linalg.norm(y, axis=1) + 1e-12
    prds = 100.0 * num / den
    pn = ((p - y) ** 2).mean(1)
    pc = (y ** 2).mean(1)
    with np.errstate(divide="ignore"):
        snrs = 10.0 * np.log10(pc / pn)
    snrs = np.where(pn > 0, snrs, np.inf)
    yc = y - y.mean(1, keepdims=True)
    pcs = p - p.mean(1, keepdims=True)
    denr = np.linalg.norm(yc, axis=1) * np.linalg.norm(pcs, axis=1)
    pears = (yc * pcs).sum(1) / np.maximum(denr, 1e-12)
    ok = (denr > 1e-9) & np.isfinite(pears)
    return dict(n=int(m), PRD=round(float(np.nanmean(prds)), 3),
                SNR=round(float(np.nanmean(snrs)), 3),
                Pearson=round(float(np.nanmean(pears[ok])), 4) if ok.any() else float("nan"))


def sample_indices(subject, n_seg, seed):
    """按受试者分层抽样（每个受试者按比例取，尽量覆盖全部 38 人）。"""
    rng = np.random.default_rng(seed)
    subj = np.asarray(subject)
    out = []
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        k = max(1, int(round(n_seg * len(idx) / len(subj))))
        k = min(k, len(idx))
        out.extend(rng.choice(idx, size=k, replace=False).tolist())
    out = np.array(sorted(out), dtype=np.int64)
    if len(out) > n_seg:                      # 取够为止（剩下的随机丢）
        keep = rng.choice(len(out), size=n_seg, replace=False)
        out = np.sort(out[keep])
    return out


# --------------------------------------------------------------------------
# 阶段 1：注入伪影 → 造特征（+ 经典方法输出）
# --------------------------------------------------------------------------
def _feat_worker(payload):
    """chunk: dict(level, idx, seed, clean_npz, real_npz, feat_dir, out_dir)。

    ⚠️ PAR.pmap 会把 worker 返回的 dict 拍平成键 → 必须返回 list。
    """
    import importlib.util as iu
    P_ = load_module("ppg_pipeline", os.path.join(SCRIPTS, "pipeline", "pipeline.py"))
    DR = load_module("domain_rand", os.path.join(SCRIPTS, "modules", "domain_rand.py"))
    CB = load_module("classical_baselines", os.path.join(SCRIPTS, "tools", "classical_baselines.py"))

    tag = payload["level"]
    idx = np.asarray(payload["idx"], dtype=np.int64)
    seed = int(payload["seed"])
    spec = level_spec(tag)
    clean_all = np.load(payload["clean_npz"])["clean"]
    real_all = np.load(payload["imu_npz"])["noisy"]            # (1900,3,N) 真实 [PPG,ACC,Gyro]
    m = len(idx)
    N = clean_all.shape[1]

    X = np.zeros((m, 3, N), dtype=np.float32)
    Y = np.zeros((m, N), dtype=np.float32)
    ST = np.zeros((m, 3), dtype=np.float32)
    SC = np.zeros(m, dtype=np.float32)
    RT = np.zeros(m, dtype=np.float32)                        # 实测 伪影/脉搏 幅度比
    preds = {k: np.zeros((m, N), dtype=np.float32) for k in CLASSICAL}

    for j, g in enumerate(idx):
        rng = np.random.default_rng([seed, int(g)])
        ref = DR.synth_reference(1, N, FS, rng)               # (1,2,N) 单位随机的合成运动
        r0, r1 = ref[0, 0], ref[0, 1]
        mom = r0 + r1 + 0.1 * r0 ** 2
        mom = mom - mom.mean()
        mom = mom / (mom.std() + 1e-12)
        noi = rng.standard_normal(N)
        noi = (noi - noi.mean()) / (noi.std() + 1e-12)
        wan = DR._lowpass(np.cumsum(rng.standard_normal(N)), FS, 0.6)
        wan = (wan - wan.mean()) / (wan.std() + 1e-12)

        clean = clean_all[g].astype(np.float64)
        sc = float(clean.std())
        if spec["motion"]:
            # 参考通道按 κ × 该段真实 CIME ACC/Gyro RMS 缩放（保证 stats3 落在真实量级）
            acc_rms = float(np.sqrt(np.mean(real_all[g, 1].astype(np.float64) ** 2)))
            gyr_rms = float(np.sqrt(np.mean(real_all[g, 2].astype(np.float64) ** 2)))
            k = spec["kappa"]
            aref = r0 / (r0.std() + 1e-12) * (k * acc_rms)
            gref = r1 / (r1.std() + 1e-12) * (k * gyr_rms)
            art = sc * (k * mom)
        else:
            aref = np.zeros(N)
            gref = np.zeros(N)
            art = np.zeros(N)
        if spec["floor"]:
            art = art + sc * (FLOOR_NOISE * noi + FLOOR_WANDER * wan)

        noisy = (clean + art).astype(np.float32)
        mhat, _ = P_.rls_decompose(noisy[None, :].astype(np.float64), aref[None, :], gref[None, :],
                                   taps=TAPS, lam=LAM)     # rls_decompose 是批接口，逐段独立
        mhat = mhat[0]
        ppg_rls = noisy.astype(np.float64) - ALPHA * mhat

        scale = np.float32(noisy.std() + 1e-8)
        X[j, 0] = (noisy / scale).astype(np.float32)
        X[j, 1] = (mhat / scale).astype(np.float32)
        X[j, 2] = (ppg_rls / scale).astype(np.float32)
        Y[j] = (clean / scale).astype(np.float32)
        SC[j] = scale
        RT[j] = float(art.std() / (sc + 1e-12))
        mi = np.log1p(np.sqrt(np.mean(aref ** 2)) + np.sqrt(np.mean(gref ** 2)))
        se = float(P_.spectral_entropy(noisy[None, :])[0])
        ar = float(np.abs(mhat).mean() / (np.abs(noisy).mean() + 1e-8))
        ST[j] = (mi, se, ar)

        for k_ in CLASSICAL:                                  # 经典方法（标准化域，与主实验一致）
            try:
                out = CB.METHODS[k_](X[j, 0], X[j, 1], None, None)
            except Exception:
                continue
            out = np.asarray(out, dtype=np.float64)
            if len(out) != N or not np.all(np.isfinite(out)):
                continue
            preds[k_][j] = out.astype(np.float32)

    os.makedirs(payload["out_dir"], exist_ok=True)
    # 每个 chunk 写自己的文件（多进程写同一文件会写坏 zip → BadZipFile），主进程再按序合并
    fpath = os.path.join(payload["out_dir"], "_part_%s_c%03d.npz" % (tag, payload["chunk"]))
    np.savez(fpath, X=X, y=Y, stats3=ST, scale=SC, idx=idx,
             ratio_target=np.full(m, spec["kappa"], dtype=np.float32),
             ratio_art=RT, **{"pred_" + k_: v for k_, v in preds.items()})
    return [dict(level=tag, feat=fpath, n=m, chunk=payload["chunk"])]


FEAT_KEYS = ["X", "y", "stats3", "scale", "idx", "ratio_target", "ratio_art"] \
            + ["pred_" + k for k in CLASSICAL]


def merge_feat(parts, out_path):
    """把各 chunk 的分片按序拼成一个 feat_<tag>.npz（并删掉分片）。"""
    ds = [np.load(p) for p in parts]
    merged = {k: np.concatenate([d[k] for d in ds], axis=0) for k in FEAT_KEYS if k in ds[0]}
    np.savez(out_path, **merged)
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    return out_path


def level_spec(tag):
    if tag == "real":                                         # 真实测试集（无注入）
        return dict(kappa=None, floor=False, motion=False, real=True)
    if tag == "0":
        return dict(kappa=0.0, floor=False, motion=False)
    if tag == "f":
        return dict(kappa=0.0, floor=True, motion=False)
    return dict(kappa=float(tag), floor=True, motion=True)


# --------------------------------------------------------------------------
# 阶段 2：神经模型前向（分组求和后落盘，避免大数组过 IPC）
# --------------------------------------------------------------------------
def _load_xy(feat_path, real_dir):
    """返回 (X (M,3,N) float32, Y (M,N) float32, stats3 (M,3) float32)。"""
    if os.path.isdir(feat_path):                              # 真实集目录（features/test.npz）
        d = np.load(os.path.join(feat_path, "test.npz"))
        return d["XrlsOnly"], d["y"], d["stats3"]
    d = np.load(feat_path)
    return d["X"], d["y"], d["stats3"]


def _pred_worker(chunk):
    """chunk: list[dict]，每个 dict 描述一组模型在一个档位上的前向。返回 list。"""
    return [r for task in chunk for r in _pred_one(task)]


def _pred_one(payload):
    """payload: dict(kind, level, models, feat, out_dir, cond_mu, cond_sd, batch)。返回 list。"""
    import torch
    R = load_module("residual", os.path.join(SCRIPTS, "modules", "residual.py"))
    CRM = load_module("crm", os.path.join(SCRIPTS, "modules", "crm.py"))
    dev = torch.device("cpu")
    X, Y, ST = _load_xy(payload["feat"], payload.get("real_dir"))
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float()
    mu = np.asarray(payload["cond_mu"], dtype=np.float64)
    sd = np.asarray(payload["cond_sd"], dtype=np.float64)
    cond_cache = {}

    acc = np.zeros((X.shape[0], X.shape[-1]), dtype=np.float64)
    per_model = []
    for p in payload["models"]:
        if payload["kind"] == "res":
            model, meta = R.load(p, device=dev)
        else:
            model, meta = CRM.load(p, device=dev)
        cd = int(meta.get("cond_dim", 0) or 0)
        if cd > 0:
            if cd not in cond_cache:
                cond_cache[cd] = torch.from_numpy(
                    ((ST[:, :cd].astype(np.float64) - mu[:cd]) / (sd[:cd] + 1e-6)).astype(np.float32))
            S = cond_cache[cd]
        else:
            S = None
        out = np.zeros((len(X), X.shape[-1]), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(X), payload["batch"]):
                xb = Xt[s:s + payload["batch"]]
                st = (S[s:s + len(xb)] if S is not None
                      else torch.zeros(len(xb), 0))
                out[s:s + len(xb)] = model(xb, st)[:, 0].numpy()
        acc += out
        met = metrics(out, Y)
        per_model.append(dict(kind=payload["kind"], name=os.path.basename(p),
                              cond_dim=cd, **met))
    mean = (acc / max(len(payload["models"]), 1)).astype(np.float32)
    os.makedirs(os.path.join(payload["out_dir"], "sums"), exist_ok=True)
    spath = os.path.join(payload["out_dir"], "sums", "%s_%s_g%d.npy"
                         % (payload["level"], payload["kind"], payload["group"]))
    np.save(spath, mean)
    return [dict(level=payload["level"], kind=payload["kind"], group=payload["group"],
                 sum=spath, n_models=len(payload["models"]), per_model=per_model)]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="伪影强度鲁棒性曲线")
    ap.add_argument("--feat-dir", default=P.artifact("final_10seed/features"))
    ap.add_argument("--clean-npz", default=P.artifact("rls_unet/data/cime_test.npz"))
    ap.add_argument("--res-dir", default=P.artifact("ens_all"))
    ap.add_argument("--res-pattern", default="residual_rls_*.pt")
    ap.add_argument("--crm-dir", default=P.artifact("b5f512/weights"))
    ap.add_argument("--crm-pattern", default="crm_rls_*.pt")
    ap.add_argument("--levels", nargs="+", default=LEVEL_TAGS)
    ap.add_argument("--n-seg", type=int, default=500, help="每档段数（400–600）")
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=6, help="每个 worker 一次前向的成员数")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--models-limit", type=int, default=0, help=">0 时只取前 N 个模型（冒烟）")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", default=P.artifact("robustness"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    PAR.log_env()
    t_all = time.perf_counter()

    res_paths = sorted(glob.glob(os.path.join(args.res_dir, args.res_pattern)))
    crm_paths = sorted(glob.glob(os.path.join(args.crm_dir, args.crm_pattern)))
    if args.models_limit:
        res_paths = res_paths[:args.models_limit]
        crm_paths = crm_paths[:max(1, args.models_limit // 4)]
    print("[rob] 残差成员 %d | CRM 成员 %d" % (len(res_paths), len(crm_paths)))

    dte = np.load(os.path.join(args.feat_dir, "test.npz"))
    dtr = np.load(os.path.join(args.feat_dir, "train.npz"))
    mu = dtr["stats3"].astype(np.float64).mean(0)
    sd = dtr["stats3"].astype(np.float64).std(0) + 1e-6      # 与 fuse_branches 同口径（训练集统计）
    idx = sample_indices(dte["subject"], args.n_seg, args.seed)
    np.save(os.path.join(args.out, "sampled_idx.npy"), idx)
    print("[rob] 抽样 %d 段（受试者 %d 人，seed=%d）" % (len(idx), len(np.unique(dte["subject"][idx])), args.seed))

    # ---------- 阶段 1：注入 + 造特征 ----------
    chunk_idx = PAR.split_chunks(idx.tolist(), max(1, min(args.workers, len(idx))))
    feat_info = {}
    for tag in args.levels:
        payloads = [dict(level=tag, idx=c, seed=args.seed, clean_npz=args.clean_npz,
                         imu_npz=args.clean_npz, feat_dir=args.feat_dir, out_dir=args.out,
                         chunk=i)
                    for i, c in enumerate(chunk_idx)]
        got, dt = PAR.pmap(_feat_worker, payloads, workers=min(args.workers, len(payloads)),
                           threads=args.threads, desc="注入+RLS 特征 κ=%s" % tag)
        got = sorted(got, key=lambda r: r["chunk"])
        path = merge_feat([r["feat"] for r in got], os.path.join(args.out, "feat_%s.npz" % tag))
        feat_info[tag] = path
        print("      κ=%-5s → %s（%d 段，%.1f s）" % (tag, os.path.basename(path),
                                                  sum(r["n"] for r in got), dt))

    # ---------- 阶段 2：神经前向 ----------
    tasks = []
    for tag in args.levels:
        for kind, paths in (("res", res_paths), ("crm", crm_paths)):
            for gi, grp in enumerate(PAR.split_chunks(paths, max(1, -(-len(paths) // args.group_size)))):
                tasks.append(dict(kind=kind, level=tag, group=gi, models=grp,
                                  feat=feat_info[tag], out_dir=args.out, real_dir=args.feat_dir,
                                  cond_mu=mu.tolist(), cond_sd=sd.tolist(), batch=args.batch))
    if not args.skip_real:
        for kind, paths in (("res", res_paths), ("crm", crm_paths)):
            for gi, grp in enumerate(PAR.split_chunks(paths, max(1, -(-len(paths) // args.group_size)))):
                tasks.append(dict(kind=kind, level="real", group=gi, models=grp,
                                  feat=args.feat_dir, out_dir=args.out, real_dir=args.feat_dir,
                                  cond_mu=mu.tolist(), cond_sd=sd.tolist(), batch=args.batch))
    print("[rob] 前向任务 %d 个（%d 档 × %d 模型组）" % (len(tasks), len(args.levels) + (0 if args.skip_real else 1),
                                                    len(res_paths) // args.group_size + 2))
    got, dt = PAR.pmap(_pred_worker, PAR.split_chunks(tasks, args.workers * 3),
                       workers=args.workers, threads=args.threads, use_torch=True,
                       desc="神经前向 %d 任务" % len(tasks))
    print("[rob] 前向 %.1f s" % dt)

    tags = list(args.levels) + ([] if args.skip_real else ["real"])
    groups = {}
    per_model = {}
    for r in got:
        groups.setdefault((r["level"], r["kind"]), []).append(r["sum"])
        per_model.setdefault((r["level"], r["kind"]), []).extend(r["per_model"])

    # ---------- 阶段 3：融合 + 指标 ----------
    table = {}
    for tag in tags:
        if tag == "real":
            Y = dte["y"]
        else:
            Y = np.load(feat_info[tag])["y"]
        rec = {}
        # 经典方法
        d = None if tag == "real" else np.load(feat_info[tag])
        if tag != "real":
            for k in CLASSICAL:
                key = "pred_" + k
                if key in d:
                    rec[k] = metrics(d[key], Y)
            rec["_ratio_art"] = dict(median=round(float(np.median(d["ratio_art"])), 3),
                                     p10=round(float(np.percentile(d["ratio_art"], 10)), 3),
                                     p90=round(float(np.percentile(d["ratio_art"], 90)), 3))
            rec["_kappa"] = float(d["ratio_target"][0])
        else:
            rec["_kappa"] = None
            rec["_ratio_art"] = None
            # 真实集也能算 raw/bandpass/rls/specsub：直接取 XrlsOnly + 真实 ACC/Gyro
            Xr = dte["XrlsOnly"]
            CB = load_module("classical_baselines", os.path.join(SCRIPTS, "tools", "classical_baselines.py"))
            for k in CLASSICAL:
                out = np.zeros_like(Y)
                for i in range(len(Y)):
                    out[i] = CB.METHODS[k](Xr[i, 0], Xr[i, 1], None, None)
                rec[k] = metrics(out, Y)
            rec["_ratio_art"] = dict(median=float("nan"), p10=float("nan"), p90=float("nan"))

        enc = {}
        for kind, w, name in (("res", 1.0, "residual_ens"), ("crm", 1.0, "crm_ens")):
            ps = groups.get((tag, kind))
            if not ps:
                continue
            mean = np.mean([np.load(p).astype(np.float64) for p in ps], axis=0).astype(np.float32)
            enc[kind] = mean
            rec[name] = metrics(mean, Y)
        if "res" in enc and "crm" in enc:
            rec["ours"] = metrics((W_RES * enc["res"] + W_CRM * enc["crm"]).astype(np.float32), Y)
        table[tag] = rec
        pm = {k: per_model.get((tag, k), []) for k in ("res", "crm")}
        table[tag]["_per_model"] = pm
        line = " ".join("%s=%.2f" % (k, rec[k]["PRD"]) for k in
                        CLASSICAL + ["residual_ens", "crm_ens", "ours"] if k in rec)
        print("[rob] %-5s %s" % (tag, line))

    # ---------- 落盘 ----------
    methods = CLASSICAL + ["residual_ens", "crm_ens", "ours"]
    order = [t for t in (["0", "f"] + ["%g" % k for k in MOTION_LEVELS]) if t in table]
    if "real" in table:
        order = order + ["real"]
    rows = []
    for tag in order:
        for k in methods:
            if k not in table[tag]:
                continue
            rows.append(dict(level=tag, kappa=table[tag]["_kappa"], method=k,
                             label=METHOD_LABEL.get(k, k), **table[tag][k]))
    wide = {k: {tag: table[tag][k]["PRD"] for tag in order if k in table[tag]} for k in methods}

    # 派生量：log-log 斜率（PRD ∝ κ^p）、相对 raw 的伪影泄漏比、真实集等效 κ
    ktags = [t for t in order if t not in ("0", "f", "real")]
    derived = {}
    for k in methods:
        if k not in wide or len(ktags) < 2:
            continue
        xs = np.array([float(t) for t in ktags])
        ys = np.array([wide[k][t] for t in ktags])
        p_all = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
        m = xs >= 1.0
        p_hi = float(np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)[0])
        # ⚠️ 键名里有小数点 → 只能用字符串键，不能写关键字参数
        derived[k] = {"loglog_slope_0.5_16": round(p_all, 3), "loglog_slope_1_16": round(p_hi, 3),
                      "leak_ratio": {t: round(wide[k][t] / (100.0 * float(t)), 4) for t in ktags}}
    derived["_real_equivalent_kappa"] = round(wide["raw"]["real"] / 100.0, 3)
    if "specsub" in wide and "ours" in wide:                 # 本方案与谱减的交叉点（线性插值，log2 κ）
        cr = None
        for i in range(1, len(ktags)):
            a, b = ktags[i - 1], ktags[i]
            d0 = wide["ours"][a] - wide["specsub"][a]
            d1 = wide["ours"][b] - wide["specsub"][b]
            if d0 <= 0 < d1:
                w = -d0 / (d1 - d0)
                cr = float(2 ** (np.log2(float(a)) + w * (np.log2(float(b)) - np.log2(float(a)))))
                break
        derived["_crossover_kappa_ours_vs_specsub"] = None if cr is None else round(cr, 3)
    pm_out = {t: {k: table[t]["_per_model"][k] for k in ("res", "crm")} for t in order}
    out = dict(
        protocol=("CIME 测试集人工注入可控强度步频伪影（κ=伪影/脉搏幅度比）；注入在未标准化 clean 上做，"
                  "再按主实验口径逐段标准化（α=0.40/λ=0.995/taps=4，与 features/test.npz 逐位核对）；"
                  "每档 %d 段（受试者分层抽样，seed=%d）；本方案=0.85·41 成员残差集成+0.15·3 成员 CRM 集成；"
                  "real 档=真实 CIME 测试集全部 1900 段（自检，应≈46.57）"
                  % (len(idx), args.seed)),
        seed=args.seed, n_seg_per_level=int(len(idx)), levels=order,
        level_defs={t: level_spec(t) for t in order},
        floor_constants=dict(noise=FLOOR_NOISE, wander=FLOOR_WANDER),
        motion_levels=list(MOTION_LEVELS),
        res_members=len(res_paths), crm_members=len(crm_paths),
        res_dir=args.res_dir, crm_dir=args.crm_dir,
        fused_weights=dict(residual=W_RES, crm=W_CRM),
        method_labels=METHOD_LABEL,
        table=rows, PRD_wide=wide,
        derived=derived,
        per_model=pm_out,
        per_level_detail={t: {k: table[t][k] for k in methods if k in table[t]} for t in order},
        ratio_art={t: table[t]["_ratio_art"] for t in order},
        sanity=dict(raw_is_100kappa=True,
                    note="raw 档的 PRD 应等于 100κ（幅度比定义），实测见 PRD_wide['raw']，是注入器的自检"),
    )
    with open(os.path.join(args.out, "robustness.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "robustness.csv"), "w", encoding="utf-8") as f:
        f.write("level,kappa,method,label,n,PRD,Pearson,SNR_dB\n")
        for r in rows:
            f.write("%s,%s,%s,%s,%d,%.3f,%.4f,%.3f\n"
                    % (r["level"], "" if r["kappa"] is None else r["kappa"], r["method"],
                       r["label"], r["n"], r["PRD"], r["Pearson"], r["SNR"]))
    with open(os.path.join(args.out, "PRD_wide.csv"), "w", encoding="utf-8") as f:
        f.write("method," + ",".join(order) + "\n")
        for k in methods:
            f.write(k + "," + ",".join("%.3f" % wide[k][t] if t in wide[k] else ""
                                       for t in order) + "\n")

    print("\n[rob] PRD 表（行=方法，列=档位）")
    hdr = "%-28s" % "方法" + "".join("%9s" % t for t in order)
    print(hdr)
    for k in methods:
        if k not in wide:
            continue
        print("%-28s" % METHOD_LABEL.get(k, k)
              + "".join(("%9.2f" % wide[k][t]) if t in wide[k] else "%9s" % "-" for t in order))
    chk = wide.get("ours", {}).get("real")
    if chk is not None:
        print("\n[rob] 自检：真实测试集 ours PRD = %.2f（目标 46.57，偏差 %.2f）" % (chk, abs(chk - 46.57)))
    print("[rob] 完成（%.1f s）-> %s" % (time.perf_counter() - t_all, args.out))
    return out


if __name__ == "__main__":
    main()

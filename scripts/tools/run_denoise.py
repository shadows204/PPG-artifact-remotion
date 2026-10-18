#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPG 运动伪影去除 · 命令行入口（软件同学可直接调用，也可作为 GUI 的后端）。

功能：读入一段 PPG（可带 IMU）→ 输出去伪影结果（三条波形）+ 心率 + 图。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python

    # ① 一键演示（内置真实 CIME 测试段，自带参考波形 → 可直观看到效果）
    $PY scripts/tools/run_denoise.py --demo --out outputs/denoise_demo

    # ② 读自己的数据（npz：键 ppg[, acc, gyro[, reference]]，可选 fs 标量）
    $PY scripts/tools/run_denoise.py --input my.npz --fs 100 --out outputs/denoise_out

    # ③ 读 CSV（列名：ppg / acc_x,acc_y,acc_z / gyro_x,gyro_y,gyro_z，可选 reference）
    $PY scripts/tools/run_denoise.py --input my.csv --fs 25 --out outputs/denoise_out

输出（<out>/ 下）
-----------------
- `result.npz`  ：t / ppg_raw / artifact / ppg_prior / ppg_clean / hr_bpm
- `result.json` ：元信息 + 质量指标 + 心率中位
- `waveforms.png`：三波形图（① 原波形 ② 伪影波形 ③ 滤净波形 [+ 参考]）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "modules"))
from _common import paths as P                                # noqa: E402
from ppg_api import PPGDenoiser                               # noqa: E402


def _load_npz(path):
    d = np.load(path)
    keys = {k.lower(): k for k in d.files}
    if "ppg" not in keys:
        raise SystemExit("npz 缺少 ppg 键（实际键：%s）；请勿把特征文件当波形输入" % list(d.files)[:8])
    ppg = d[keys["ppg"]]
    acc = d[keys["acc"]] if "acc" in keys else None
    gyro = d[keys["gyro"]] if "gyro" in keys else None
    ref = d[keys["reference"]] if "reference" in keys else (d[keys["clean"]] if "clean" in keys else None)
    fs = float(d[keys["fs"]]) if "fs" in keys and np.ndim(d[keys["fs"]]) == 0 else None
    return ppg, acc, gyro, ref, fs


def _load_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = list(csv.DictReader(f))
    if not rd:
        raise ValueError("CSV 为空")
    cols = {c.lower(): c for c in rd[0].keys()}
    get = lambda name: (np.array([float(r[cols[name]]) for r in rd]) if name in cols else None)
    ppg = get("ppg")
    if ppg is None:
        raise ValueError("CSV 缺少 ppg 列")
    accs = [get("acc_%s" % a) for a in "xyz"]
    acc = np.stack([a for a in accs if a is not None]) if any(a is not None for a in accs) else None
    gs = [get("gyro_%s" % a) for a in "xyz"]
    gyro = np.stack([g for g in gs if g is not None]) if any(g is not None for g in gs) else None
    ref = get("reference")
    return ppg, acc, gyro, ref, None


def _demo():
    """内置演示：优先取真实 CIME 测试段；**仓库未带数据时**用合成样例（可开箱自测）。

    返回 (ppg, acc, gyro, reference, fs)。合成样例：HR 70 bpm 干净脉搏 + 3 Hz 运动伪影 + 噪声，
    reference 是干净脉搏（仅供 demo 计算 PRD）。
    """
    root = getattr(P, "ROOT", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in (("metadata", "cime_test.npz"), ("rls_unet", "data", "cime_test.npz")):
        path = os.path.join(root, "outputs", *rel)
        if os.path.exists(path):
            d = np.load(path)
            noisy, clean = d["noisy"][0], d["clean"][0]
            print("[demo] 使用真实 CIME 段：%s" % path)
            return noisy[0], noisy[1:2], noisy[2:3], clean, 100.0
    # ---- 合成样例（无数据时的开箱演示）----
    fs, n = 100.0, 3000
    t = np.arange(n) / fs
    hr = 70.0 / 60.0
    clean = (80 * np.sin(2 * np.pi * hr * t) + 25 * np.sin(2 * np.pi * 2 * hr * t + 0.8)
             + 12 * np.sin(2 * np.pi * 3 * hr * t + 1.7))
    motion = 40 * np.sin(2 * np.pi * 3.0 * t) + 18 * np.sin(2 * np.pi * 1.5 * t + 0.4)
    rng = np.random.RandomState(0)
    ppg = clean + motion + 3 * rng.randn(n)
    acc = (motion / 40.0 * 2.0)[None, :]                     # 与伪影同源的"加速度"
    gyro = (np.roll(motion, 20) / 40.0 * 0.8)[None, :]      # 异质参考（滞后 0.2 s）
    print("[demo] 未找到真实 CIME 段 → 使用**合成样例**（HR 70 bpm + 3 Hz 运动伪影）")
    return ppg, acc, gyro, clean, fs


def main():
    ap = argparse.ArgumentParser(description="PPG 运动伪影去除 · CLI")
    ap.add_argument("--input", default=None, help="npz 或 csv（见文件头说明）")
    ap.add_argument("--demo", action="store_true", help="内置真实 CIME 段演示")
    ap.add_argument("--fs", type=float, default=None, help="采样率 Hz（npz 含 fs 时可省）")
    ap.add_argument("--out", default=P.artifact("denoise_out"))
    ap.add_argument("--preset", default="lite", choices=["lite", "full"])
    ap.add_argument("--crm-weight", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-hr", action="store_true", help="不估计心率（更快）")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--eval-test", type=int, default=0,
                    help="自检：在 CIME 测试集前 N 段上跑一遍，报告 原波形/滤净 的平均 PRD（0=不做）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.demo:
        _is_synthetic = not os.path.exists(os.path.join(
            getattr(P, "ROOT", "."), "outputs", "rls_unet", "data", "cime_test.npz")) and \
            not os.path.exists(os.path.join(getattr(P, "ROOT", "."), "outputs",
                                            "metadata", "cime_test.npz"))
        ppg, acc, gyro, ref, fs = _demo()
        fs = args.fs or fs
    elif args.input:
        if args.input.endswith(".npz"):
            ppg, acc, gyro, ref, fs0 = _load_npz(args.input)
            fs = args.fs or fs0
        else:
            ppg, acc, gyro, ref, fs0 = _load_csv(args.input)
            fs = args.fs or fs0
        if fs is None:
            raise SystemExit("请用 --fs 指定采样率（文件中没有）")
    else:
        raise SystemExit("请给 --input 或 --demo")

    d = PPGDenoiser(preset=args.preset, device=args.device, alpha=args.alpha,
                    crm_weight=args.crm_weight, verbose=True).load()
    print("[run] 输入 %d 点 @ %g Hz → %s 模式" %
          (len(np.asarray(ppg).reshape(-1)), fs, "有 IMU" if acc is not None else "无 IMU"))
    r = d.denoise(ppg, fs=fs, acc=acc, gyro=gyro, want_hr=not args.no_hr)
    p_npz = r.to_npz(os.path.join(args.out, "result.npz"))
    with open(os.path.join(args.out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(r.to_json(), f, ensure_ascii=False, indent=2)
    print("[run] 完成：%.2f s | 模式 %s | 伪影占比 %.3f | 旁路 %s | HR 中位 %s bpm"
          % (r.meta["elapsed_s"], r.meta["mode"], r.quality["artifact_ratio"],
             r.quality["bypass_applied"],
             ("—" if r.hr_median is None else "%.1f" % r.hr_median)))
    if ref is not None:
        ref = np.asarray(ref, float).reshape(-1)[:len(r.ppg_clean)]
        prd = 100 * np.linalg.norm(r.ppg_clean[:len(ref)] - ref) / (np.linalg.norm(ref) + 1e-9)
        prd_raw = 100 * np.linalg.norm(r.ppg_raw[:len(ref)] - ref) / (np.linalg.norm(ref) + 1e-9)
        print("[run] 参考对比：PRD 原波形 %.2f → 滤净 %.2f%s"
              % (prd_raw, prd,
                 "（合成样例：仅验证接口可用，PRD 不代表方法性能；真实数据上 CIME 测试集 PRD 为 46.57）"
                 if _is_synthetic else ""))
    if not args.no_plot:
        png = d.plot(r, os.path.join(args.out, "waveforms.png"), reference=ref)
        print("[run] 图已保存：%s" % png)
    print("[run] 产物：%s（result.npz / result.json / waveforms.png）" % os.path.abspath(args.out))

    if args.eval_test > 0:
        dtest = np.load(os.path.join(P.ROOT, "outputs", "rls_unet", "data", "cime_test.npz"))
        noisy, clean = dtest["noisy"], dtest["clean"]
        n = min(args.eval_test, len(clean))
        prd_r, prd_c = [], []
        for i in range(n):
            ri = d.denoise(noisy[i, 0], fs=100.0, acc=noisy[i, 1:2],
                           gyro=noisy[i, 2:3], want_hr=False)
            ref_i = clean[i]
            prd_r.append(100 * np.linalg.norm(ri.ppg_raw - ref_i) / np.linalg.norm(ref_i))
            prd_c.append(100 * np.linalg.norm(ri.ppg_clean - ref_i) / np.linalg.norm(ref_i))
        print("[eval] API 自检（前 %d 段）：原波形 PRD %.2f → 滤净 PRD **%.2f**"
              "（官方测试集 1900 段口径为 46.37）"
              % (n, float(np.mean(prd_r)), float(np.mean(prd_c))))
    return r


if __name__ == "__main__":
    main()

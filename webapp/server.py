#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PPG 运动伪影去除 · Web 工作台后端

单文件 Flask 服务：托管静态页面 + 调用仓库内的 PPGDenoiser 推理接口。

启动：
    python webapp/server.py --host 127.0.0.1 --port 8000
然后浏览器打开 http://127.0.0.1:8000

仅依赖仓库 requirements（numpy/scipy/torch/matplotlib）+ Flask。
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

# ---- 冻结（PyInstaller 打包）支持 ----------------------------------------
# 打包后：资源在 sys._MEIPASS（onefile）或 exe 同级目录（onedir）；源码运行时在仓库根。
FROZEN = bool(getattr(sys, "frozen", False))
# 无控制台（windowed）模式下 stdout/stderr 为 None：先挂到日志文件，避免 print/logging 崩溃
_LOG_FH = None
if FROZEN and (sys.stdout is None or sys.stderr is None):
    try:
        _log_path = Path(sys.executable).resolve().parent / "workbench.log"
        _LOG_FH = open(_log_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        import io
        _LOG_FH = io.StringIO()
    if sys.stdout is None:
        sys.stdout = _LOG_FH
    if sys.stderr is None:
        sys.stderr = _LOG_FH
if FROZEN:
    RES = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    BASE = RES / "webapp"
    ROOT = RES
else:
    BASE = Path(__file__).resolve().parent
    ROOT = BASE.parent

# ppg_api 用**文件路径**加载 pipeline.py；冻结模式下它不在 __file__ 相对位置
# → 通过环境变量显式告知（ppg_api 已兼容读取该变量）。
_PIPELINE_PY = ROOT / "scripts" / "pipeline" / "pipeline.py"
if _PIPELINE_PY.exists():
    os.environ.setdefault("PPG_PIPELINE_PY", str(_PIPELINE_PY))
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "modules")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from ppg_api import PPGDenoiser, FS_MODEL, _as_1d, _fill_nan  # noqa: E402

APP = Flask(__name__, static_folder=str(BASE / "static"), static_url_path="/static")
APP.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
APP.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024          # 上传上限 64 MB
APP.config["JSON_AS_ASCII"] = False

MODEL_DIR = ROOT / "models" / "ppg_denoise_v1"
MANIFEST_PATH = MODEL_DIR / "manifest.json"
MAX_RUNS = 12
MAX_RUN_SAMPLES = 40_000_000
MAX_INPUT_SAMPLES = 2_000_000
PLOT_POINTS = 3000

RUNS: "OrderedDict[str, dict]" = OrderedDict()
RUNS_LOCK = threading.Lock()
MODELS: dict[str, PPGDenoiser] = {}
MODEL_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _num(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _int(v, default=None):
    x = _num(v, None)
    return int(x) if x is not None else default


def _flag(v, default=True):
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "off", "no", "")


def _clean(obj):
    """把 NaN/Inf/numpy 类型递归转为合法 JSON 值。"""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _downsample_idx(x: np.ndarray, target: int = PLOT_POINTS) -> np.ndarray:
    """用 min/max 包络降采样，保留尖峰，同时限制前端点数。"""
    n = len(x)
    if n <= target:
        return np.arange(n)
    k = int(math.ceil(n / target))
    idx: list[int] = []
    for s in range(0, n, k):
        seg = x[s:s + k]
        if len(seg) == 0:
            continue
        idx.append(s + int(np.nanargmin(seg)))
        idx.append(s + int(np.nanargmax(seg)))
    idx.extend([0, n - 1])
    return np.unique(np.asarray(idx, dtype=int))


def _resample_to(x: np.ndarray, fs_in: float, fs_out: float = FS_MODEL) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if abs(fs_in - fs_out) < 1e-9:
        return x
    from fractions import Fraction
    from scipy.signal import resample_poly
    fr = Fraction(fs_out / fs_in).limit_denominator(200)
    return resample_poly(x, fr.numerator, fr.denominator)


def _align_reference(ref, fs_ref: float, n_out: int):
    """参考波形 → 与输出等长的 100 Hz 波形；长度不一致时按相对时间插值。"""
    if ref is None:
        return None
    r = _fill_nan(_as_1d(ref))
    if len(r) == 0:
        return None
    r = _resample_to(r, fs_ref)
    if len(r) != n_out:
        if len(r) == 1:
            r = np.full(n_out, r[0])
        else:
            src = np.linspace(0.0, 1.0, len(r))
            dst = np.linspace(0.0, 1.0, n_out)
            r = np.interp(dst, src, r)
    return r


def _reference_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred = np.asarray(pred, float).reshape(-1)
    ref = np.asarray(ref, float).reshape(-1)
    n = min(len(pred), len(ref))
    pred, ref = pred[:n], ref[:n]
    err = pred - ref
    nr = float(np.linalg.norm(ref))
    ne = float(np.linalg.norm(err))
    out = {
        "n": int(n),
        "prd": round(100.0 * ne / (nr + 1e-12), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "pearson": None,
        "snr_db": None,
    }
    if np.std(pred) > 1e-12 and np.std(ref) > 1e-12:
        out["pearson"] = round(float(np.corrcoef(pred, ref)[0, 1]), 4)
    ve = float(np.var(err))
    if ve > 1e-12:
        out["snr_db"] = round(float(10.0 * np.log10(float(np.var(ref)) / ve)), 2)
    return out


def _hr_times(n_hr: int) -> np.ndarray:
    # 8 s 窗 / 2 s 步进，窗中心 = 4 + 2i
    return 4.0 + 2.0 * np.arange(n_hr, dtype=float)


# ---------------------------------------------------------------------------
# 数据源
# ---------------------------------------------------------------------------
def _ppg_template(ph):
    """真实感 PPG 波形模板（相位 ph 为弧度）：基波 + 2/3/4 次谐波 + 相移。

    比"三个纯正弦"更接近真实脉搏：收缩峰陡峭、下降支带重搏波切迹，
    使残差网络的 Δ 落在训练分布内（v1 的纯正弦输入会被网络过度修正）。
    """
    return (np.sin(ph)
            + 0.38 * np.sin(2.0 * ph - 0.62)
            + 0.20 * np.sin(3.0 * ph - 1.35)
            + 0.09 * np.sin(4.0 * ph - 2.00))


# 各场景：κ = 伪影标准差 / 脉搏标准差（与真实 CIME 的 κ 分布对齐）
DEMO_KAPPA = {"rest": 0.10, "moderate": 1.0, "strong": 2.6}
# 演示参数预设：静息场景刻意把旁路阈值调高，用来展示"伪影过小时不做处理"的可靠性门控
DEMO_DEFAULTS = {
    "rest": {"bypass_ratio": 0.50},
    "moderate": {},
    "strong": {},
}
DEMO_NOISE = {"rest": 1.0, "moderate": 2.4, "strong": 4.2}
DEMO_LABELS = {
    "rest": "静息（弱伪影：应触发静息旁路，输出≈原波形）",
    "moderate": "中等运动（默认：伪影与脉搏同量级，最接近赛题主场景）",
    "strong": "强运动（边界：伪影远大于脉搏，用于观察方法失效边界）",
}


def synth_demo(scenario: str = "moderate"):
    """合成演示信号（v2）：真实 PPG 形态 + 与 IMU 非线性相关的运动伪影。

    与 v1 的区别（v1 是 3 个纯正弦 + 伪影的精确线性副本，对网络属分布外）：
    1. 干净脉搏用真实模板（收缩峰/重搏波）+ 呼吸基线漂移 + 呼吸性心率变异；
    2. 伪影是幅度调制的多谐波（非纯正弦），强度按 κ 标定到真实 CIME 的水平；
    3. ACC/Gyro 是伪影的**非线性**变换（tanh + 微分 + 噪声）→ RLS 无法完美估计，
       更接近真实设备；避免 RLS 在弱伪影时把脉搏本身也拟合掉。

    仍为**合成信号**：PRD 只用于验证端到端链路，不代表真实方法性能（真实 CIME 46.57）。
    """
    fs, n = 100.0, 3000
    t = np.arange(n) / fs

    # ---- 干净脉搏：真实形态 + 呼吸基线 + 心率变异 ----
    hr0 = 70.0 / 60.0
    hr_t = hr0 * (1.0 + 0.035 * np.sin(2 * np.pi * 0.25 * t + 0.3))     # ±3.5% RSA
    ph = 2.0 * np.pi * np.cumsum(hr_t) / fs
    pulse = _ppg_template(ph)
    pulse = pulse / (np.std(pulse) + 1e-9) * 22.0                        # 幅度 ~22 a.u.
    resp = 3.0 * np.sin(2 * np.pi * 0.22 * t + 0.7)                      # 呼吸基线漂移
    clean = pulse + resp

    # ---- 运动伪影：幅度调制的多谐波，强度按 κ 标定 ----
    kappa = DEMO_KAPPA.get(scenario, 1.0)
    f_m = 1.85                                                           # 步频 ~1.85 Hz
    am = 1.0 + 0.45 * np.sin(2 * np.pi * 0.33 * t + 0.2)                 # 幅度调制
    m = am * (np.sin(2 * np.pi * f_m * t)
              + 0.55 * np.sin(2 * np.pi * 2 * f_m * t + 0.8)
              + 0.22 * np.sin(2 * np.pi * 3 * f_m * t + 1.9))
    m = m / (np.std(m) + 1e-9) * (kappa * np.std(clean))
    # 慢漂移（手部/身体姿态变化），与脉搏无关
    drift = 0.35 * kappa * np.std(clean) * np.cumsum(np.random.RandomState(3).randn(n)) / np.sqrt(n)
    motion = m + drift

    # ---- IMU：**物理幅度**的加速度/角速度（含固定传感器噪声底）----
    # ⚠️ 关键：ACC 幅度必须随真实运动幅度变化（κ 小 → ACC 小 → RLS 无法凭空拟合），
    #    同时带固定噪声底，使弱运动时估计质量下降——这才是真实设备的物理关系。
    rng = np.random.RandomState(7)
    ms = np.std(motion) + 1e-9
    acc_gain = 0.02                                   # 伪影 a.u. → ~g
    acc_x = acc_gain * (motion + 0.25 * motion ** 3 / ms ** 2)   # 轻微非线性
    acc_x = acc_x + 0.055 * rng.randn(n)                          # 固定噪声底
    acc_y = 0.65 * np.roll(acc_x, 3) + 0.045 * rng.randn(n)
    acc = np.vstack([acc_x, acc_y, 0.25 * np.roll(acc_x, 7) + 0.03 * rng.randn(n)])
    g = np.gradient(motion)
    gyro_y = 0.9 * g / (np.std(g) + 1e-9) * (ms * 0.02) + 0.05 * rng.randn(n)
    gyro = np.vstack([0.2 * np.roll(gyro_y, 5), gyro_y, np.zeros(n)])

    # ---- 传感器噪声（随运动增强）+ 含噪 PPG ----
    noise_sd = DEMO_NOISE.get(scenario, 2.4)
    ppg = clean + motion + noise_sd * rng.randn(n)

    return dict(ppg=ppg, acc=acc, gyro=gyro, reference=clean, fs=fs,
                description=DEMO_LABELS.get(scenario, scenario), scenario=scenario,
                kappa=float(kappa), synthetic=True,
                demo_defaults=dict(DEMO_DEFAULTS.get(scenario, {})))


def _npz_from_bytes(raw: bytes):
    d = np.load(io.BytesIO(raw), allow_pickle=False)
    keys = {str(k).lower(): k for k in d.files}
    if "ppg" not in keys:
        raise ValueError("npz 缺少 ppg 键；实际键：%s" % list(d.files)[:12])
    ppg = d[keys["ppg"]]
    acc = d[keys["acc"]] if "acc" in keys else None
    gyro = d[keys["gyro"]] if "gyro" in keys else None
    ref = (d[keys["reference"]] if "reference" in keys
           else (d[keys["clean"]] if "clean" in keys else None))
    fs = None
    if "fs" in keys and np.ndim(d[keys["fs"]]) == 0:
        fs = float(d[keys["fs"]])
    return ppg, acc, gyro, ref, fs


def _csv_from_bytes(raw: bytes):
    text = raw.decode("utf-8-sig", errors="replace")
    rd = list(csv.DictReader(io.StringIO(text)))
    if not rd:
        raise ValueError("CSV 为空")
    cols = {str(c).strip().lower(): c for c in rd[0].keys() if c is not None}

    def get(name):
        if name not in cols:
            return None
        vals = []
        for row in rd:
            try:
                vals.append(float(row[cols[name]]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        return np.asarray(vals, dtype=float)

    ppg = get("ppg")
    if ppg is None:
        raise ValueError("CSV 缺少 ppg 列（需要列名 ppg / acc_x,acc_y,acc_z / gyro_x..z / reference）")
    axes = [get("acc_%s" % a) for a in "xyz"]
    acc = np.stack([a for a in axes if a is not None]) if any(a is not None for a in axes) else None
    gaxes = [get("gyro_%s" % a) for a in "xyz"]
    gyro = np.stack([g for g in gaxes if g is not None]) if any(g is not None for g in gaxes) else None
    ref = get("reference")
    return ppg, acc, gyro, ref, None


def _json_from_bytes(raw: bytes):
    obj = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("JSON 顶层应为对象，例如 {\"ppg\": [...], \"fs\": 100}")
    low = {str(k).lower(): v for k, v in obj.items()}
    if "ppg" not in low:
        raise ValueError("JSON 缺少 ppg 字段")
    ref = low.get("reference", low.get("clean"))
    fs = low.get("fs")
    return (np.asarray(low["ppg"], float),
            None if low.get("acc") is None else np.asarray(low["acc"], float),
            None if low.get("gyro") is None else np.asarray(low["gyro"], float),
            None if ref is None else np.asarray(ref, float),
            None if fs is None else float(fs))


def load_input(req) -> dict:
    """从 multipart 表单 / JSON 请求解析输入。"""
    if req.is_json:
        body = req.get_json(force=True) or {}
        source = str(body.get("source", "json")).lower()
        if source == "demo":
            d = synth_demo(str(body.get("scenario", "moderate")))
            d.update(source="demo", filename=None)
            return d
        if "ppg" not in body:
            raise ValueError("JSON 请求需要 ppg 字段")
        ref = body.get("reference", body.get("clean"))
        return dict(source="json", filename=body.get("filename"),
                    ppg=np.asarray(body["ppg"], float),
                    acc=None if body.get("acc") is None else np.asarray(body["acc"], float),
                    gyro=None if body.get("gyro") is None else np.asarray(body["gyro"], float),
                    reference=None if ref is None else np.asarray(ref, float),
                    fs=None if body.get("fs") is None else float(body["fs"]))

    source = str(req.form.get("source", "demo")).lower()
    if source == "demo":
        d = synth_demo(str(req.form.get("scenario", "moderate")))
        d.update(source="demo", filename=None)
        return d

    f = req.files.get("file")
    if f is None or not f.filename:
        raise ValueError("请上传文件，或选择演示数据")
    raw = f.read()
    name = f.filename
    low = name.lower()
    if low.endswith(".npz"):
        ppg, acc, gyro, ref, fs = _npz_from_bytes(raw)
    elif low.endswith(".csv"):
        ppg, acc, gyro, ref, fs = _csv_from_bytes(raw)
    elif low.endswith(".json"):
        ppg, acc, gyro, ref, fs = _json_from_bytes(raw)
    else:
        raise ValueError("仅支持 .npz / .csv / .json")
    return dict(source="file", filename=name, ppg=ppg, acc=acc, gyro=gyro,
                reference=ref, fs=fs)


# ---------------------------------------------------------------------------
# 模型 / 推理
# ---------------------------------------------------------------------------
def get_model(preset: str) -> PPGDenoiser:
    with MODEL_LOCK:
        if preset not in MODELS:
            d = PPGDenoiser(preset=preset, device="cpu", verbose=True,
                            model_dir=str(MODEL_DIR / preset)).load()
            MODELS[preset] = d
        return MODELS[preset]


def _sig_len(x) -> int:
    a = np.asarray(x)
    if a.ndim <= 1:
        return int(a.shape[0]) if a.ndim == 1 else 1
    if a.shape[0] <= 4 and a.shape[1] > 4:
        return int(a.shape[1])
    if a.shape[1] <= 4 and a.shape[0] > 4:
        return int(a.shape[0])
    return int(a.shape[-1])


def _execute(inp: dict, params: dict) -> dict:
    ppg = _fill_nan(_as_1d(inp["ppg"]))
    n_in = len(ppg)
    if n_in < 2:
        raise ValueError("输入过短")
    if n_in > MAX_INPUT_SAMPLES:
        raise ValueError("输入过长（%d 点）；请先降采样或分段处理（上限 %d）"
                         % (n_in, MAX_INPUT_SAMPLES))
    for key, label in (("acc", "ACC"), ("gyro", "Gyro")):
        if inp.get(key) is not None and _sig_len(inp[key]) != n_in:
            raise ValueError("%s 长度（%d）与 PPG（%d）不一致；请保证同一采样率、同一时间轴"
                             % (label, _sig_len(inp[key]), n_in))
    fs = params.get("fs") or inp.get("fs")
    if fs is None:
        raise ValueError("未提供采样率：请在表单填写 fs，或在 npz/JSON 中加入 fs 字段")
    fs = float(fs)
    if not (0.5 <= fs <= 10000):
        raise ValueError("采样率 fs 应在 0.5–10000 Hz 之间")

    has_acc = inp.get("acc") is not None
    has_gyro = inp.get("gyro") is not None and np.any(np.asarray(inp["gyro"]))
    effective_preset = params["preset"]
    auto_switched = False
    if effective_preset != "no_imu" and not has_acc and params["auto_no_imu"]:
        effective_preset = "no_imu"
        auto_switched = True

    warnings = []
    if inp.get("source") == "demo":
        warnings.append("当前为合成演示信号：PRD 仅用于验证接口与流程，不代表方法在真实数据上的性能。")
    if auto_switched:
        warnings.append("未提供 ACC：已自动切换到 no_imu 单通道模型（伪影波形恒为 0）。")
    if effective_preset == "no_imu" and has_acc:
        warnings.append("选择了 no_imu 预设：IMU 不参与计算，仅用 PPG 单通道模型（伪影波形恒为 0）。")

    # no_imu 预设按定义只使用 PPG：即使输入带 IMU 也不参与 RLS/伪影估计
    acc_use = None if effective_preset == "no_imu" else inp.get("acc")
    gyro_use = None if effective_preset == "no_imu" else inp.get("gyro")

    model = get_model(effective_preset)
    with INFER_LOCK:
        model.alpha = params["alpha"]
        model.crm_weight = params["crm_weight"]
        model.bypass_ratio = params["bypass_ratio"]
        model.taps = params["taps"]
        model.lam = params["lam"]
        res = model.denoise(
            inp["ppg"], fs=fs,
            acc=acc_use, gyro=gyro_use,
            want_hr=params["want_hr"],
        )

    if res.quality.get("bypass_applied"):
        warnings.append("伪影能量占比低于阈值，已触发静息旁路：输出等于原波形。")

    synthetic = bool(inp.get("synthetic") or inp.get("source") == "demo")
    # 链路自检（演示态代替 PRD 展示：PRD 对合成数据无物理意义）
    link_check = {
        "length_ok": bool(len(res.ppg_clean) == len(res.ppg_raw)),
        "finite": bool(np.isfinite(res.ppg_clean).all()),
        "changed": bool(np.max(np.abs(res.ppg_clean - res.ppg_raw)) > 1e-9),
        "hr_ok": bool(res.hr_median is not None and np.isfinite(res.hr_median)),
        "n_hr": int(0 if res.hr_bpm is None else len(res.hr_bpm)),
        "artifact_nonzero": bool(np.any(np.abs(res.artifact) > 0)),
        "bypass": bool(res.quality.get("bypass_applied")),
    }
    ref = _align_reference(inp.get("reference"), fs, len(res.ppg_raw))
    ref_metrics = None
    if ref is not None:
        ref_metrics = {
            "raw": _reference_metrics(res.ppg_raw, ref),
            "clean": _reference_metrics(res.ppg_clean, ref),
        }

    idx = _downsample_idx(res.ppg_raw)
    series = {
        "t": np.round(res.t[idx], 3).tolist(),
        "ppg_raw": np.round(res.ppg_raw[idx], 5).tolist(),
        "artifact": np.round(res.artifact[idx], 5).tolist(),
        "ppg_prior": np.round(res.ppg_prior[idx], 5).tolist(),
        "ppg_clean": np.round(res.ppg_clean[idx], 5).tolist(),
        "reference": (np.round(ref[idx], 5).tolist() if ref is not None else None),
    }
    hr = None
    if res.hr_bpm is not None:
        hr = [None if not np.isfinite(v) else round(float(v), 2) for v in np.asarray(res.hr_bpm)]

    payload = {
        "ok": True,
        "run_id": None,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": {
            "source": inp.get("source"),
            "filename": inp.get("filename"),
            "scenario": inp.get("scenario"),
            "description": inp.get("description"),
            "fs": fs,
            "n": int(n_in),
            "duration_s": round(n_in / fs, 3),
            "has_acc": has_acc,
            "has_gyro": bool(has_gyro),
            "has_reference": ref is not None,
            "preset_requested": params["preset"],
            "preset_effective": effective_preset,
            "auto_switched": auto_switched,
        },
        "result": {
            **res.to_json(),
            "hr_t": (np.round(_hr_times(len(res.hr_bpm)), 2).tolist()
                     if res.hr_bpm is not None else []),
            "hr_bpm": hr if hr is not None else [],
        },
        "series": series,
        "reference_metrics": ref_metrics,
        "synthetic": synthetic,
        "link_check": link_check,
        "warnings": warnings,
        "params": params,
    }
    payload = _clean(payload)

    run_id = uuid.uuid4().hex[:12]
    payload["run_id"] = run_id
    payload["downloads"] = {
        "npz": "/api/download/%s/npz" % run_id,
        "json": "/api/download/%s/json" % run_id,
        "csv": "/api/download/%s/csv" % run_id,
        "png": "/api/download/%s/png" % run_id,
    }
    rec = {
        "id": run_id,
        "payload": payload,
        "result": res,
        "reference": ref,
        "input": payload["input"],
        "created": time.time(),
    }
    with RUNS_LOCK:
        RUNS[run_id] = rec
        while RUNS and (len(RUNS) > MAX_RUNS
                        or sum(len(x["result"].t) for x in RUNS.values()) > MAX_RUN_SAMPLES):
            RUNS.popitem(last=False)
    payload["server"] = {
        "elapsed_s": res.meta.get("elapsed_s"),
        "n_model": res.meta.get("n_residual", 0) + res.meta.get("n_crm", 0),
        "models_loaded": sorted(MODELS.keys()),
    }
    return payload


def _parse_params(req) -> dict:
    src = (req.get_json(force=True) or {}) if req.is_json else req.form

    def get(k, default=None):
        return src.get(k, default) if hasattr(src, "get") else default

    return {
        "preset": str(get("preset", "lite")).lower(),
        "auto_no_imu": _flag(get("auto_no_imu"), True),
        "fs": _num(get("fs"), None),
        "alpha": min(1.5, max(0.0, _num(get("alpha"), 0.40))),
        "crm_weight": min(1.0, max(0.0, _num(get("crm_weight"), 0.15))),
        "bypass_ratio": min(1.0, max(0.0, _num(get("bypass_ratio"), 0.05))),
        "taps": min(16, max(1, _int(get("taps"), 4))),
        "lam": min(0.99999, max(0.8, _num(get("lam"), 0.995))),
        "want_hr": _flag(get("want_hr"), True),
    }


# ---------------------------------------------------------------------------
# 下载产物
# ---------------------------------------------------------------------------
def _record(run_id: str) -> dict:
    with RUNS_LOCK:
        rec = RUNS.get(run_id)
    if rec is None:
        raise KeyError("结果已过期或不存在：%s" % run_id)
    return rec


def _json_bytes(rec: dict) -> bytes:
    r = rec["result"]
    doc = r.to_json()
    doc["input"] = rec["input"]
    doc["reference_metrics"] = rec["payload"].get("reference_metrics")
    doc["note"] = ("hr_bpm 为逐窗心率；series 全量数据见 result.npz；"
                   "npz 键：t/ppg_raw/artifact/ppg_prior/ppg_clean/hr_bpm/hr_t/reference")
    return (json.dumps(_clean(doc), ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _npz_bytes(rec: dict) -> bytes:
    r = rec["result"]
    buf = io.BytesIO()
    arrs = dict(
        t=r.t, ppg_raw=r.ppg_raw, artifact=r.artifact, ppg_prior=r.ppg_prior,
        ppg_clean=r.ppg_clean,
        hr_bpm=(np.asarray(r.hr_bpm) if r.hr_bpm is not None else np.zeros(0)),
        hr_t=(_hr_times(len(r.hr_bpm)) if r.hr_bpm is not None else np.zeros(0)),
    )
    if rec.get("reference") is not None:
        arrs["reference"] = rec["reference"]
    np.savez_compressed(buf, **arrs)
    buf.seek(0)
    return buf.read()


def _csv_bytes(rec: dict) -> bytes:
    r = rec["result"]
    ref = rec.get("reference")
    cols = ["t", "ppg_raw", "artifact", "ppg_prior", "ppg_clean"]
    if ref is not None:
        cols.append("reference")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for i in range(len(r.t)):
        row = [round(float(r.t[i]), 5), round(float(r.ppg_raw[i]), 8),
               round(float(r.artifact[i]), 8), round(float(r.ppg_prior[i]), 8),
               round(float(r.ppg_clean[i]), 8)]
        if ref is not None:
            row.append(round(float(ref[i]), 8))
        w.writerow(row)
    return buf.getvalue().encode("utf-8-sig")


def _png_bytes(rec: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = rec["result"]
    ref = rec.get("reference")
    t = r.t
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, r.ppg_raw, color="#2f7cf6", lw=0.9)
    axes[0].set_ylabel("1) Raw PPG\n(with artifact)")
    axes[0].set_title("PPG Motion-Artifact Removal", fontsize=13, fontweight="bold")
    axes[1].plot(t, r.artifact, color="#e5484d", lw=0.9)
    axes[1].set_ylabel(r"2) Estimated artifact $\hat{M}$")
    axes[2].plot(t, r.ppg_clean, color="#16a34a", lw=0.9, label="clean output")
    if ref is not None:
        axes[2].plot(t, ref, color="#6b7280", lw=0.8, ls="--", label="reference")
        axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_ylabel("3) Clean PPG")
    axes[2].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.margins(x=0)
    q = r.quality
    info = ("HR median %s bpm  |  artifact ratio %.3f  |  bypass %s  |  mode %s  |  %.2f s"
            % ("--" if r.hr_median is None else "%.1f" % r.hr_median,
               q.get("artifact_ratio", 0), q.get("bypass_applied"),
               r.meta.get("mode"), r.meta.get("elapsed_s", 0)))
    fig.text(0.5, 0.005, info, ha="center", fontsize=9, color="#333333")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@APP.get("/")
def index():
    return send_from_directory(APP.static_folder, "index.html")


@APP.post("/api/shutdown")
def shutdown():
    """退出应用（无控制台打包版没有 Ctrl+C，由网页「退出应用」按钮调用）。"""
    def _later():
        time.sleep(0.4)
        os._exit(0)
    threading.Thread(target=_later, daemon=True).start()
    return jsonify({"ok": True, "message": "服务正在退出，可以关闭本页面"})


@APP.get("/api/health")
def health():
    import torch
    return jsonify({
        "ok": True,
        "service": "PPG artifact removal webapp",
        "version": "2.0",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "models_loaded": sorted(MODELS.keys()),
        "runs": len(RUNS),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


@APP.get("/api/info")
def info():
    manifest = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
    presets = {}
    for name, meta in (manifest.get("presets") or {}).items():
        presets[name] = {
            "expected_test_prd": meta.get("expected_test_prd"),
            "n_residual": meta.get("n_residual"),
            "n_crm": meta.get("n_crm"),
            "total_bytes": meta.get("total_bytes"),
            "note": meta.get("note"),
            "available": (MODEL_DIR / name).is_dir(),
        }
    if "no_imu" in manifest:
        m = manifest["no_imu"]
        presets["no_imu"] = {
            "expected_test_prd": None,
            "n_residual": len(m.get("files") or {}),
            "n_crm": 0,
            "total_bytes": sum(v.get("bytes", 0) for v in (m.get("files") or {}).values()),
            "note": m.get("note"),
            "available": (MODEL_DIR / "no_imu").is_dir(),
        }

    ds_candidates = [
        os.environ.get("PPG_DATASETS"),
        ROOT / "datasets",
        ROOT.parent / "datasets",
    ]
    datasets, seen_ds = [], set()
    for ds_root in ds_candidates:
        if not ds_root:
            continue
        ds_root = Path(ds_root)
        if not ds_root.is_dir():
            continue
        for d in sorted(p for p in ds_root.iterdir() if p.is_dir()):
            if d.name in seen_ds:
                continue
            files = [f for f in d.rglob("*") if f.is_file()]
            if not files:
                continue
            seen_ds.add(d.name)
            datasets.append({
                "name": d.name,
                "files": len(files),
                "bytes": sum(f.stat().st_size for f in files),
            })

    return jsonify({
        "ok": True,
        "manifest": manifest,
        "presets": presets,
        "defaults": {
            "preset": "lite", "alpha": 0.40, "crm_weight": 0.15,
            "bypass_ratio": 0.05, "taps": 4, "lam": 0.995,
            "auto_no_imu": True, "want_hr": True,
        },
        "limits": {
            "max_upload_mb": APP.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            "max_input_samples": MAX_INPUT_SAMPLES,
            "fs_model": FS_MODEL,
        },
        "demo": {
            "note": "内置合成演示数据（真实 PPG 形态 + 物理幅度 IMU）。合成样例不计算 PRD；"
                    "上传含 reference 的真实数据后显示 PRD/Pearson/SNR。",
            "scenarios": [
                {"key": k, "label": DEMO_LABELS[k], "kappa": DEMO_KAPPA[k],
                 "defaults": DEMO_DEFAULTS.get(k, {})}
                for k in ("rest", "moderate", "strong")
            ],
        },
        "datasets": datasets,
        "report": {
            "version": "2.0",
            "cime_prd_raw": 103.41, "cime_prd_clean": 46.57,
            "cime_pearson": 0.9395, "cime_snr_db": 7.78,
            "lite_prd": 46.37,
            "hr_mae_range": "1.06–16.03 bpm",
        },
    })


@APP.get("/api/demo")
def demo():
    scenario = str(request.args.get("scenario", "moderate")).lower()
    d = synth_demo(scenario)
    idx = _downsample_idx(d["ppg"])
    return jsonify(_clean({
        "ok": True,
        "scenario": scenario,
        "description": d["description"],
        "fs": d["fs"],
        "n": int(len(d["ppg"])),
        "duration_s": round(len(d["ppg"]) / d["fs"], 3),
        "series": {
            "t": np.round(np.arange(len(d["ppg"])) / d["fs"], 3)[idx].tolist(),
            "ppg_raw": np.round(d["ppg"][idx], 5).tolist(),
            "reference": np.round(d["reference"][idx], 5).tolist(),
        },
    }))


@APP.post("/api/denoise")
def denoise():
    try:
        params = _parse_params(request)
        if params["preset"] not in ("lite", "full", "no_imu"):
            raise ValueError("preset 只能是 lite / full / no_imu")
        inp = load_input(request)
        payload = _execute(inp, params)
        return jsonify(payload)
    except Exception as exc:                            # noqa: BLE001
        code = 400 if isinstance(exc, (ValueError, KeyError, FileNotFoundError)) else 500
        return jsonify({
            "ok": False,
            "error": str(exc),
            "trace": traceback.format_exc() if APP.debug else None,
        }), code


@APP.post("/api/selfcheck")
def selfcheck():
    """一键自检：合成中等运动样例 → 完整推理 → 报告 PRD。"""
    try:
        d = synth_demo("moderate")
        params = {
            "preset": "lite", "auto_no_imu": True, "fs": 100.0,
            "alpha": 0.40, "crm_weight": 0.15, "bypass_ratio": 0.05,
            "taps": 4, "lam": 0.995, "want_hr": True,
        }
        payload = _execute(d, params)
        rm = payload.get("reference_metrics") or {}
        return jsonify(_clean({
            "ok": True,
            "link_check": payload.get("link_check"),
            "prd_raw": (rm.get("raw") or {}).get("prd"),
            "prd_clean": (rm.get("clean") or {}).get("prd"),
            "hr_median": payload["result"].get("hr_median"),
            "mode": payload["result"].get("meta", {}).get("mode"),
            "elapsed_s": payload["result"].get("meta", {}).get("elapsed_s"),
            "note": "合成样例仅验证端到端链路；真实 CIME 测试集 PRD 见报告（46.57）。",
        }))
    except Exception as exc:                            # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc),
                        "trace": traceback.format_exc() if APP.debug else None}), 500


@APP.get("/api/run/<run_id>")
def get_run(run_id):
    try:
        rec = _record(run_id)
        return jsonify(rec["payload"])
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404


@APP.get("/api/runs")
def list_runs():
    with RUNS_LOCK:
        items = [{
            "run_id": rid,
            "created": rec["payload"].get("created"),
            "source": rec["input"].get("source"),
            "filename": rec["input"].get("filename"),
            "hr_median": rec["payload"]["result"].get("hr_median"),
            "mode": rec["payload"]["result"].get("meta", {}).get("mode"),
            "preset": rec["input"].get("preset_effective"),
        } for rid, rec in reversed(RUNS.items())]
    return jsonify({"ok": True, "runs": items})


@APP.get("/api/download/<run_id>/<kind>")
def download(run_id, kind):
    try:
        rec = _record(run_id)
        base = "ppg_result_%s" % run_id
        if kind == "npz":
            return send_file(io.BytesIO(_npz_bytes(rec)), as_attachment=True,
                             download_name=base + ".npz", mimetype="application/octet-stream")
        if kind == "json":
            return send_file(io.BytesIO(_json_bytes(rec)), as_attachment=True,
                             download_name=base + ".json", mimetype="application/json")
        if kind == "csv":
            return send_file(io.BytesIO(_csv_bytes(rec)), as_attachment=True,
                             download_name=base + "_waveforms.csv",
                             mimetype="text/csv; charset=utf-8")
        if kind == "png":
            return send_file(io.BytesIO(_png_bytes(rec)), as_attachment=True,
                             download_name=base + "_waveforms.png", mimetype="image/png")
        raise KeyError("未知下载类型：%s（npz/json/csv/png）" % kind)
    except KeyError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404


def _pick_port(port: int, host: str = "127.0.0.1", tries: int = 20) -> int:
    """端口被占用时自动顺延，避免演示现场手动改参数。"""
    import socket
    for p in [port] + list(range(port + 1, port + tries)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return port


def _setup_frozen_log():
    """打包版：日志同时写到 exe 同目录，便于用户反馈问题。"""
    if not FROZEN:
        return None
    try:
        base = Path(sys.executable).resolve().parent
        fh = open(base / "workbench.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = fh
        print("=" * 60)
        print("启动：", time.strftime("%Y-%m-%d %H:%M:%S"))
        return fh
    except Exception:
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PPG 运动伪影去除 Web 工作台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-browser", action="store_true",
                    help="不自动打开浏览器（打包版默认自动打开）")
    args = ap.parse_args()

    # 这类小模型在 128 核机器上被线程同步开销拖垮（实测 4.9s → 0.1s）
    try:
        import torch
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 16)))
    except Exception:
        pass

    _setup_frozen_log()
    port = _pick_port(args.port, "127.0.0.1" if args.host in ("127.0.0.1", "0.0.0.0") else args.host)
    url = "http://%s:%d" % ("127.0.0.1" if args.host == "0.0.0.0" else args.host, port)
    print("=" * 68)
    print("  PPG 运动伪影去除 · Web 工作台")
    print("  打开：%s" % url)
    print("  模型：%s" % MODEL_DIR)
    print("  数据：内置合成演示数据；可导入 .npz / .csv / .json")
    print("  Ctrl+C 停止（或直接关闭本窗口）")
    print("=" * 68)
    if not args.no_browser:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    APP.run(host=args.host, port=port, debug=args.debug, threaded=True)

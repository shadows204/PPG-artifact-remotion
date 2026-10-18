# PPG 运动伪影去除（第十一届全国生物医学工程创新设计大赛 · 赛道三）

> 单通道 PPG 的运动伪影去除与跨域心率估计。
> **核心思路**：物理先验为主干（RLS 时域自适应抵消）→ 网络只做残差修正（仅训练域）→ 决策层负责跨域可靠性。
> 结果：CIME 波形 PRD **103.41 → 46.57**（Pearson 0.9395 / SNR 7.78 dB）；6 个公开域心率 MAE **1.06–16.03 bpm**。
> 完整方法与实验见 **[实验最终报告.md](实验最终报告.md)**；框架图见 [figures/pipeline_framework.png](figures/pipeline_framework.png)。

---

## 1. 目录结构

```
.
├── README.md                     # 本文件：结构 + 使用方法（含**软件接口用法**）
├── 实验最终报告.md                # 方法流程/方法汇总/创新点/消融/适用范围/baseline
├── requirements.txt
├── run_demo.sh                   # 一键：建环境 → 跑接口 demo → 出三波形图
├── scripts/
│   ├── _common/                  # 路径解析 paths.py、多进程 parallel.py
│   ├── modules/
│   │   ├── ppg_api.py            # ★ 软件接口：PPGDenoiser（三波形输出）
│   │   ├── residual.py crm.py models.py   # 网络结构（残差 U-Net / 频域 CRM / 主干 U-Net）
│   │   ├── ext_loaders.py        # 6 个公开数据集加载器（采样率/通道/NaN 统一）
│   │   ├── metrics.py  dataset.py  rls.py  domain_rand.py  gan.py
│   │   ├── gan.py
│   │   └── (pipeline 主流程见 scripts/pipeline/pipeline.py)
│   ├── pipeline/pipeline.py      # 主流水线（数据→RLS 标定→特征→训练→评估→HR）
│   └── tools/                    # 必要脚本（见 §7）
├── models/ppg_denoise_v1/        # 训练好的权重（manifest.json 含 sha256）
│   ├── lite/                     # 5 残差 + 3 CRM，15 MB，PRD 46.37（推荐，软件用）
│   ├── full/                     # 41 残差 + 3 CRM，108 MB，PRD 46.57
│   └── no_imu/                   # 无 IMU 退化路径（1 通道）
├── datasets/                     # 只放 README（数据需自行下载，见 datasets/README.md）
├── figures/pipeline_framework.png
└── outputs/                      # 运行产物（仓库内为空）
```

---

## 2. 环境安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy scipy torch matplotlib wfdb PyEMD
```

推理只需 `numpy scipy torch matplotlib`；CPU 即可（30 s 信号约 0.2 s）。

---

## 3. 快速开始：软件接口（三波形）

这是给**软件/界面同学**用的主入口：给一段 PPG（可带 IMU），拿到
**① 原波形 ② 伪影波形 M̂ ③ 滤净波形** + 心率 + 质量指标。

### 3.1 三行调用

```python
import sys; sys.path.insert(0, "scripts/modules")
from ppg_api import PPGDenoiser

d = PPGDenoiser(preset="lite")                        # 加载模型（约 0.3 s，CPU 即可）
r = d.denoise(ppg, fs=100, acc=acc, gyro=gyro)        # 任意长度、任意采样率；IMU 可选

r.ppg_raw      # ① 原波形（含伪影）
r.artifact     # ② 估计的伪影波形 M̂（无 IMU 时全 0）
r.ppg_clean    # ③ 滤净波形（最终输出）
r.hr_median    # 心率中位（bpm，可能为 None）
d.plot(r, "waveforms.png")                            # 三波形图（即软件预览可直接复用）
```

### 3.2 输入约定

| 参数 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `ppg` | `(N,)` 或 `(n_lead, N)` | ✅ | 原始 PPG，任意单位（无需定标）；多导联自动取方差最大的一路 |
| `fs` | float | ✅ | 采样率 Hz，**任意**（内部重采样到 100 Hz；25/50/64/125/256 Hz 均可） |
| `acc` | `(3,N)` / `(N,3)` / `(N,)` | ⭕ | 加速度（推荐与 PPG 同侧）；缺省走"无 IMU"分支 |
| `gyro` | 同上 | ⭕ | 陀螺仪；**有则用 "ACC_x + Gyro_y" 异质两路参考**（实测比仅 ACC 好 27%） |
| `want_hr` | bool | ⭕ | 是否同时估心率（默认 True；只画波形可设 False 更快） |

- **支持任意长度**：>30 s 自动 30 s 窗 + 15 s 步进 Hann 重叠相加；<30 s 反射填充、输出长度与输入一致；
  <1 s 抛 `ValueError`。
- **NaN 自动插值**；常数/全零输入触发旁路。
- npz 输入键：`ppg`（必需）、`acc`、`gyro`、`reference`（可选真值）、`fs`（可选标量）。
- CSV 输入列名：`ppg`、`acc_x/acc_y/acc_z`、`gyro_x/gyro_y/gyro_z`、`reference`（可选）。

### 3.3 输出（`DenoiseResult`）

| 字段 | 形状 | 含义 | 用途 |
|---|---|---|---|
| `t` | `(N,)` | 时间轴（100 Hz） | 画图 x 轴 |
| `ppg_raw` | `(N,)` | **① 原波形** | 第一条曲线 |
| `artifact` | `(N,)` | **② 伪影波形 M̂**（无 IMU 时全 0） | 第二条曲线（建议隐藏或标注"无 IMU"） |
| `ppg_prior` | `(N,)` | 物理先验（`ppg_raw − 0.40·M̂`） | 可选对照曲线 |
| `ppg_clean` | `(N,)` | **③ 滤净波形**（最终输出） | 第三条曲线 |
| `hr_bpm` | `(M,)` | 逐窗心率（8 s/2 s，可能含 NaN） | 心率曲线 |
| `hr_median` | float / None | 心率中位 | 数值显示 |
| `quality` | dict | `artifact_ratio` / `bypass_applied` / `snr_improve_db` | 质量徽标、是否"信号干净无需处理" |
| `meta` | dict | 版本/模式（`imu`/`imu_acc`/`no_imu`）/耗时/模型数 | 调试与日志 |

序列化：`r.to_npz("result.npz")`、`r.to_json()`（可直接 `json.dump`）。

### 3.4 行为要点（软件侧须知）

1. **静息旁路**：伪影能量占比 < 0.05 时直接透传原波形（`quality["bypass_applied"]=True`），
   避免"无伪影却过度处理"（实测无伪影时仍有约 15 PRD 本底）。
2. **三种模式**（`meta["mode"]`）：`imu`（PPG+ACC+Gyro，最佳）/ `imu_acc`（仅 ACC）/ `no_imu`（仅 PPG，
   此时 `artifact` 恒为 0，心率可用但无法估计伪影波形）——`no_imu` 会自动加载 1 通道模型。
3. **性能**：30 s 信号 CPU **≈0.2 s**；5 min ≈1.2 s；内存 <1 GB；推理确定（无随机）。
4. **确定性**：同输入逐位一致；不修改全局 torch 状态（不会影响同进程训练）。

### 3.5 命令行版本

```bash
# ① 开箱即用（无需任何数据：内置合成样例，验证接口可用）
python scripts/tools/run_denoise.py --demo --out outputs/denoise_demo
#   → outputs/denoise_demo/{result.npz, result.json, waveforms.png}

# ② 读自己的数据
python scripts/tools/run_denoise.py --input my.npz  --fs 100 --out outputs/out
python scripts/tools/run_denoise.py --input my.csv  --fs 25  --out outputs/out

# ③ 有 CIME 数据时：在真实测试集前 60 段上自检（应得 滤净 PRD 35.72，与官方管线 35.71 对齐）
python scripts/tools/run_denoise.py --demo --out outputs/out --eval-test 60
```

### 3.6 建议的界面布局（三波形）

```
┌─────────────────────────── PPG 运动伪影去除 ───────────────────────────┐
│ ① 原波形 (含伪影)    [蓝]                                                 │
│ ② 伪影波形 M̂        [红]   （无 IMU 时隐藏或标注）                        │
│ ③ 滤净波形          [绿]   （有参考时可叠加虚线）                          │
│ 底部：HR 中位 65.6 bpm ｜ 伪影占比 0.641 ｜ 旁路 False ｜ 模式 imu        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 模型与预设

`models/ppg_denoise_v1/manifest.json` 记录版本、每个权重的 sha256 与期望指标。

| 预设 | 组成 | 大小 | 测试集 PRD | 适用 |
|---|---|---|---|---|
| `lite`（推荐） | 5 残差 + 3 频域 CRM | 15 MB | **46.37** | 软件/部署（比 full 略优且小 7 倍） |
| `full` | 41 残差 + 3 CRM | 108 MB | 46.57 | 复现论文最终数字 |
| `no_imu` | 3 个 1 通道模型 | 7.8 MB | — | 无 IMU 时自动启用 |

```python
PPGDenoiser(preset="lite")      # 或 "full"
PPGDenoiser(model_dir="models/ppg_denoise_v1/no_imu", preset="no_imu")
```

---

## 5. 数据准备

本仓库**不含原始数据**（约 19 GB，且各有许可）。请按 **[datasets/README.md](datasets/README.md)** 的
官方链接自行下载并放到 `datasets/<名称>/`；公开数据集可用：

```bash
python scripts/tools/download_datasets.py --list
python scripts/tools/download_datasets.py --only troika wrist
```

**不下载数据也能跑通软件接口 demo**（§3.5 ①）。

---

## 6. 复现实验

```bash
# 0) 回归门禁：一条命令核对全部关键数字（数据齐备时）
python scripts/tools/regression_check.py

# 1) 波形路径（CIME）：特征 → 训练残差 U-Net → 训练 CRM → 融合
python scripts/pipeline/pipeline.py all            # 或分步：step1_prepare … step11_strata
python scripts/tools/train_residual.py --help      # 单模型训练（多种子/集成）
python scripts/tools/train_crm.py     --help       # 频域 CRM
python scripts/tools/ensemble_cime.py --weights-dir models/ppg_denoise_v1/lite --out outputs/lite_eval
python scripts/tools/fuse_branches.py --res-dir models/ppg_denoise_v1/lite \
        --crm-dirs models/ppg_denoise_v1/lite --tags crm512 --out outputs/b5_fuse

# 2) 心率路径（6 域）：零样本 / AdaBN / TENT / 双导联
python scripts/tools/cross_dataset_eval.py --datasets troika bidmc capnobase wesad wrist dalia --adapt none
python scripts/tools/mae_with_exclusions.py        # 生成最终表（含剔除与 CI）
python scripts/tools/final_tables.py               # 汇总

# 3) baseline 与消融
python scripts/tools/classical_baselines.py        # 17 种经典方法
python scripts/tools/troika_baselines.py           # SpaMA/JOSS 复现
python scripts/tools/robustness_sweep.py           # 伪影强度鲁棒性曲线
python scripts/tools/ref_ablation.py               # 参考通道策略
python scripts/tools/blend_holdout.py              # WESAD 混合（5 折）
python scripts/tools/draw_framework.py             # 框架图
```

---

## 7. 脚本一览（仓库内保留的必要脚本）

| 脚本 | 作用 |
|---|---|
| `scripts/modules/ppg_api.py` | **软件接口**：`PPGDenoiser` 三波形推理 |
| `scripts/tools/run_denoise.py` | 接口命令行（demo / npz / csv / 自检） |
| `scripts/tools/regression_check.py` | 回归门禁：核对全部关键数字 |
| `scripts/pipeline/pipeline.py` | 主流水线（11 步：数据→RLS 标定→特征→训练→评估→HR→汇总） |
| `scripts/tools/train_residual.py` | 残差 U-Net 训练（多消融开关） |
| `scripts/tools/train_crm.py` | 频域 CRM 训练 |
| `scripts/tools/ensemble_cime.py` | CIME 集成评估 |
| `scripts/tools/fuse_branches.py` | 时域×频域融合（α 在验证集选） |
| `scripts/tools/cross_dataset_eval.py` | 6 域心率评估（真值/域适应/双导联/覆盖率） |
| `scripts/tools/classical_baselines.py` | 17 种经典波形 baseline |
| `scripts/tools/troika_baselines.py` | TROIKA 经典方法（SpaMA/JOSS）复现 |
| `scripts/tools/robustness_sweep.py` | 伪影强度（κ）鲁棒性曲线 |
| `scripts/tools/ref_ablation.py` | 参考通道策略消融 |
| `scripts/tools/blend_holdout.py` | WESAD RLS+模型混合（受试者 K 折） |
| `scripts/tools/mae_with_exclusions.py` / `final_tables.py` | 最终 HR 表（剔除/CI/分层） |
| `scripts/tools/selective_pred.py` | 决策层（置信信号选择性预测）空间 |
| `scripts/tools/make_features_alpha.py` / `make_seg_features.py` | α / 段长消融特征 |
| `scripts/tools/draw_framework.py` | 框架图渲染 |
| `scripts/tools/download_datasets.py` | 公开数据集下载 |

---

## 8. 结果摘要

| 任务 | 指标 | 原始 | 最强 baseline | 本方案 |
|---|---|---|---|---|
| CIME 波形 | PRD ↓ / Pearson ↑ | 103.41 / — | 65.71（谱减） | **46.57 / 0.9395** |
| TROIKA 心率 | MAE ↓ | 6.64 | 15.59（SpaMA 复现） | **2.11** |
| BIDMC / CapnoBase | MAE ↓ | 3.00 / 1.06 | — | **1.48 / 1.06** |
| WESAD | MAE ↓ | 9.09 | — | **7.15**（5 折 CV） |
| DaLiA / WRIST | MAE ↓ | 11.66 / 35.91 | — | **10.93 / 16.03** |

适用边界（重要）：**强伪影（κ≫1）与中高强度腕部运动是主要失效区**；八度/谐波锁定被证明
在仅 PPG+IMU 信息下不可修（需第二模态）。详见 **[实验最终报告.md](实验最终报告.md)** §6。

---

## 9. 许可与致谢

- 代码：供本赛题评审与学术复现使用。
- 数据集：版权归各数据集/赛方所有，本仓库**不重新分发**，请遵守各自许可（见 `datasets/README.md`）。
- 上游参考实现：`uEDA-Net`、`Kernel-Adaptive-Filtering`（仅在实验中作对照，未包含在本仓库）。

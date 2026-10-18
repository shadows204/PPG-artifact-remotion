# 数据集说明

> 本仓库**不包含**原始数据集（总体积约 19 GB，且各数据集有各自的许可协议，重新分发可能侵权）。
> 请按下面的官方来源自行下载，并按"放置位置"放到 `datasets/<名称>/` 下即可被脚本自动读取
> （所有路径都通过 `scripts/_common/paths.py` 的 `datasets_root()` 解析，可用环境变量 `PPG_DATASETS` 覆盖）。

| 数据集 | 用途 | 规模 | 官方来源 | 放置位置 |
|---|---|---|---|---|
| **CIME PPG 挑战赛数据** | 主数据集（唯一有**同步干净 PPG 真值**，波形指标 PRD 只能在此算） | 48 受试者（10 训练 / 38 测试），5000/1900 段 × 30 s，100 Hz，通道 `[带噪PPG, ACC_x, Gyro_y]` | 全国生物医学工程创新设计大赛（第十一届）赛道三官方数据，需向赛方获取 | `datasets/CIME_PPG_dataset/{train,test}_all_noise.mat` |
| **IEEE SPC / TROIKA** | 跑步/运动域心率 | 12 训练 + 10 测试记录，125 Hz，ECG+PPG+ACC | <https://github.com/zhilin007/physionet/tree/master/SPC> | `datasets/IEEEPPG_TROIKA/{Training_data,TestData}/` |
| **BIDMC PPG-BP** | 静息临床域心率 | 53 记录，125 Hz，PLETH+II 导 ECG | PhysioNet `bidmc` <https://physionet.org/content/bidmc/> | `datasets/BIDMC/` |
| **CapnoBase** | 术中/麻醉静息域心率 | 42 记录，300 Hz，pleth+ecg+co2 | <https://www.capnobase.org/> | `datasets/CapnoBase/data/`（`.mat` 或 `csv/`） |
| **WESAD** | 腕部日常活动域 | 15 受试者，BVP/ACC/ECG 700 Hz | UCI / <https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/> | `datasets/WESAD/SXX/SXX.pkl` |
| **PPG-DaLiA** | 腕部多活动域（含中高强度） | 15 受试者，BVP 64 Hz、ACC 32 Hz、ECG 700 Hz | UCI `PPG-DaLiA`；本项目使用 Zenodo MATLAB 镜像 | `datasets/PPG_DALIA/*.mat` |
| **WRIST** | 腕部 + 同步 ACC/Gyro（参考通道最优策略在此验证） | 19 记录，256 Hz WFDB，PPG+ACC(3)+Gyro(3) | <https://physionet.org/content/wrist/> | `datasets/WRIST/` |
| **HUAWEI MIDHIGH** | 华为中高强度波形（**无真值**，仅用于伪影强度/漂移诊断） | 91 段 | 厂商/赛方提供 | `datasets/HUAWEI_MIDHIGH/` |


## 可选：从 `datasets` 分支直接获取开放许可子集

本仓库另有一个 **`datasets` 分支**（与 `main` 无共同历史），存放**许可允许再分发**且单文件 < 5 MB 的公开数据集：
**TROIKA（13 MB）+ WRIST（50 MB）+ BIDMC（181 MB）**，共约 244 MB。取用方式（在仓库根目录）：

```bash
# 只取数据文件，不切换分支（推荐）
git fetch origin datasets
git checkout origin/datasets -- datasets/

# 或把数据放到独立目录
git worktree add ../ppg-datasets origin/datasets
```

其余数据集（CIME / CapnoBase / WESAD / PPG-DaLiA / HUAWEI）因**许可限制或单文件超过 GitHub 100 MB 上限**
（WESAD 单个 pkl ≈ 1 GB、DaLiA 单个 .mat 102 MB）无法入库，请按上表官方链接下载。

## 一键下载（自动的部分）

`scripts/tools/download_datasets.py` 覆盖了**公开可直接下载**的数据集（TROIKA / BIDMC / CapnoBase / WRIST / WESAD / PPG-DaLiA）：

```bash
python scripts/tools/download_datasets.py --list          # 看清单
python scripts/tools/download_datasets.py --only troika wrist   # 下指定数据集
```

CIME 与 HUAWEI 数据需向赛方/厂商获取，脚本不提供下载。

## 数据核对

各数据集的采样率、通道顺序、受试者划分在 `datasets/README.md`（本文件）之外的历史文档里有逐条核对；
代码里对应的加载器为 `scripts/modules/ext_loaders.py`（`load_troika / load_bidmc / load_capnobase /
load_wesad / load_wrist / load_dalia / load_huawei`），已处理：采样率统一到 100 Hz（心率路径按需）、
通道顺序、NaN 线性插值、R 峰/真值窗口对齐。

## 最小可运行子集

只想跑通"软件接口"（`scripts/tools/run_denoise.py --demo`）**不需要任何数据集**——内置合成样例可直接运行。
想复现论文中的数字则至少需要：CIME（波形）+ 6 个有真值域（心率）。

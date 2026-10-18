# datasets 分支：可直接使用的开放许可数据集

> 本分支与 `main` **无共同历史**（orphan branch），只存放体积较大但**许可允许再分发**的公开数据集，
> 免去下载与格式转换。共 3 个数据集 / 约 244 MB，所有文件均 < 5 MB（可正常被 git 跟踪）。

## 内容

| 数据集 | 记录 | 用途 | 许可 |
|---|---|---|---|
| `IEEEPPG_TROIKA/` | 12 训练 + 10 测试 | 跑步/运动域心率（通道：ECG/PPG/PPG/ACC，125 Hz） | IEEE SPC 挑战赛数据，学术使用 |
| `WRIST/` | 19 记录 | 腕部 + 同步 ACC(3)+Gyro(3)，参考通道策略验证（256 Hz，WFDB） | PhysioNet（ODC-BY 1.0，见 https://physionet.org/content/wrist/） |
| `BIDMC/` | 53 记录 | 静息临床域心率（PLETH + II 导 ECG，125 Hz，WFDB） | PhysioNet（ODC-BY 1.0，见 https://physionet.org/content/bidmc/） |

**署名**：BIDMC 数据来自 Pimentel 等（IEEE TBME 2017）与 PhysioNet；WRIST 数据来自 Karas 等（IEEE TBME 2021）与 PhysioNet；
TROIKA 数据来自 Zhang 等（IEEE TBME 2015）与 IEEE SPC 2015 挑战赛。使用请遵守各数据集许可并引用原始论文。

## 未包含的数据集（体积或许可不允许）

| 数据集 | 原因 | 获取方式 |
|---|---|---|
| CIME | 赛方数据，不可再分发 | 向全国生物医学工程创新设计大赛赛方获取 |
| CapnoBase | 注册同意的 Data Use Agreement | https://www.capnobase.org/ |
| WESAD | 17 GB，单个 pkl ≈1 GB（超 GitHub 100 MB 限制） | https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/ |
| PPG-DaLiA | 单个 .mat 102 MB（超限） | UCI 数据集 / https://zenodo.org/ |
| HUAWEI_MIDHIGH | 厂商数据 | 向厂商/赛方获取 |

## 如何取用

在主仓库（`main`）根目录下任选一种：

```bash
# 方式 1：只要数据文件（推荐，不切换分支）
git fetch origin datasets
git checkout origin/datasets -- datasets/
#   之后 datasets/ 下就是 IEEEPPG_TROIKA / WRIST / BIDMC，脚本可直接读取

# 方式 2：把数据放到独立目录
git worktree add ../ppg-datasets origin/datasets
#   然后把 ../ppg-datasets/datasets/* 移入主仓库 datasets/  （或设 PPG_DATASETS=../ppg-datasets/datasets）

# 方式 3：只下载该分支（不要代码）
git clone --single-branch --branch datasets git@github.com:shadows204/PPG-artifact-remotion.git ppg-datasets
```

其余数据集请按主仓库 `datasets/README.md` 的官方链接下载。

> 注意：本分支只含数据文件与说明；代码、模型、报告都在 `main` 分支。

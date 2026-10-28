# PPG运动伪影去除

单通道PPG在运动时会被伪影淹掉。跑步、骑行时步频大概1.5–3Hz，和心率带（0.7–3Hz）几乎重合，带通和陷波拿它没办法。这个仓库是我们在第十一届全国生物医学工程创新设计大赛（赛道三）的作品：输入一段PPG，可选带IMU，输出去伪影后的波形和逐窗心率。

先说结果。

- CIME测试集1900段：波形误差PRD从103.41降到46.57，Pearson0.9395，SNR7.78dB。作为对照，16种经典方法里最好的谱减法是65.71，端到端深度模型最好49.54。
- 6个公开数据集的心率MAE：TROIKA2.11、BIDMC1.48、CapnoBase1.06、WESAD7.15、PPG-DaLiA10.93、WRIST16.03bpm。这六个数据集没有干净波形真值，只评心率。
- CPU上30s信号推理0.2s，lite模型15MB。

方法上就一句话：先用RLS从IMU里估出伪影，得到物理先验；网络只学先验还差多少；跨设备时靠决策层（Viterbi、双导联选路、弃权）兜底。为什么这么设计、做了哪些消融、失效边界在哪，都在[实验最终报告.md](实验最终报告.md)里，框架图见[figures/pipeline_framework.png](figures/pipeline_framework.png)。

## 分支

- `main`：算法、训练与评估脚本、模型权重、实验报告。
- `app`：软件部分，Web工作台和单文件exe（不用装Python），有自己的README。
- `datasets`：可以再分发的公开数据子集（TROIKA+WRIST+BIDMC，约244MB），和main没有共同历史。

```bash
git clone -b app  git@github.com:shadows204/PPG-artifact-remotion.git ppg-app
git clone -b main git@github.com:shadows204/PPG-artifact-remotion.git ppg-main
git fetch origin datasets && git checkout origin/datasets -- datasets/   # 在ppg-main里取数据
```

## 目录

```
.
├── 实验最终报告.md          # 方法、实验、消融、适用边界
├── scripts/
│   ├── modules/             # ppg_api接口、网络、RLS、数据加载
│   ├── pipeline/            # 主流水线
│   └── tools/               # 训练、评估、baseline、回归门禁
├── models/ppg_denoise_v1/   # lite / full / no_imu三套权重，manifest里带sha256
├── datasets/                # 只有说明，数据自己下
├── figures/
└── outputs/                 # 跑出来的东西放这里，仓库里是空的
```

## 跑起来

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

最快的验证方式是不需要任何数据集，用内置合成样例跑一遍接口：

```bash
python scripts/tools/run_denoise.py --demo --out outputs/denoise_demo
```

也可以直接调接口：

```python
import sys; sys.path.insert(0, "scripts/modules")
from ppg_api import PPGDenoiser

d = PPGDenoiser(preset="lite")                 # 另外还有full / no_imu
r = d.denoise(ppg, fs=100, acc=acc, gyro=gyro) # IMU可以不给
r.ppg_clean                                    # 滤净波形
r.artifact                                     # 估计出的伪影，无IMU时全0
r.hr_median                                    # 心率中位，可能为None
```

输入可以是npz、csv或json。npz用`ppg`（必需）、`acc`、`gyro`、`reference`、`fs`这几个键；csv用`ppg, acc_x/y/z, gyro_x/y/z, reference`列名。采样率任意，内部统一重采样到100Hz；长度也任意，超过30s会滑窗重叠相加，短于1s直接报错；NaN会自动插值。

数据齐了之后可以跑回归门禁，一条命令核对报告里的关键数字：

```bash
python scripts/tools/regression_check.py
```

## 模型

`models/ppg_denoise_v1/`下有3个预设。`lite`是5个残差模型加3个频域模型，15MB，软件里默认用它；`full`是41加3个成员，108MB，用来复现报告里46.57那个数；`no_imu`是3个单通道模型，没有IMU时自动切换。每个权重的sha256记录在`manifest.json`里。

## 数据

原始数据不随仓库分发：总体积约19GB，而且各有各的许可。官方下载地址和放置目录都写在[datasets/README.md](datasets/README.md)。CIME是唯一有同步干净PPG真值的数据集，波形指标只能在这里算；其余6个公开数据集只有ECG心率参考。

## 复现实验

训练和评估比较耗时，不是必须跑的。注意大部分脚本依赖先跑过`pipeline.py`生成的中间特征。

```bash
python scripts/pipeline/pipeline.py all                # 主流水线
python scripts/tools/train_residual.py --help          # 单模型训练
python scripts/tools/train_crm.py --help               # 频域分支训练
python scripts/tools/classical_baselines.py            # 经典方法对照
python scripts/tools/cross_dataset_eval.py --help      # 六域心率评估
```

产物都写进`outputs/`，目录说明见[outputs/README.md](outputs/README.md)。

## 已知限制

- 波形质量只能在CIME上定量评，其他数据集没有干净真值，所以只报心率。
- 中高强度腕部运动是明显短板。误差主要来自谐波与八度锁定：步频谐波落在真实心率附近，光靠PPG和IMU的频谱分不开，需要多波长之类的第二模态。
- 伪影特别强（κ远大于1）时，我们的结果不如谱减法。这是有界残差方案的代价，报告里给了完整的强度扫描曲线。
- 没有IMU时拿不到伪影估计，只能给滤净波形和心率。
- 这是研究原型，不是医疗器械，不能用于诊断。

## 说明

代码供赛题评审和学术复现使用。数据集版权归各自作者与赛方所有，本仓库不重新分发。Web工作台和exe的构建方法见`app`分支的README。

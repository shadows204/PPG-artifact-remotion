# PPG 运动伪影去除 · Web 工作台（软件源码分支 `app`）

> **分支定位：本分支只放"可运行的软件"**——网页工作台、v1/v2 两个版本、单文件 exe 打包配置与构建工作流。
> 算法细节、实验报告、训练脚本与数据集说明在 **`main` 分支**；开放许可数据集子集在 **`datasets` 分支**。

---

## 1. 两个版本

| 版本 | 源码目录 | 界面/演示特点 | exe 产物 |
|---|---|---|---|
| **v2（推荐）** | `webapp/` | 演示数据为**真实 PPG 形态 + 物理幅度 IMU**；演示态显示「链路自检」而非 PRD；静息场景演示旁路门控 | `PPG-Workbench-v2.exe` |
| **v1（原始版）** | `webapp_v1/` | 初版行为：合成正弦演示、演示态显示 PRD（用于对照） | `PPG-Workbench-v1.exe` |

> 算法与指标完全一致（都调用同一套 `scripts/modules/ppg_api.py` + 模型）；差异只在**演示数据生成**与**演示态指标展示**。

---

## 2. 目录结构

```
app/
├── README.md                        # 本文件
├── webapp/                          # ★ v2 软件源码
│   ├── server.py                    #   Flask 后端（静态页 + 推理 API + 下载 + 退出接口）
│   ├── static/index.html            #   单文件前端（HTML/CSS/JS，无 CDN）
│   ├── 使用说明书.md                 #   功能/接口/故障排查说明
│   ├── start.sh                     #   源码启动脚本
│   └── packaging/                   #   ★ 打包配置
│       ├── ppg_workbench_onefile.spec   # onefile + 无控制台（VARIANT=v1|v2）
│       ├── ppg_workbench.spec           # （可选）onedir 模式
│       ├── build_windows.bat            # Windows 一键构建 v1+v2
│       ├── build_linux.sh               # Linux/macOS 构建（用于验证冻结逻辑）
│       ├── gen_test_npz.py              # 冒烟测试用合成输入
│       └── README.md                    # 打包说明与常见问题
├── webapp_v1/                       # v1 原始版源码（含其 使用说明书.md）
├── scripts/
│   ├── modules/ppg_api.py           # 软件接口 PPGDenoiser（三波形 + 心率）
│   ├── modules/residual.py crm.py models.py   # 网络结构
│   ├── pipeline/pipeline.py         # ppg_api 依赖：RLS/PSD/Viterbi 等实现
│   ├── _common/paths.py             # 路径解析
│   └── tools/run_denoise.py         # 命令行版接口（demo/npz/csv 自检）
├── models/ppg_denoise_v1/           # 权重（manifest.json 含 sha256）
│   ├── lite/    5 残差 + 3 CRM（15 MB，PRD 46.37，推荐）
│   ├── full/    41 残差 + 3 CRM（108 MB，PRD 46.57）
│   └── no_imu/  无 IMU 退化路径（3 个 1 通道模型）
├── .github/workflows/               # 构建（Artifacts）+ tag 发布（Release）
├── requirements.txt                 # numpy scipy torch matplotlib flask (+wfdb/PyEMD 可选)
├── run_demo.sh                      # 一键：装依赖 → 命令行接口 demo
├── figures/pipeline_framework.png   # 框架图（来自 main 分支）
└── outputs/                         # 运行产物（分支内为空）
```

---

## 3. 源码方式启动（最快）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 方式 A：脚本
OMP_NUM_THREADS=2 bash webapp/start.sh           # → http://127.0.0.1:8000

# 方式 B：直接跑（指定解释器，不必 activate）
OMP_NUM_THREADS=2 python webapp/server.py --port 8000
```

- 打包版会自动打开浏览器；源码版默认也自动打开（`--no-browser` 可关）。
- 首次请求才加载模型（lite 8 个权重，约 1–3 s），之后 30 s 信号约 **0.1–0.3 s**。
- **务必设 `OMP_NUM_THREADS=2`**（或直接跑 exe，它已内置线程限制）：128 核机器上 torch 默认开 128 线程会把推理拖慢到 ~5 s。

---

## 4. 单文件 exe（双击即用，无需 Python）

### 4.1 产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `PPG-Workbench-v1.exe` | ≈ 294 MB | 原始版；**单文件**、无控制台窗口 |
| `PPG-Workbench-v2.exe` | ≈ 294 MB | 增强版；同上 |

双击 → 自动打开浏览器（默认 `http://127.0.0.1:8000`，端口被占自动顺延）。
**退出**：网页右上角 **⏻**（无控制台时没有 Ctrl+C）；日志在 exe 同目录 `workbench.log`。
**拷贝**：只拷这一个 exe 即可；首次启动解压约 20–60 s 属正常。

### 4.2 构建（三选一）

**A. GitHub Actions（推荐，不需要 Windows 机器）**

- 本分支已带两个工作流：
  - `.github/workflows/build-windows-exe.yml`：push 到 `main`/`app` 或手动 Run → 产出两个 Artifacts（`PPG-Workbench-v1-windows.zip` / `-v2-windows.zip`）；
  - `.github/workflows/release-windows.yml`：**推 `v*` tag** → 构建 v1/v2 → 做冒烟测试 → **自动创建 Release** 并附两个 zip + `SHA256SUMS.txt`。
- 发布一条（**tag 必须打在 `app` 分支上**，release 工作流随本分支提供）：
  `git checkout app && git tag v2.0.0 && git push origin v2.0.0`

**B. Windows 本机**

双击 `webapp\packaging\build_windows.bat`（本机只需装 Python 用于构建；成品不需要）→ `dist\PPG-Workbench-v1.exe`、`-v2.exe`。

**C. Linux/macOS（验证冻结逻辑）**

```bash
bash webapp/packaging/build_linux.sh "v1 v2"
```

> 跨平台限制：PyInstaller 不能交叉编译，Windows 的 exe 必须在 Windows（或 Actions 的 windows-latest）上构建。

---

## 5. 网页功能

- **输入**：内置演示数据（三档）/ 上传 `.npz` `.csv` `.json`；
- **输出**：三波形 ① 原波形 ② 伪影 M̂ ③ 滤净波形（+ 物理先验）；**心率曲线**（8 s 窗 / 2 s 步进，Viterbi 全局路径）；质量指标（伪影占比/旁路/SNR/模式/耗时）；
- **交互**：滚轮缩放、拖拽平移、双击复位、悬停读数、图例开关、明暗主题、历史记录（localStorage）；
- **导出**：NPZ / JSON / CSV / PNG；
- **接口**：见 §6。

---

## 6. REST API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 服务/版本/torch 版本/已加载模型 |
| GET | `/api/info` | manifest、预设、限制、演示场景、报告数字 |
| GET | `/api/demo?scenario=rest\|moderate\|strong` | 预览演示数据（不推理） |
| POST | `/api/denoise` | 主接口（multipart 或 JSON） |
| POST | `/api/selfcheck` | 合成样例端到端自检（返回链路自检项） |
| GET | `/api/run/<run_id>` `/api/runs` | 读取/列出服务端缓存结果（≤12 条） |
| GET | `/api/download/<run_id>/{npz,json,csv,png}` | 下载产物 |
| POST | `/api/shutdown` | 退出服务（无控制台版的退出方式） |

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/api/denoise \
  -F source=demo -F scenario=moderate -F preset=lite -F want_hr=1

curl -X POST http://127.0.0.1:8000/api/denoise \
  -F source=file -F file=@my.npz -F preset=lite -F want_hr=1
```

返回关键字段：`result.ppg_raw/artifact/ppg_prior/ppg_clean`、`result.hr_bpm/hr_t/hr_median`、`result.quality`、`result.meta`、`series`（降采样绘图数据）、`reference_metrics`（有参考时）、`link_check`（演示态链路自检）。

---

## 7. 输入格式

**NPZ（推荐）**

```python
np.savez_compressed("my.npz",
    ppg=ppg,          # 必需，(N,) 或 (n_lead, N)，任意单位
    acc=acc,          # 可选，(3,N)/(N,3)/(N,)
    gyro=gyro,        # 可选，有则用 "ACC_x + Gyro_y" 异质两路参考（实测更优）
    reference=clean,  # 可选，干净 PPG 真值（也可用键 clean）→ 显示 PRD/Pearson/SNR
    fs=100.0,         # 可选标量（缺省时须在界面填写）
)
```

**CSV**：列名 `ppg[, acc_x/y/z, gyro_x/y/z, reference]`；**CSV 不含采样率，必须在界面左侧填 fs**。

**JSON**：`{"ppg": [...], "acc": [[...]], "gyro": [[...]], "reference": [...], "fs": 100}`。

约束：上传 ≤ 64 MB、输入 ≤ 2,000,000 点；任意长度（>30 s 滑窗重叠相加）、任意采样率（内部重采样到 100 Hz）；NaN 自动插值。

---

## 8. 打包实现要点（为什么能单文件 + 无黑框）

| 点 | 做法 |
|---|---|
| 单文件 | PyInstaller `onefile`；模型/网页/`pipeline.py` 作为 data 打进 exe，运行时解压到 `sys._MEIPASS` |
| 无控制台 | spec `console=False`；启动即把 `stdout/stderr` 重定向到 exe 同目录 `workbench.log` |
| 退出 | 无 Ctrl+C → `POST /api/shutdown` + 网页右上角 **⏻** 按钮 |
| 自动开浏览器 | 打包版默认打开默认浏览器；`--no-browser` 可关 |
| 性能 | 启动时 `torch.set_num_threads(min(4, cpu//16))`：128 核上 4.9 s → 0.1 s |
| 端口占用 | 8000 起自动顺延到 8019 |
| 体积 | 构建用 **CPU 版 torch**（`--index-url .../whl/cpu`），避免把 CUDA 打进包 |

详细说明与常见问题见 [`webapp/packaging/README.md`](webapp/packaging/README.md)。

---

## 9. 与其他分支的关系

| 分支 | 内容 | 本分支如何依赖 |
|---|---|---|
| **`main`** | 算法主流程、训练/评估脚本、实验报告、数据集说明、**模型权重源** | 本分支的 `models/`、`scripts/` 关键文件即从 main 同步 |
| **`datasets`** | 开放许可数据集子集（TROIKA/WRIST/BIDMC） | 可选：`git fetch origin datasets && git checkout origin/datasets -- datasets/` |
| `app`（本分支） | 软件源码 + 打包配置 + 构建/发布工作流 | — |

> 算法贡献、消融、baseline、适用范围等请阅读 main 分支的 **`实验最终报告.md`**。

---

## 10. 许可与致谢

- 代码：供本赛题评审与学术复现使用。
- 数据集：版权归各数据集/赛方所有，不随本分支分发。
- 上游参考实现：`uEDA-Net`、`Kernel-Adaptive-Filtering`（仅 main 分支实验中作对照）。

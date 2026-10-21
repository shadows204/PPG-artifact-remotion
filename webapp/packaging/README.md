# 打包成免安装软件（**单文件 exe + 无控制台窗口**）

目标：**双击即用、不依赖 Python、演示数据内置、可导入外部数据**；产物只有**一个 .exe 文件**，
运行时不弹黑色命令行窗口，自动打开浏览器；退出用网页右上角 **⏻** 按钮。

## 1. 成品

| 产物 | 内容 | 说明 |
|---|---|---|
| **`PPG-Workbench-v1.exe`** | 原始版（v1.0） | 演示数据为合成正弦、演示态显示 PRD；用于对照 |
| **`PPG-Workbench-v2.exe`** | 增强版（v2.0） | 真实 PPG 形态演示 + 演示态「链路自检」（不显示 PRD）+ 静息旁路演示 |

- **单文件**：内含 Python 运行时、torch(CPU)、numpy/scipy/matplotlib、模型（lite + no_imu）、网页与算法；
  拷贝**这一个 exe** 即可运行。体积约 **450–700 MB**（zip 后约 300–450 MB）。
- **无控制台**：`console=False`。日志写入 exe 同目录 `workbench.log`；退出方式见下。
- **首启较慢**：onefile 每次启动会先解压到临时目录，约 **20–60 s**（机器越快越短），之后打开浏览器。

## 2. 启动后怎么用

1. 双击 exe → 自动打开浏览器（默认 `http://127.0.0.1:8000`，端口被占自动顺延）；
2. 页面里：选「演示数据」或「上传文件」→ 选模型 → 开始处理；
3. **退出应用**：点右上角 **⏻** 按钮（调用 `POST /api/shutdown`），进程随即结束；也可在任务管理器结束 exe。

## 3. 三种构建方式

### 3.1 GitHub Actions（推荐，不需要 Windows 机器）

仓库工作流 [`.github/workflows/build-windows-exe.yml`](../../.github/workflows/build-windows-exe.yml)
用 **matrix 同时构建 v1 与 v2**（onefile + 无控制台 + 冒烟测试 + 退出接口测试）：

1. 仓库 → **Actions** → `build-windows-exe` → **Run workflow**（可选 `bundle_full`）；
2. push 改动 `webapp*/ scripts/ models/` 时也会自动触发；
3. 约 15–30 分钟，在运行页底部 **Artifacts** 下载：
   - `PPG-Workbench-v1-windows.zip`
   - `PPG-Workbench-v2-windows.zip`

> Linux/macOS 无法交叉编译 Windows 程序（PyInstaller 硬限制），故 Windows exe 必须在 Windows runner 上构建。

### 3.2 Windows 本机一键构建

双击 `webapp\packaging\build_windows.bat`（本机只需装 Python 用于**构建**；成品不需要）：

```
dist\PPG-Workbench-v1.exe
dist\PPG-Workbench-v2.exe
```

变体：`set BUNDLE_FULL=1` 后再运行 → 额外打包 full 预设（+109 MB）。

### 3.3 Linux / macOS 本地构建（用于验证冻结逻辑）

```bash
bash webapp/packaging/build_linux.sh "v1 v2"      # 产物 dist/PPG-Workbench-v1、-v2
```

## 4. 关键实现（为什么能"单文件 + 无黑框"）

| 点 | 做法 |
|---|---|
| 单文件 | PyInstaller `onefile`；资源（模型/网页/pipeline）作为 data 打进 exe，运行时在 `sys._MEIPASS` |
| 无控制台 | spec 里 `console=False`；启动即把 `stdout/stderr` 重定向到 exe 同目录 `workbench.log`（否则 windowed 模式下 print 会崩） |
| 退出 | 无 Ctrl+C → 新增 `POST /api/shutdown` + 网页 **⏻** 按钮（v1/v2 都加） |
| 自动开浏览器 | 打包版默认打开默认浏览器；`--no-browser` 可关闭（CI 冒烟测试用） |
| 端口占用 | 8000 起自动顺延到 8019 |
| 性能 | 启动时 `torch.set_num_threads(min(4, cpu//16))`：128 核机器上推理 4.9 s → 0.1 s |
| 依赖瘦身 | 构建环境装 **CPU 版 torch**（`--index-url .../whl/cpu`），避免把 CUDA 打进包 |

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| 双击后等很久才出浏览器 | onefile 首启解压，20–60 s 属正常；看 exe 同目录 `workbench.log` |
| Windows SmartScreen「未知发布者」 | 未签名程序提示 → 更多信息 → 仍要运行（对内演示可接受；对外分发建议代码签名） |
| 杀毒软件误报 | PyInstaller 单文件常见 → 加白名单，或改用 onedir 版 |
| 想关掉服务 | 点网页右上角 **⏻**；或任务管理器结束 exe |
| 端口被占用 | 自动顺延；也可 `PPG-Workbench-v2.exe --port 8100` |
| 想带 full 预设 | 构建时 `BUNDLE_FULL=1`（+109 MB） |
| v1 与 v2 区别 | v1 = 原版（合成正弦演示、显示 PRD）；v2 = 真实形态演示 + 链路自检 + 静息旁路演示 |

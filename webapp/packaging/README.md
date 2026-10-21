# 打包成免安装软件（Windows .exe / Linux 二进制）

目标：**双击即用、不依赖 Python 环境、演示数据内置、可导入外部数据**。

## 0. 版本

当前打包 **v2.0**（演示数据升级 + 演示态显示「链路自检」而非 PRD）。
v1 原版已备份：本地 `webapp_原始版_v1备份_20261021/`、git tag `webapp-v1-original`。

## 1. 成品长什么样

| 形态 | 文件 | 启动 | 体积（参考） | 说明 |
|---|---|---|---|---|
| **onedir（推荐）** | `PPG-Workbench/` 文件夹（内含 `PPG-Workbench.exe`） | 双击 exe，**2–5 s** 打开网页 | ~450–700 MB | 启动快；发给别人请打包成 zip |
| onefile | 单个 `PPG-Workbench.exe` | 双击，**20–60 s**（每次先解包） | ~450–700 MB | 只有一个文件，但启动慢、杀软误报更多 |

运行后行为：
1. 弹出控制台窗口，打印 `http://127.0.0.1:8000`（端口被占自动顺延）；
2. **自动打开默认浏览器**；
3. 网页里即可用「内置演示数据」或上传 `.npz/.csv/.json`；
4. 关闭控制台窗口（或 Ctrl+C）即退出；日志写入 exe 同目录 `workbench.log`。

## 2. 内置了什么

| 内容 | 说明 |
|---|---|
| 算法与接口 | `scripts/modules/ppg_api.py` + `residual/crm/models` + `scripts/pipeline/pipeline.py` |
| 网页 | `webapp/static/index.html`（单文件，无 CDN） |
| 模型权重 | `models/ppg_denoise_v1/lite`（5 残差 + 3 CRM）+ `no_imu`；可用 `BUNDLE_FULL=1` 再带 `full`（+109 MB） |
| 演示数据 | **代码合成**（静息/中等运动/强运动三档，见 `synth_demo()`），无需任何外部数据文件 |
| 说明书 | `webapp/使用说明书.md` |

> 演示数据是**合成信号**（UI 与报告都会提示"PRD 不代表真实性能"）。真实数据请通过网页上传导入。

## 3. 三种构建方式

### 3.1 GitHub Actions（推荐，不需要 Windows 机器，也不需要本地 Python）

仓库已带工作流 [`.github/workflows/build-windows-exe.yml`](../../.github/workflows/build-windows-exe.yml)：

1. 打开仓库 → **Actions** → 左侧 **build-windows-exe** → **Run workflow**；
2. 可选参数：`mode`（`onedir` 推荐 / `onefile`）、`bundle_full`（是否带 full 预设）；
3. 约 10–20 分钟后，在该次运行的 **Artifacts** 里下载 `PPG-Workbench-Windows-onedir.zip`。

> Linux/macOS 无法交叉编译 Windows 程序（PyInstaller 的硬限制），所以 Windows exe 必须在 Windows
> runner 上构建——这就是用 Actions 的原因。

### 3.2 Windows 本机一键构建

需要本机装 Python 3.10+（**只是构建需要**，成品 exe 不需要）：

```
双击 webapp\packaging\build_windows.bat
```

产物：`dist\PPG-Workbench\PPG-Workbench.exe` 与 `dist\PPG-Workbench-Windows.zip`。
变体：`set BUILD_MODE=onefile`（单文件）/ `set BUNDLE_FULL=1`（带 full 预设）后重新运行。

### 3.3 Linux / macOS 本地构建（用于验证冻结逻辑）

```bash
bash webapp/packaging/build_linux.sh onedir     # 或 onefile
# 产物：dist/PPG-Workbench/PPG-Workbench
```

## 4. 关键实现细节（为什么能"免环境"）

1. **冻结路径**：`server.py` 检测 `sys.frozen`，资源根取 `sys._MEIPASS`（onefile）或 exe 同级；
   模型/静态页/说明书/`pipeline.py` 全部作为 data 打进包。
2. **pipeline 按路径加载**：`ppg_api._pipeline()` 支持 `PPG_PIPELINE_PY` 环境变量（打包版由 server 注入）。
3. **线程限制**：启动时 `torch.set_num_threads(min(4, cpu//16))` —— 小模型在 128 核机器上
   默认 128 线程会被同步开销拖垮（实测 4.9 s → 0.1 s）。
4. **端口自适应**：8000 被占自动顺延到 8001…8019。
5. **自动开浏览器**：`--no-browser` 可关闭。
6. **CPU 版 torch**：构建环境用 `pip install torch --index-url .../cpu`，避免把 CUDA 库打进包（否则 +1.5 GB）。

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| 双击后没反应/浏览器没弹 | 看控制台窗口是否报错；看同目录 `workbench.log`；或手动打开打印出的地址 |
| Windows SmartScreen「未知发布者」 | 未签名程序的正常提示 → 更多信息 → 仍要运行（对内演示可接受；对外分发建议代码签名） |
| 杀毒软件误报 | PyInstaller 单文件常见 → 改用 onedir 版，或把文件夹加入白名单 |
| 首次处理慢（几秒） | 首次调用才加载模型（lite 8 个权重）→ 之后 0.1–0.3 s |
| 端口被占用 | 程序自动顺延；也可 `PPG-Workbench.exe --port 8100` |
| 想换模型 | 重新构建时 `set BUNDLE_FULL=1`；或把权重放到 exe 同级 `models/ppg_denoise_v1/<preset>/`（onedir 版会优先用包内资源） |

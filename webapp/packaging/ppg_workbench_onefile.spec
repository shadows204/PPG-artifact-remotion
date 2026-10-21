# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：**单文件（onefile）+ 无控制台窗口**

用法（在仓库根目录）：
    VARIANT=v2 pyinstaller --noconfirm --clean webapp/packaging/ppg_workbench_onefile.spec
    VARIANT=v1 pyinstaller --noconfirm --clean webapp/packaging/ppg_workbench_onefile.spec

环境变量：
    VARIANT=v1|v2     默认 v2。v1 = 原始版（webapp_v1/），v2 = 增强版（webapp/）
    BUNDLE_FULL=1     额外打包 full 预设（+109 MB；默认只带 lite + no_imu）

产物：
    dist/PPG-Workbench-v1.exe   （Windows）或 dist/PPG-Workbench-v1（Linux）
    dist/PPG-Workbench-v2.exe

特点：
- **单文件**：exe 内含 Python 运行时、torch、模型、网页与算法；拷贝一个文件即可运行。
- **无控制台窗口**：console=False；日志写入 exe 同目录 workbench.log；
  退出方式 = 网页右上角「⏻ 退出应用」按钮（调用 POST /api/shutdown）。
- 首次启动会把内容解压到临时目录（约 20–60 s），之后同样需要解压（onefile 特性）。
"""
import os
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent.parent                 # noqa: F821  (PyInstaller 注入)
VARIANT = os.environ.get("VARIANT", "v2").strip().lower()
BUNDLE_FULL = os.environ.get("BUNDLE_FULL", "0") == "1"
APP_NAME = "PPG-Workbench-%s" % VARIANT

if VARIANT == "v1":
    SRC = ROOT / "webapp_v1"
else:
    VARIANT = "v2"
    SRC = ROOT / "webapp"

ENTRY = SRC / "server.py"
assert ENTRY.exists(), "找不到入口脚本：%s" % ENTRY

# ---------------------------------------------------------------------------
# 随包分发（注意：冻结后 server.py 以 _MEIPASS/webapp 为 BASE，故统一映射到 webapp/）
# ---------------------------------------------------------------------------
datas = [
    (str(SRC / "static"), "webapp/static"),
    (str(SRC / "使用说明书.md"), "webapp"),
    (str(ROOT / "scripts" / "pipeline" / "pipeline.py"), "scripts/pipeline"),
    (str(ROOT / "scripts" / "_common"), "scripts/_common"),
    (str(ROOT / "models" / "ppg_denoise_v1" / "manifest.json"), "models/ppg_denoise_v1"),
    (str(ROOT / "models" / "ppg_denoise_v1" / "lite"), "models/ppg_denoise_v1/lite"),
    (str(ROOT / "models" / "ppg_denoise_v1" / "no_imu"), "models/ppg_denoise_v1/no_imu"),
]
if BUNDLE_FULL:
    datas.append((str(ROOT / "models" / "ppg_denoise_v1" / "full"),
                  "models/ppg_denoise_v1/full"))

hiddenimports = [
    "ppg_api", "residual", "crm", "models",
    "matplotlib.backends.backend_agg",
    "scipy.signal", "scipy.io", "scipy.ndimage",
]

excludes = [
    "torchvision", "torchaudio", "torchtext", "onnx", "onnxruntime",
    "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "IPython", "jupyter", "notebook", "nbconvert", "pytest", "sphinx",
    "pandas", "polars", "pyarrow", "sympy", "sklearn", "statsmodels",
    "cv2", "wx",
]

a = Analysis(                                                # noqa: F821
    [str(ENTRY)],
    pathex=[str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "modules"), str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)                                            # noqa: F821

# onefile：所有 binaries/datas 打进单个可执行文件；console=False → 不弹黑框
exe = EXE(                                                   # noqa: F821
    pyz, a.scripts, a.binaries, a.datas, [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：PPG 运动伪影去除 · Web 工作台

用法（在仓库根目录）：
    pyinstaller --noconfirm --clean webapp/packaging/ppg_workbench.spec

环境变量：
    BUILD_MODE=onedir|onefile   默认 onedir（推荐：启动快；onefile 单文件但每次解包慢）
    BUNDLE_FULL=1               额外打包 full 预设（+109 MB；默认只带 lite + no_imu）

产物：
    dist/PPG-Workbench/PPG-Workbench(.exe)    ← onedir：双击即可运行
    dist/PPG-Workbench(.exe)                  ← onefile：单文件

说明：
- 资源全部内置：网页静态文件、说明书、pipeline.py、训练好的权重（models/ppg_denoise_v1）。
- 演示数据由代码合成（`synth_demo()`），不依赖外部文件；外部数据通过网页上传导入。
- 依赖 CPU 版 torch（构建环境用 `pip install torch --index-url .../cpu`），避免把 CUDA 打进包里。
"""
import os
from pathlib import Path

# SPECPATH = <repo>/webapp/packaging
ROOT = Path(SPECPATH).resolve().parent.parent                 # noqa: F821  (PyInstaller 注入)
MODE = os.environ.get("BUILD_MODE", "onedir").lower()
BUNDLE_FULL = os.environ.get("BUNDLE_FULL", "0") == "1"
APP_NAME = "PPG-Workbench"

# ---------------------------------------------------------------------------
# 需要随包分发的数据（全部为仓库内相对路径）
# ---------------------------------------------------------------------------
datas = [
    (str(ROOT / "webapp" / "static"), "webapp/static"),
    (str(ROOT / "webapp" / "使用说明书.md"), "webapp"),
    (str(ROOT / "scripts" / "pipeline" / "pipeline.py"), "scripts/pipeline"),
    (str(ROOT / "scripts" / "_common"), "scripts/_common"),
    (str(ROOT / "models" / "ppg_denoise_v1" / "manifest.json"), "models/ppg_denoise_v1"),
    (str(ROOT / "models" / "ppg_denoise_v1" / "lite"), "models/ppg_denoise_v1/lite"),
    (str(ROOT / "models" / "ppg_denoise_v1" / "no_imu"), "models/ppg_denoise_v1/no_imu"),
]
if BUNDLE_FULL:
    datas.append((str(ROOT / "models" / "ppg_denoise_v1" / "full"),
                  "models/ppg_denoise_v1/full"))

# 代码模块：server.py 会 import ppg_api / residual / crm / models
hiddenimports = [
    "ppg_api", "residual", "crm", "models",
    "matplotlib.backends.backend_agg",
    "scipy.signal", "scipy.io", "scipy.ndimage",
]

# 明显用不到、且体积大的模块（CPU 版 torch 本身已不含 CUDA 库）
excludes = [
    "torchvision", "torchaudio", "torchtext", "onnx", "onnxruntime",
    "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "IPython", "jupyter", "notebook", "nbconvert", "pytest", "sphinx",
    "pandas", "polars", "pyarrow", "sympy", "sklearn", "statsmodels",
    "cv2", "PIL.ImageQt", "wx",
]

a = Analysis(                                          # noqa: F821
    [str(ROOT / "webapp" / "server.py")],
    pathex=[str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "modules"),
            str(ROOT / "webapp")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)                                      # noqa: F821

if MODE == "onefile":
    exe = EXE(                                          # noqa: F821
        pyz, a.scripts, a.binaries, a.datas, [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,          # 保留控制台窗口：显示地址/日志，关闭窗口即退出
        disable_windowed_traceback=False,
    )
else:
    exe = EXE(                                          # noqa: F821
        pyz, a.scripts, [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
    )
    coll = COLLECT(                                     # noqa: F821
        exe, a.binaries, a.datas, [],
        strip=False, upx=False, name=APP_NAME,
    )

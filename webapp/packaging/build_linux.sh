#!/usr/bin/env bash
# ===========================================================================
#  PPG 运动伪影去除 · Web 工作台 —— Linux / macOS 本地打包（用于验证冻结逻辑）
#  用法：bash webapp/packaging/build_linux.sh [onedir|onefile]
#  产物：dist/PPG-Workbench/PPG-Workbench（onedir）或 dist/PPG-Workbench（onefile）
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

MODE="${1:-${BUILD_MODE:-onedir}}"
export BUILD_MODE="$MODE"
VENV="${BUILD_VENV:-.venv-build}"
PY="${PYTHON:-python3}"

echo "==> 构建模式：$BUILD_MODE  ｜ 虚拟环境：$VENV"
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip -q
python -c "import torch, flask" 2>/dev/null || {
  echo "==> 安装依赖（CPU 版 torch，约 3-5 分钟）"
  pip install -q numpy scipy matplotlib flask pyinstaller
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
}
pip install -q pyinstaller

echo "==> PyInstaller 打包（约 3-10 分钟）"
pyinstaller --noconfirm --clean webapp/packaging/ppg_workbench.spec

if [ "$BUILD_MODE" = "onefile" ]; then
  SRC="dist/PPG-Workbench"
else
  SRC="dist/PPG-Workbench"
fi
echo "==> 产物：$SRC"
du -sh "$SRC" 2>/dev/null || true

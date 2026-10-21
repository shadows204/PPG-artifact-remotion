#!/usr/bin/env bash
# ===========================================================================
#  PPG 运动伪影去除 · Web 工作台 —— 本地打包（单文件 onefile + 无控制台）
#  用法：bash webapp/packaging/build_linux.sh [v1|v2|"v1 v2"]   （默认 v2）
#  产物：dist/PPG-Workbench-v1 / dist/PPG-Workbench-v2（Linux 对应无扩展名）
# ===========================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

VARIANTS="${1:-${VARIANT:-v2}}"
VENV="${BUILD_VENV:-.venv-build}"
PY="${PYTHON:-python3}"

echo "==> 变体：$VARIANTS ｜ 模式：onefile（单文件，无控制台）｜ 虚拟环境：$VENV"
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

for V in $VARIANTS; do
  echo "==> PyInstaller 打包 $V（onefile，约 3-10 分钟）"
  VARIANT="$V" pyinstaller --noconfirm --clean webapp/packaging/ppg_workbench_onefile.spec
done

echo "==> 产物："
ls -lh dist/ | grep -E "PPG-Workbench-(v1|v2)" || true

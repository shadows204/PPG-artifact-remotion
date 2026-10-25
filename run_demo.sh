#!/usr/bin/env bash
# 一键：建虚拟环境 → 安装依赖 → 跑软件接口 demo（无需数据集）→ 出三波形图
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}

if [ ! -d .venv ]; then
  echo "[demo] 创建虚拟环境 .venv ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[demo] 安装依赖 ..."
pip install -q -r requirements.txt

echo "[demo] 运行接口 demo（内置合成样例；也可用 --input 传自己的 npz/csv）"
python scripts/tools/run_denoise.py --demo --out outputs/denoise_demo

echo
echo "[demo] 完成：outputs/denoise_demo/"
echo "        - result.npz / result.json   （三波形数值）"
echo "        - waveforms.png              （① 原波形 ② 伪影波形 ③ 滤净波形）"

#!/usr/bin/env bash
# PPG 运动伪影去除 · Web 工作台一键启动
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 webapp/server.py "$@"

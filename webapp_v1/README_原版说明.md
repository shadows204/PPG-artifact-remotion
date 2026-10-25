# webapp_v1 —— 原始版（v1.0）完整源码

- 来源：git tag `webapp-v1-original`（提交 `45d3448`），与当时交付的原版**行为一致**（演示数据为合成正弦、演示态显示 PRD）。
- 用途：① 作为对照版本；② 与 v2 一起打包成**单文件 exe**（`PPG-Workbench-v1.exe`）。
- 与当初原版的唯一差异（仅为"无控制台打包"适配，不改业务逻辑）：
  1. 无控制台模式下把 stdout/stderr 重定向到 `workbench.log`（否则 windowed 模式会崩）；
  2. 新增 `POST /api/shutdown` + 网页「退出应用」按钮（无控制台时没有 Ctrl+C 可用）。

独立运行：`python webapp_v1/server.py --port 8020`

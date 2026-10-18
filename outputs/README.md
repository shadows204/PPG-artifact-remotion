# 运行产物目录（仓库内为空）

所有脚本的产物都写到这里（`scripts/_common/paths.py` 的 `artifact(name)` → `outputs/<name>/`）。

常见产物：

| 目录 | 生成脚本 | 内容 |
|---|---|---|
| `denoise_demo/` | `run_denoise.py --demo` | 三波形 `result.npz/json` + `waveforms.png` |
| `final_10seed/features/` | `pipeline.py step3_features` | 波形路径训练/测试特征（`XrlsOnly` 等） |
| `final_10seed/weights/` | `pipeline.py step4_train` / `train_residual.py` | 训练权重 |
| `b5_fuse/` | `fuse_branches.py` | 融合结果 `final_fused.json`（PRD 46.57） |
| `lite_eval/` `full_eval/` | `ensemble_cime.py` | 集成评估 |
| `cross_dataset_hr*.json` | `cross_dataset_eval.py` | 6 域心率结果 |
| `figures/` | `draw_framework.py` | 框架图 |

> 仓库不携带任何运行产物（避免"结果无法验证"的问题）；一切数字都可用 §6 的命令重新生成。

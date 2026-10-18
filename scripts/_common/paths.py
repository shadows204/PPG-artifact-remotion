"""PPG 运动伪影抑制项目 · 统一路径解析。

目录约定（脚本可自由移动，结构只在这里定义一次）::

    <工作区根>/
    ├── scripts/    所有脚本
    │   ├── _common/    共享工具（本文件）
    │   ├── pipeline/   主流水线 pipeline.py（RLS + 1D U-Net，10 步）
    │   ├── modules/    模块化实现（RLS / U-Net / 数据集 / 评估）
    │   └── vendor/     上游参考实现（uEDA-Net、Kernel-Adaptive-Filtering）
    ├── reports/    赛题、方案、实验汇报、对标文献
    ├── outputs/    全部数据、中间特征、模型权重、实验结果
    └── datasets/   原始公开数据集（当前快照未包含）

环境变量（全部可选，用于覆盖默认位置）:

    PPG_WORKSPACE   工作区根目录
    PPG_DATASETS    原始数据集根（其下应含 CIME_PPG_dataset / IEEEPPG_TROIKA）
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))       # <root>/scripts/_common


def workspace_root() -> str:
    """向上查找同时含 scripts/ 与 outputs/ 的目录。"""
    env = os.environ.get("PPG_WORKSPACE")
    if env:
        return os.path.abspath(env)
    p = os.path.dirname(os.path.dirname(HERE))          # 默认：scripts/_common -> 工作区根
    while True:
        if os.path.isdir(os.path.join(p, "scripts")) and os.path.isdir(os.path.join(p, "outputs")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return os.path.abspath(os.path.join(HERE, "..", ".."))
        p = parent


ROOT = workspace_root()
SCRIPTS = os.path.join(ROOT, "scripts")
REPORTS = os.path.join(ROOT, "reports")
OUTPUTS = os.path.join(ROOT, "outputs")


def artifact(name: str) -> str:
    """outputs/ 下的产物目录，例如 artifact("rls_unet")。

    `name` 可以带子目录（如 "ppg_ma/data"），返回值只做字符串拼接，不创建目录。
    """
    return os.path.join(OUTPUTS, name)


def datasets_root() -> str:
    """原始公开数据集根目录。

    当前工作区快照未包含 datasets/，因此这里只返回候选路径；
    需要读取原始 .mat / TROIKA 的步骤（step1/step9/step10）会自行报错提示。
    """
    env = os.environ.get("PPG_DATASETS")
    if env:
        return os.path.abspath(env)
    return os.path.join(ROOT, "datasets")


def ensure(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)

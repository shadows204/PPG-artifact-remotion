#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""画当前 PPG 运动伪影去除方案的框架图（可重复执行，参数与数字取自归档产物）。

图里刻意区分三类内容：
1. 主干（实线）：实际在用的流程 —— 物理先验(Rls) + 残差 U-Net（仅训练域）+ 决策层(PSD→可靠性加权→Viterbi)；
2. 系统层（虚线）：部署时的 OOD 门控 / AdaBN / 护栏（保证"永不比不做差"）；
3. 已排除（灰框 ×）：经消融或跨域实验证伪的组件，避免后人重做。

用法
----
    PY=/home/qingdongj/code/Owns/备选方案/venv/bin/python
    $PY scripts/tools/draw_framework.py --out outputs/figures
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, os.path.join(SCRIPTS, "modules"))
from _common import paths as P                                      # noqa: E402

C = dict(
    data="#e8f1fb", pre="#eef6ee", phys="#ffe9d6", wave="#e6e6fa",
    hr="#e0f4f4", dep="#f3e8f7", excl="#f0f0f0", res="#fff4cc", edge="#333333",
)


def setup_font():
    # 中文（Noto Sans CJK SC）+ 拉丁/符号（DejaVu Sans）双字体回退
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans", "Noto Sans CJK JP"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, fc, fs=9, ec=None, lw=1.2, style="round,pad=0.3",
        ls="-", bold=False, tc="#111111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                fc=fc, ec=ec or C["edge"], lw=lw, ls=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, color=tc, fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, p1, p2, color="#333333", lw=1.4, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, lw=lw,
                                 color=color, linestyle=ls, zorder=1,
                                 connectionstyle="arc3,rad=%.2f" % rad))


def main():
    ap = argparse.ArgumentParser(description="画 PPG 伪影去除框架图")
    ap.add_argument("--out", default=P.artifact("figures"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_font()

    fig, ax = plt.subplots(figsize=(17.5, 11.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.text(50, 97.4, "PPG 运动伪影去除：当前流程框架", ha="center", va="center",
            fontsize=19, fontweight="bold")
    ax.text(50, 94.2, "物理先验为主干（时域自适应抵消）· 网络只做残差修正（仅训练域）· 决策层负责跨域可靠性",
            ha="center", va="center", fontsize=10.5, color="#444444")

    # ---------------- 1. 数据层 ----------------
    ax.text(1.2, 90.6, "① 数据与真值", fontsize=11.5, fontweight="bold")
    box(ax, 1.5, 81.5, 22, 7.6,
        "CIME PPG（主数据集）\n48 受试者 = 10 训练 / 38 测试\n"
        "5000 训练段 + 1900 测试段（30 s @100 Hz）\n"
        "[带噪PPG, ACC_x, Gyro_y]  →  同步干净 PPG", C["data"], fs=8.6)
    box(ax, 25, 81.5, 21, 7.6,
        "TROIKA / IEEE SPC 2015（运动域）\n12 训练 + 10 测试，125 Hz\n"
        "sig=[ECG,PPG1,PPG2,ACCx,ACCy,ACCz]\nBPM0（8 s 窗 / 2 s 步进）", C["data"], fs=8.6)
    box(ax, 47.5, 81.5, 20, 7.6,
        "WESAD（腕部日常）\n15 受试者，BVP 64 / ACC 32 / ECG 700 Hz\n"
        "腕部 BVP+ACC，胸带 ECG\n（几乎无运动伪影 → 只测「不退化」）", C["data"], fs=8.6)
    box(ax, 69, 81.5, 29.5, 7.6,
        "BIDMC（53 例）/ CapnoBase（42 例）\n125 Hz / 300 Hz，各 8 min\n"
        "PLETH + ECG  →  无加速度计\n→ 只能走单通道路径（RLS 不可复现）\n"
        "＋ HUAWEI_MIDHIGH（91 包，无真值；漂移主导 → 仅鲁棒性参考）\n"
        "＋ WRIST（19 条，运动中腕部 PPG+ACC+Gyro，R 峰真值）\n"
        "＋ PPG-DALiA（117 条，腕部 E4 + 胸带 ECG 真值）", C["data"], fs=7.0)

    # ---------------- 2. 预处理 ----------------
    ax.text(1.2, 78.4, "② 预处理", fontsize=11.5, fontweight="bold")
    box(ax, 1.5, 68.6, 46, 8.4,
        "波形路径：切 30 s 段（3000 点）  →  段内标准化（除以含噪 PPG 的 std）\n"
        "心率路径：切 8 s 窗 / 2 s 步进（与 BPM0 对齐）  →  零填充 ×4 的 Hann 窗 PSD\n"
        "（跨数据集统一：每 30 s 段内标准化；不再用整段 std）", C["pre"], fs=8.8)
    box(ax, 49.5, 68.6, 49, 8.4,
        "多导联处理：取第 1 路 PPG（TROIKA 2 导联、WESAD 1 导联、CapnoBase pleth）\n"
        "重采样：TROIKA 125→100 Hz、WESAD 64→100 Hz、BIDMC 125→100 Hz、CapnoBase 300→100 Hz\n"
        "采样率/通道顺序已按数据集官方说明逐条核对（见 datasets/README.md）", C["pre"], fs=8.8)

    # ---------------- 3. 物理先验 ----------------
    ax.text(1.2, 65.4, "③ 物理先验：RLS 自适应抵消（主干）", fontsize=11.5, fontweight="bold")
    box(ax, 1.5, 54.0, 97, 10.4,
        "参考 = 运动通道（CIME: ACC_x + Gyro_y 两路；TROIKA/WESAD: 3 轴 ACC）\n"
        "$\\hat{M}$ = RLS(PPG, 参考)   →   PPG_RLS = PPG − α·$\\hat{M}$       参数：taps = 4, λ = 0.995, α = 0.40"
        "（CIME 训练集调出，train split，无泄漏）\n"
        "CIME α 消融：PRD 103.41(α=0) → 83.78(α=0.40) → 79.71(α=0.60 最优，曲线单峰)　|　"
        "参考通道消融：仅 ACC 89.05 / 仅 Gyro 90.44 / 两路 83.78（异质参考有用）", C["phys"], fs=8.8)
    arrow(ax, (50, 68.6), (50, 64.4))

    # ---------------- 4. 两条路径 ----------------
    ax.text(1.2, 51.6, "④ 两条路径（分离：U-Net 不进心率路径）", fontsize=11.5, fontweight="bold")

    # 上：波形恢复
    box(ax, 1.5, 38.2, 47, 12.4,
        "路径 A · 波形恢复（仅 CIME：唯一有同步干净真值）\n"
        "特征 X = [PPG, $\\hat{M}$, PPG_RLS]（3 通道，段内标准化）\n"
        "残差 U-Net：out = PPG_RLS + Δ(X)，Δ 末层零初始化（起点＝只用物理先验）\n"
        "主干归一化：3 通道用 InstanceNorm / 1 通道用 BatchNorm（AdaBN 依赖）\n"
        "多种子波形集成（41 成员，免训练）＋ 频域 CRM 分支融合（权重 15%）\n"
        "0.67 M 参数、5 级 stride-2、加性跳连；损失 = MSE + 0.5·L1\n"
        "epoch 选择：从训练集留出受试者 9 做验证（不用测试集）", C["wave"], fs=8.6)
    box(ax, 1.5, 28.0, 47, 9.4,
        "评估（1900 测试段，受试者级 15 种子）\n"
        "Raw 103.41 / RLS-only 83.78 / 本方案 46.57（0.85×残差集成 47.15 ＋ 0.15×频域 CRM 54.09；单次融合口径）\n"
        "Pearson 0.9395 · SNR 7.78 dB　　判决 4/5（C5 临界：std 0.0095 vs 0.01）", C["res"], fs=8.8)
    arrow(ax, (25, 38.2), (25, 37.4))
    arrow(ax, (25, 54.0), (25, 50.6))

    # 下：心率估计
    box(ax, 51.5, 42.4, 47, 8.2,
        "路径 B · 心率估计（跨域可用，不需要真值）\n"
        "输入：有 IMU → RLS 3 轴（α=1.0）；无 IMU → raw / 带通\n"
        "8 s 窗 PSD → 窗口级可靠性加权 w = s2^0.5，s2 = 峰值处 P_PPG/(P_PPG+P_art)\n"
        "logP_eff = w·logP + (1−w)·mean_band(logP)　（每窗一个标量，无频率偏好）", C["hr"], fs=8.6)
    box(ax, 51.5, 32.4, 47, 8.6,
        "Viterbi 全局最优路径（λ=0.5, jump=4.0, band=0.7–3.0 Hz，抛物线插值到亚 bin）\n"
        "超参用留一记录 CV 标定（12 条 TROIKA 训练记录 → 稳定性选择）\n"
        "→ HR 序列（bpm），与 BPM0 / ECG 比较", C["hr"], fs=8.6)
    box(ax, 51.5, 28.0, 47, 3.4,
        "TROIKA 1328 窗：Raw 6.64 → RLS 2.73 → +可靠性加权 2.62 → +双导联选路 2.11（margin 仅训练集选；原 2.10 为测试集选）→ +弃权 1.42（80% 覆盖率曲线点，探索性）\n"
        "WESAD：RLS 8.28 → RLS+模型混合 **7.15 ± 2.13**（5 折 CV，5/5 改善；单次留出 6.02 仅 3 人）｜BIDMC 3.00 → AdaBN+TENT 1.48｜CapnoBase 1.06（剔除 1 条 0.80）\n"
        "适用边界（6 个有真值域 / 256 条记录）：静息临床 1.06–1.48 ｜ 低运动腕部 1.70–7.15 ｜ 跑步指夹 2.11 ｜\n"
        "运动中腕部 11.7–36.6（失败区） → 伪影/谐波锁定，已证仅凭 PPG+ACC 不可修 → 需第二模态",
        C["res"], fs=8.6)
    arrow(ax, (75, 54.0), (75, 50.6))
    arrow(ax, (75, 42.4), (75, 41.0))
    arrow(ax, (75, 32.4), (75, 31.4))

    # ---------------- 5. 部署层 ----------------
    ax.text(1.2, 25.4, "⑤ 部署层：可靠性门控（保证「永不比不做差」）", fontsize=11.5, fontweight="bold")
    box(ax, 1.5, 14.6, 31, 9.4,
        "OOD 判据（离散门控）\nBN 统计漂移 > 0.15 → 不让网络介入\n"
        "有 IMU → RLS(3轴)；无 IMU → raw\n"
        "实测四域 100% 门控 → 逐位等于物理先验\n"
        "（TROIKA 2.73 / WESAD 8.28 / BIDMC 3.00 / CapnoBase 1.06）", C["dep"], fs=8.5)
    box(ax, 34, 14.6, 30, 9.4,
        "无标签域适配（第 28 轮验证定稿）\n"
        "有 IMU：RLS + 可靠性加权（TROIKA 2.62）\n"
        "　　　　网络只能追平（AdaBN/TENT 无益）\n"
        "无 IMU：AdaBN + TENT → BIDMC 3.00→1.29\n"
        "　　　　CapnoBase 7.40→1.06（raw 本身 1.06）\n"
        "静息/临床可加「次谐波优先」→ 1.49\n"
        "（>120 bpm 启用；勿与前项叠加）", C["dep"], fs=7.6)
    box(ax, 65.5, 14.6, 33, 9.4,
        "网络使用建议（基于全部证据）\n"
        "有真值域：波形增强（CIME 103→51）\n"
        "无 IMU 域：AdaBN+TENT 优于 raw\n"
        "有 IMU 域：仅 WESAD 上 DR 残差小幅占优\n"
        "伪标签微调：中位 3.44、无可靠增益\n"
        "自监督适配（无标签）：三域追平物理基线\n"
        "（33.4→3.5、9.4→1.1、10.5→7.4）但未超越", C["dep"], fs=7.6)
    arrow(ax, (25, 28.0), (17, 24.0), color="#8a6d3b", ls="--", rad=-0.15)
    arrow(ax, (75, 28.0), (80, 24.0), color="#8a6d3b", ls="--", rad=0.15)

    # ---------------- 6. 已排除 ----------------
    box(ax, 1.5, 3.0, 97, 10.0,
        "× 已被消融/跨域实验排除（不要重做）：\n"
        "逐 bin 频率加权（伪影谱/步频/谐波梳；60 配置全部不优于基线，跑步时伪影与脉搏带重叠 → 压掉真峰）　|　"
        "硬陷波 notch（7.29 vs 6.38）　|　能量守恒损失 + |Δ| 惩罚（50.94→52.8，方差×4）\n"
        "域随机化（域内 55.04 vs 53.07，跨域 −19~−33）　|　Attention Gate / FiLM / 频域损失 / GAN（严格口径下无一项稳定超过简单基线）\n"
        "线性混合网络与物理先验（权重-误差非单调）　|　自洽粗先验门控（2.69→3.61）　|　"
        "自参考物理先验（延迟自参考/包络/SSA/双导联：BIDMC 30.7·64.5、TROIKA 6.19~18.0）　|　"
        "去相关损失单独使用（TROIKA 33.4→13.9，会连脉搏一起消掉）",
        C["excl"], fs=8.3, ls="--", tc="#555555")

    # 侧栏：密钥
    ax.text(50, 100.2, "", ha="center")
    fig.tight_layout()
    path = os.path.join(args.out, "pipeline_framework.png")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print("[fig] 已保存 -> %s" % path)
    return path


if __name__ == "__main__":
    main()

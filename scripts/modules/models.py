"""1D U-Net（PPG 版）

架构蓝本：yongbin98/uEDA-Net 的 model/Billal_UNet.py::denoising_model
  - 5 级 stride=2 编码 + 5 级解码，**加法**跳连（x + x4 形式）
  - 上采样用 ConvTranspose1d + output_padding=1，与奇数核 + same padding 严格互逆

本实现在其之上做三处对齐改动：
  1. 首层 in_channels: 1 -> C（C = 1 + RLS 特征 + IMU 通道）
  2. 新增 FiLM 条件注入：stats (B, 2) = [运动强度, 谱熵]，逐层调制特征
  3. 内部自动 pad 到 32 的倍数并在输出裁回原长（CIME 段长 3000 不是 32 的倍数）

形状关系（L 为 32 的倍数时严格可逆）：
  L -> L/2 -> L/4 -> L/8 -> L/16 -> L/32 -> L/16 -> L/8 -> L/4 -> L/2 -> L
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PPGDenoiserUNet", "count_parameters"]


class FiLM(nn.Module):
    """Feature-wise Linear Modulation：用条件向量缩放/平移特征。

    stats 的第 0 维驱动 gamma，第 1 维驱动 beta（沿用 uEDA-Net 的 FiLM 写法），
    这里 stats = [运动强度, PPG 谱熵]。零初始化 -> 训练初期等价于恒等映射。
    """

    def __init__(self, features: int, cond_dim: int = 2):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, features)
        self.beta = nn.Linear(cond_dim, features)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, stats):
        gamma = self.gamma(stats).unsqueeze(-1) + 1.0     # (B, C, 1)
        beta = self.beta(stats).unsqueeze(-1)
        return x * gamma + beta


def make_norm(kind: str, ch: int) -> nn.Module:
    """归一化层工厂（P3 域不变实验）。

    - `bn`：BatchNorm1d（原实现；running stats 跨域漂移是零样本失效主因，实测 0.08→0.48）
    - `in`：InstanceNorm1d（**无 running stats**，逐样本逐通道归一化 → 天然域不变）
    - `gn`：GroupNorm（同上，通道分组；小 batch 更稳）
    """
    kind = (kind or "bn").lower()
    if kind == "bn":
        return nn.BatchNorm1d(ch)
    if kind == "in":
        return nn.InstanceNorm1d(ch, affine=True, track_running_stats=False)
    if kind == "gn":
        g = 8 if ch % 8 == 0 else (4 if ch % 4 == 0 else 1)
        return nn.GroupNorm(g, ch)
    raise ValueError("未知 norm: %s" % kind)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, padding, stride, cond_dim=2, use_film=True, norm="bn"):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=padding, stride=stride)
        self.bn = make_norm(norm, out_ch)
        self.act = nn.LeakyReLU(0.2)
        self.film = FiLM(out_ch, cond_dim) if use_film else None

    def forward(self, x, stats=None):
        x = self.conv(x)
        x = self.bn(x)
        if self.film is not None and stats is None:
            raise ValueError("模型带 FiLM（cond_dim>0），forward 必须传 stats（原实现会静默跳过条件）")
        if self.film is not None and stats is not None:
            x = self.film(x, stats)
        return self.act(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, padding, out_padding, stride,
                 cond_dim=2, use_film=True, norm="bn"):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, k, padding=padding,
                                         output_padding=out_padding, stride=stride)
        self.bn = make_norm(norm, out_ch)
        self.act = nn.LeakyReLU(0.2)
        self.film = FiLM(out_ch, cond_dim) if use_film else None

    def forward(self, x, stats=None):
        x = self.deconv(x)
        x = self.bn(x)
        if self.film is not None and stats is not None:
            x = self.film(x, stats)
        return self.act(x)


class PPGDenoiserUNet(nn.Module):
    """输入 (B, C, L) + 条件 (B, 2)  ->  输出 (B, 1, L)。"""

    def __init__(self, in_channels: int = 5, cond_dim: int = 2,
                 use_film: bool = True, out_channels: int = 1, norm: str = "bn"):
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.norm = (norm or "bn").lower()
        self.mult = 32                      # 5 级 stride=2
        uf = use_film

        # ---- encoder（核长/填充与原仓库一致）----
        self.conv1 = ConvBlock(in_channels, 16, 65, 32, 2, cond_dim, uf, self.norm)
        self.conv2 = ConvBlock(16, 32, 31, 15, 2, cond_dim, uf, self.norm)
        self.conv3 = ConvBlock(32, 64, 15, 7, 2, cond_dim, uf, self.norm)
        self.conv4 = ConvBlock(64, 128, 7, 3, 2, cond_dim, uf, self.norm)
        self.conv5 = ConvBlock(128, 256, 7, 3, 2, cond_dim, uf, self.norm)

        # ---- decoder（加法跳连）----
        self.deconv1 = DeconvBlock(256, 128, 7, 3, 1, 2, cond_dim, uf, self.norm)
        self.deconv2 = DeconvBlock(128, 64, 7, 3, 1, 2, cond_dim, uf, self.norm)
        self.deconv3 = DeconvBlock(64, 32, 15, 7, 1, 2, cond_dim, uf, self.norm)
        self.deconv4 = DeconvBlock(32, 16, 31, 15, 1, 2, cond_dim, uf, self.norm)
        self.deconv5 = nn.ConvTranspose1d(16, out_channels, 7, padding=3,
                                          output_padding=1, stride=2)

    # ---- 对外接口 ----
    def forward(self, x, stats=None):
        L = x.shape[-1]
        pad = (-L) % self.mult
        if pad:
            # replicate 填充比补零更贴近原仓库的 padding_mode 取向
            x = F.pad(x, (0, pad), mode="replicate")
        y = self._forward_padded(x, stats)
        return y[..., :L]

    def _forward_padded(self, x, stats):
        x1 = self.conv1(x, stats)
        x2 = self.conv2(x1, stats)
        x3 = self.conv3(x2, stats)
        x4 = self.conv4(x3, stats)
        x5 = self.conv5(x4, stats)

        y = self.deconv1(x5, stats)
        y = self.deconv2(y + x4, stats)
        y = self.deconv3(y + x3, stats)
        y = self.deconv4(y + x2, stats)
        y = self.deconv5(y + x1)
        return y


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

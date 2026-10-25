#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""残差化 RLS+U-Net：把"网络整体重写波形"改成"只学相对物理先验的小修正"。

动机（来自本次跨域诊断）
------------------------
CIME 上网络输出与输入的相关 0.92~0.95；到域外掉到 **0.17~0.51**，且 0.7–4 Hz 能量占比
被大幅改写（BIDMC −0.55）——网络在域外**整体重写**波形，把脉搏当伪影删掉。

做法
----
    out = base(x) + Δ(x)
其中 `base(x)` 是**物理先验**（有 IMU 时 = PPG_RLS = PPG − α·M̂；无 IMU 时 = PPG），
`Δ` 由 U-Net 预测，且**末层零初始化** → 训练起点恰好等价于"只用 RLS/raw"，
域外 Δ 学不到东西时会自然退化回物理先验，而不是把信号改坏。

配套损失（在 `train_residual.py` 里）：
- 重建：MSE + 0.5·L1（与主流水线一致）；
- **心率带能量守恒**：惩罚 out 与 clean 的 0.7–4 Hz 能量占比之差（对数域），直接对症"削掉心率带"；
- **Δ 能量惩罚**：鼓励最小改动（identity 保护）。

用法（作为库）
--------------
    from residual import build, save, load
    m = build(in_channels=3, base_index=2)      # 3 通道，物理先验取第 2 通道 PPG_RLS
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import PPGDenoiserUNet, count_parameters       # noqa: E402


class ResidualUNet(nn.Module):
    """out = x[:, base_index] + Δ(x)；Δ 由 `PPGDenoiserUNet` 预测（末层零初始化）。"""

    def __init__(self, in_channels: int = 3, cond_dim: int = 0, base_index: int = 2,
                 direct: bool = False, norm: str = "bn", zero_init: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.base_index = base_index
        self.direct = bool(direct)          # True = 直接输出整条波形（消融对照），不做残差
        self.norm = (norm or "bn").lower()  # P3：bn / in / gn
        self.net = PPGDenoiserUNet(in_channels=in_channels, cond_dim=cond_dim,
                                   use_film=(cond_dim > 0), norm=self.norm)
        if not self.direct and zero_init:
            self._zero_last()          # 零初始化 → Δ≡0，起点恰为物理先验（设计选择，可消融）

    def _zero_last(self):
        """把最后一个 ConvTranspose1d 的权重与偏置置零 → Δ 初始恒为 0（等价于只用物理先验）。"""
        last = None
        for m in self.net.modules():
            if isinstance(m, nn.ConvTranspose1d):
                last = m
        if last is not None:
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(self, x, stats=None):
        if stats is None:
            stats = x.new_zeros((x.shape[0], 0))
        if self.direct:
            return self.net(x, stats)
        base = x[:, self.base_index:self.base_index + 1]
        return base + self.net(x, stats)

    def delta(self, x, stats=None):
        return self.forward(x, stats) - x[:, self.base_index:self.base_index + 1]


def build(in_channels: int = 3, cond_dim: int = 0, base_index: int = 2,
          direct: bool = False, norm: str = "bn", zero_init: bool = True) -> ResidualUNet:
    return ResidualUNet(in_channels=in_channels, cond_dim=cond_dim, base_index=base_index,
                        direct=direct, norm=norm, zero_init=zero_init)


def count_parameters_(m: nn.Module) -> int:
    return count_parameters(m)


def save(path: str, model: ResidualUNet, meta: dict | None = None) -> None:
    """存成与主流水线兼容的 checkpoint（`model` 是**基础网络**的 state_dict + residual 标记）。"""
    payload = dict(model={k: v.detach().cpu() for k, v in model.net.state_dict().items()},
                   in_channels=int(model.in_channels), cond_dim=int(model.cond_dim),
                   # xkey/cond 按结构推导（旧版硬编码 "XrlsOnly"/"none"：1 通道模型会被下游错配）
                   xkey=str((meta or {}).get("xkey") or
                            ("Xppg" if int(model.in_channels) == 1
                             else ("XnoRLS" if int(model.in_channels) == 5 else "XrlsOnly"))),
                   cond=str((meta or {}).get("cond") or ("stats" if int(model.cond_dim) > 0 else "")),
                   residual=(not model.direct), direct=bool(model.direct),
                   base_index=int(model.base_index), norm=str(getattr(model, "norm", "bn")))
    for k, v in (meta or {}).items():
        if isinstance(v, (int, float, str, bool, type(None))):
            payload[k] = v
        elif isinstance(v, (list, tuple)) and all(
                isinstance(x, (int, float)) for x in v):
            payload[k] = [float(x) for x in v]      # FiLM 条件标准化参数（cond_mu/cond_sd）
    torch.save(payload, path)


def load(path: str, device="cpu", strict: bool = True):
    """读回残差模型；返回 (ResidualUNet, meta)。"""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    in_ch, base_i = int(ck["in_channels"]), int(ck.get("base_index", 2))
    model = ResidualUNet(in_channels=in_ch, cond_dim=int(ck.get("cond_dim", 0)), base_index=base_i,
                         direct=bool(ck.get("direct", False)), norm=str(ck.get("norm", "bn")))
    model.net.load_state_dict(ck["model"], strict=strict)
    model.eval(); model.to(device)
    meta = {k: v for k, v in ck.items() if k != "model"}
    return model, meta


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def band_ratio_torch(x, fs: float = 100.0, band=(0.7, 4.0)):
    """(B,1,L) → (B,) 的心率带能量占比。"""
    P = torch.fft.rfft(x, dim=-1).abs() ** 2
    fr = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
    m = (fr >= band[0]) & (fr <= band[1])
    return P[..., m].sum(-1) / (P.sum(-1) + 1e-12)


def energy_consistency_loss(out, target, fs: float = 100.0, band=(0.7, 4.0)):
    """心率带能量占比的对数差（越小越说明没有把脉搏带削掉/造出来）。"""
    ro, rt = band_ratio_torch(out, fs, band), band_ratio_torch(target, fs, band)
    return (torch.log(ro + 1e-6) - torch.log(rt + 1e-6)).abs().mean()


if __name__ == "__main__":     # 自检
    torch.manual_seed(0)
    print("[自检] 残差 U-Net")
    for in_ch, bi in ((3, 2), (1, 0)):
        m = build(in_ch, 0, bi).eval()
        x = torch.randn(2, in_ch, 3000)
        with torch.no_grad():
            y = m(x, None)
        d = (y - x[:, bi:bi + 1]).abs().max().item()
        print("  in=%d base=%d  out%s  初始 Δmax=%.2e（应为 0，等价于只用物理先验）  参数 %.3f M"
              % (in_ch, bi, tuple(y.shape), d, count_parameters_(m) / 1e6))
        assert d < 1e-6
    # 损失可反传
    m = build(3, 0, 2).train()
    x = torch.randn(4, 3, 512); y = torch.randn(4, 1, 512)
    out = m(x, None)
    loss = nn.functional.mse_loss(out, y) + 0.5 * energy_consistency_loss(out, y) \
        + 0.02 * out.sub(x[:, 2:3]).abs().mean()
    loss.backward()
    print("  损失 %.4f 反传 OK" % float(loss))
    print("[自检] 通过 ✅")

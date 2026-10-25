#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""频域分支：复数比率掩码（CRM）残差模型（批次 5）。

设计（与残差哲学一致）
----------------------
- STFT(PPG_RLS) 作为基准复谱 `S0`；网络预测**残差掩码** Δ（实/虚两通道）；
- 掩码 `M = 1 + Δ`，重建 `out = iSTFT(M · S0)`；
- **末层零初始化** → M≡1 → `out = iSTFT(S0) = PPG_RLS`（起点恰为物理先验，与 `ResidualUNet` 同构）。
- 输入谱特征：`[Re S0, Im S0, log1p|STFT(M̂)|, log1p|STFT(PPG)|]`（4 通道，逐样本标准化）。

用 2D U-Net（频率×时间）处理复谱；`F.interpolate` 处理奇数频率维，保证任意 `n_fft` 可重建。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(kind, ch):
    kind = (kind or "in").lower()
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        g = 8 if ch % 8 == 0 else (4 if ch % 4 == 0 else 1)
        return nn.GroupNorm(g, ch)
    return nn.InstanceNorm2d(ch, affine=True, track_running_stats=False)


class ConvB(nn.Module):
    def __init__(self, i, o, norm="in"):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, padding=1)
        self.n1 = make_norm(norm, o)
        self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.n2 = make_norm(norm, o)

    def forward(self, x):
        x = F.leaky_relu(self.n1(self.c1(x)), 0.2)
        return F.leaky_relu(self.n2(self.c2(x)), 0.2)


class CRMNet(nn.Module):
    """CRM 网。注意：**未实现 FiLM**，cond_dim>0 会被忽略（reports/41 审计 P1-6）。"""
    """(B,3,L) → (B,1,L)。内部做 STFT/iSTFT，接口与 ResidualUNet 相同（x, st）。"""

    def __init__(self, in_channels=3, base_index=2, n_fft=256, hop=64, base=16,
                 norm="in", cond_dim=0):
        super().__init__()
        self.in_channels = in_channels
        self.base_index = base_index
        self.n_fft = n_fft
        self.hop = hop
        self.norm_kind = norm
        self.cond_dim = cond_dim
        self.direct = False
        self.levels = 3
        self.e1 = ConvB(4, base, norm)
        self.e2 = ConvB(base, base * 2, norm)
        self.e3 = ConvB(base * 2, base * 4, norm)
        self.bott = ConvB(base * 4, base * 4, norm)
        self.u3 = nn.Conv2d(base * 8, base * 2, 3, padding=1)
        self.n3 = make_norm(norm, base * 2)
        self.u2 = nn.Conv2d(base * 4, base, 3, padding=1)
        self.n2 = make_norm(norm, base)
        self.u1 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.n1 = make_norm(norm, base)
        self.head = nn.Conv2d(base, 2, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)   # M ≡ 1 起步

    # ---- STFT 工具 ----
    def _stft(self, x):
        """x: (B,L) → (B,F,T) 复谱（本版 torch.stft 只接受 1D/2D 输入）。"""
        win = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=win,
                          return_complex=True, center=True)

    def _istft(self, S, length):
        # dtype 跟随输入（原硬编码 float32：double 输入会 RuntimeError）
        win = torch.hann_window(self.n_fft, device=S.device,
                                dtype=(S.real.dtype if torch.is_complex(S) else S.dtype))
        return torch.istft(S, n_fft=self.n_fft, hop_length=self.hop, window=win,
                           length=length, center=True)

    def forward(self, x, st=None):
        if x.dim() == 2:
            x = x[:, None, :]
        B, C, L = x.shape
        ppg = x[:, 0]
        mhat = x[:, 1] if C > 1 else torch.zeros_like(ppg)
        rls = x[:, self.base_index]
        S0 = self._stft(rls)
        Sm = self._stft(mhat).abs()
        Sp = self._stft(ppg).abs()
        inp = torch.cat([S0.real.unsqueeze(1), S0.imag.unsqueeze(1),
                         torch.log1p(Sm).unsqueeze(1), torch.log1p(Sp).unsqueeze(1)],
                        dim=1)                                   # (B,4,F,T)
        mu = inp.mean(dim=(1, 2, 3), keepdim=True)
        sd = inp.std(dim=(1, 2, 3), keepdim=True) + 1e-5
        inp = (inp - mu) / sd

        e1 = self.e1(inp)
        e2 = self.e2(F.avg_pool2d(e1, 2, ceil_mode=True))
        e3 = self.e3(F.avg_pool2d(e2, 2, ceil_mode=True))
        b = self.bott(F.avg_pool2d(e3, 2, ceil_mode=True))
        u3 = F.interpolate(b, size=e3.shape[2:], mode="nearest")
        u3 = F.leaky_relu(self.n3(self.u3(torch.cat([u3, e3], dim=1))), 0.2)
        u2 = F.interpolate(u3, size=e2.shape[2:], mode="nearest")
        u2 = F.leaky_relu(self.n2(self.u2(torch.cat([u2, e2], dim=1))), 0.2)
        u1 = F.interpolate(u2, size=e1.shape[2:], mode="nearest")
        u1 = F.leaky_relu(self.n1(self.u1(torch.cat([u1, e1], dim=1))), 0.2)
        d = self.head(u1)                                    # (B,2,F,T)
        mask = torch.complex(1.0 + d[:, 0], d[:, 1])          # (B,F,T)
        out = self._istft(mask * S0, L)                       # (B,L)
        if out.shape[-1] < L:                                 # 长度兜底
            out = F.pad(out, (0, L - out.shape[-1]))
        return out[:, :L].unsqueeze(1)


def build(in_channels=3, base_index=2, n_fft=256, hop=64, base=16, norm="in", cond_dim=0):
    return CRMNet(in_channels=in_channels, base_index=base_index, n_fft=n_fft, hop=hop,
                  base=base, norm=norm, cond_dim=cond_dim)


def save(path, model, meta=None):
    payload = dict(arch="crm", model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                   in_channels=int(model.in_channels), cond_dim=int(model.cond_dim),
                   base_index=int(model.base_index), norm=str(model.norm_kind),
                   n_fft=int(model.n_fft), hop=int(model.hop),
                   base=int(model.e1.c1.out_channels), residual=True, cond="none")
    for k, v in (meta or {}).items():
        if isinstance(v, (int, float, str, bool, type(None))):
            payload[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(z, (int, float)) for z in v):
            payload[k] = [float(z) for z in v]
    torch.save(payload, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    # base 优先读元信息；旧 checkpoint 没有该字段时**从 state_dict 形状推断**（head 输入通道数）
    base = int(ck.get("base") or ck["model"]["head.weight"].shape[1])
    m = CRMNet(in_channels=int(ck["in_channels"]), base_index=int(ck.get("base_index", 2)),
               n_fft=int(ck.get("n_fft", 256)), hop=int(ck.get("hop", 64)), base=base,
               norm=str(ck.get("norm", "in")), cond_dim=int(ck.get("cond_dim", 0)))
    m.load_state_dict(ck["model"])
    m.eval(); m.to(device)
    return m, {k: v for k, v in ck.items() if k != "model"}

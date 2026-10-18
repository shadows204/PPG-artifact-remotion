"""GAN 双判别器增强臂：时域 PatchGAN + 频域（幅度谱）判别器。

方案位置
--------
消融矩阵（``scripts/pipeline/pipeline.py::VARIANTS``）里的 E1/E2 最后一格：
在已训练好的 1D U-Net 波形恢复器之上，叠加**对抗式微调**，看能否在 PRD 之外
改善波形的"感知质量"（更锐利的脉搏峰、更干净的频带），同时不牺牲 PRD。

设计要点
--------
* **生成器**：直接 ``import`` 复用 ``scripts/modules/models.py::PPGDenoiserUNet``，
  不复制、不改写。本模块只负责加载/保存它的权重。
* **时域判别器** ``TemporalPatchDiscriminator``：4 层 ``stride=2`` 的 Conv1d + LeakyReLU，
  末层 stride=1 输出 **patch 级 logits**（PatchGAN 思路：不判"整段真假"，
  而是判"局部片段真假"，对 1D 生理波形比全局判别更稳、也更省参数）。
  除第一层外每层后接 InstanceNorm（不用 BatchNorm：GAN 训练 batch 小、
  BN 还会把同 batch 样本的统计量串起来，InstanceNorm 对 batch=1 也成立）。
* **频域判别器** ``SpectralDiscriminator``：先把波形做 ``torch.fft.rfft`` 取幅度谱
  （默认 ``log1p`` 压缩动态范围），再对谱轴做 1D Conv 判别，输出谱 patch logits。
  动机：运动伪影在频域表现为 2.5~10 Hz 的能量堆积，时域 PatchGAN 对
  "能量该落在哪个频带"不敏感，频域判别器正好补上这一维。
* **损失**：对抗损失默认 **LSGAN**（最小二乘，见 ``adversarial_d_loss``），
  重建损失为 L1（``--lambda-l1`` 加权），可选 **feature matching**（``--lambda-fm``）。
  判别器与生成器**交替更新**（先 D 后 G，同一 batch 内各更新一次）。
* **checkpoint**：只写 ``torch.Tensor`` 与内建类型（PyTorch >= 2.6 的
  ``torch.load`` 默认 ``weights_only=True``）；``assert_plain_checkpoint`` 会在写盘前
  递归校验，防止混入 numpy 数组导致之后加载失败。

与 ``pipeline.py`` 权重的兼容性（**重要**）
-------------------------------------------
``outputs/rls_unet/weights/*.pt`` 是 ``pipeline.py::make_model`` 里**内联定义**的网络
训出来的，层名是 ``e1.c.weight`` / ``e1.bn.*`` / ``d1.c.weight`` / ``d5.weight``，
而 ``models.py::PPGDenoiserUNet`` 的层名是 ``conv1.conv.weight`` / ``deconv1.deconv.weight`` /
``deconv5.weight``。``remap_legacy_unet_keys`` 负责两种命名的互转（含
``film.g/film.b`` -> ``film.gamma/film.beta``）。

另有一处**结构差异**（无法在不改 pipeline.py 的前提下消除，见 GAN_说明.md）：
``pipeline.py::ConvB`` 是 ``film(act(bn(conv(x))))``，``models.py::ConvBlock`` 是
``act(film(bn(conv(x))))``；两者仅在 FiLM 生效（``cond != "none"``）时不等价。
默认 ``--init rls_only.pt``（``cond_dim=0``，无 FiLM）完全等价。
"""
from __future__ import annotations

import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from models import PPGDenoiserUNet, count_parameters  # noqa: E402

__all__ = [
    "TemporalPatchDiscriminator",
    "SpectralDiscriminator",
    "adversarial_d_loss",
    "adversarial_g_loss",
    "feature_matching_loss",
    "spectral_magnitude",
    "remap_legacy_unet_keys",
    "load_generator_checkpoint",
    "make_gan_checkpoint",
    "assert_plain_checkpoint",
    "save_gan_checkpoint",
    "count_parameters",
]


# ---------------------------------------------------------------------------
# 判别器
# ---------------------------------------------------------------------------
def _norm_layer(kind: str, channels: int) -> nn.Module:
    """判别器用的归一化层；``none`` 时返回 Identity。"""
    if kind == "instance":
        # affine=True 让它有可学的缩放/平移；track_running_stats=False（默认）
        # 使其在 batch=1 的推理下行为一致
        return nn.InstanceNorm1d(channels, affine=True)
    if kind == "batch":
        return nn.BatchNorm1d(channels)
    return nn.Identity()


def _run_conv_stack(body: nn.Sequential, head: nn.Conv1d, h: torch.Tensor,
                    return_features: bool):
    """时域/频域判别器共用的前向：收集每个 LeakyReLU 之后的特征（feature matching 用）。"""
    feats = []
    for blk in body:
        h = blk(h)
        if isinstance(blk, nn.LeakyReLU):
            feats.append(h)
    logits = head(h)
    if return_features:
        return logits, feats
    return logits


class TemporalPatchDiscriminator(nn.Module):
    """时域 1D PatchGAN 判别器。

    ``n_layers`` 层 ``stride=2`` Conv1d（每层通道数翻倍，上限 ``max_channels``），
    每层后接 LeakyReLU；第一层不加归一化（PatchGAN 惯例），之后各层加 InstanceNorm。
    最后再叠一层 ``stride=1`` 的 Conv1d -> 1 通道，得到 **patch 级 logits**
    ``(B, 1, T')``，LSGAN 直接对每个 patch 位置算最小二乘。

    输入 ``(B, C, L)``（本任务 C=1，干净/生成 PPG），``L=3000`` 时 T' = 188。
    """

    def __init__(self, in_channels: int = 1, base: int = 32, n_layers: int = 4,
                 kernel: int = 15, norm: str = "instance",
                 negative_slope: float = 0.2, max_channels: int = 512):
        super().__init__()
        assert n_layers >= 2, "至少需要 2 层卷积"
        self.in_channels = in_channels
        self.n_layers = n_layers

        blocks = []
        ch_in, ch = in_channels, base
        for i in range(n_layers):
            blocks.append(nn.Conv1d(ch_in, ch, kernel, stride=2, padding=kernel // 2))
            if i > 0:
                blocks.append(_norm_layer(norm, ch))
            blocks.append(nn.LeakyReLU(negative_slope, inplace=True))
            ch_in = ch
            ch = min(ch * 2, max_channels)
        self.body = nn.Sequential(*blocks)
        # 输出 1 通道 patch logits：不加激活（LSGAN/BCE-with-logits 都要 raw logits）
        self.head = nn.Conv1d(ch_in, 1, kernel, stride=1, padding=kernel // 2)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """x: (B, C, L) -> logits (B, 1, T')；``return_features`` 时额外返回中间特征表。"""
        return _run_conv_stack(self.body, self.head, x, return_features)


class SpectralDiscriminator(nn.Module):
    """频域判别器：对 ``|rFFT|``（默认 ``log1p``）的幅度谱做 1D Conv 判别。

    输入仍是时域波形 ``(B, C, L)``，内部做 ``torch.fft.rfft``（可导，梯度能回到生成器）。
    ``L=3000`` 时谱轴 1501 点，4 层 stride=2 后约 94 点，输出谱 patch logits ``(B, 1, F')``。

    ``spectrum="log"`` 用 ``log1p(|rFFT|)`` 压缩动态范围（推荐，脉搏谱峰与伪影谱峰
    幅度可能差几个数量级，直接线性输入会让判别器只盯着最大峰）；
    ``spectrum="mag"`` 用原始幅度谱；``spectrum="psd"`` 用 ``|rFFT|**2 / L``。

    注意：与 ``dataset.py::spectral_entropy`` 一致，这里用**未归一化**的 rfft，
    不做任何窗函数（判别器看的是整段谱形状，加窗会平滑掉它要抓的差异）。
    """

    def __init__(self, in_channels: int = 1, base: int = 32, n_layers: int = 4,
                 kernel: int = 9, norm: str = "instance", spectrum: str = "log",
                 negative_slope: float = 0.2, max_channels: int = 512):
        super().__init__()
        assert n_layers >= 2, "至少需要 2 层卷积"
        assert spectrum in ("log", "mag", "psd"), "spectrum 只能是 log/mag/psd"
        self.in_channels = in_channels
        self.spectrum = spectrum

        blocks = []
        ch_in, ch = in_channels, base
        for i in range(n_layers):
            blocks.append(nn.Conv1d(ch_in, ch, kernel, stride=2, padding=kernel // 2))
            if i > 0:
                blocks.append(_norm_layer(norm, ch))
            blocks.append(nn.LeakyReLU(negative_slope, inplace=True))
            ch_in = ch
            ch = min(ch * 2, max_channels)
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv1d(ch_in, 1, kernel, stride=1, padding=kernel // 2)

    def to_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) -> (B, C, F) 的谱表示（可导）。"""
        mag = torch.fft.rfft(x, dim=-1).abs()
        if self.spectrum == "psd":
            return mag ** 2 / max(1, x.shape[-1])
        if self.spectrum == "log":
            return torch.log1p(mag)
        return mag

    def forward(self, x: torch.Tensor, return_features: bool = False):
        return _run_conv_stack(self.body, self.head, self.to_spectrum(x), return_features)


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------
def adversarial_d_loss(kind: str, real_logits: torch.Tensor,
                       fake_logits: torch.Tensor) -> torch.Tensor:
    """判别器对抗损失。

    ``kind="lsgan"``（默认）：``0.5 * [ mean((D(x)-1)^2) + mean(D(G(z))^2) ]``
      —— 最小二乘 GAN。选它的理由：① 梯度随误差线性衰减而不是被 sigmoid 饱和掉，
      微调阶段（生成器已经"大致对"）不会一开始就梯度消失；② 只有 MSE，
      不需要调 BCE 的 logits 数值稳定性；③ 对学习率比 BCE 鲁棒。
    ``kind="bce"``：标准 ``BCEWithLogits``，便于与文献结果对齐时切换。
    """
    if kind == "lsgan":
        return 0.5 * (F.mse_loss(real_logits, torch.ones_like(real_logits))
                      + F.mse_loss(fake_logits, torch.zeros_like(fake_logits)))
    if kind == "bce":
        return 0.5 * (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                      + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
    raise ValueError("未知对抗损失类型：%r（可选 lsgan/bce）" % (kind,))


def adversarial_g_loss(kind: str, fake_logits: torch.Tensor) -> torch.Tensor:
    """生成器对抗损失（把假样本推向"真"）。"""
    if kind == "lsgan":
        return F.mse_loss(fake_logits, torch.ones_like(fake_logits))
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    raise ValueError("未知对抗损失类型：%r（可选 lsgan/bce）" % (kind,))


def feature_matching_loss(feats_real, feats_fake) -> torch.Tensor:
    """Feature matching：L1 对齐判别器中间特征（跨层取平均）。

    ``feats_real`` 应当来自 ``torch.no_grad()`` 下的 D(真值)，只作为回归目标。
    """
    assert len(feats_real) == len(feats_fake) and len(feats_real) > 0
    total = feats_real[0].new_zeros(())
    for fr, ff in zip(feats_real, feats_fake):
        total = total + F.l1_loss(ff, fr.detach())
    return total / len(feats_real)


def spectral_magnitude(x: torch.Tensor, log: bool = True) -> torch.Tensor:
    """对外暴露的谱幅度工具（画图/统计用），(B,C,L) -> (B,C,F)。"""
    mag = torch.fft.rfft(x, dim=-1).abs()
    return torch.log1p(mag) if log else mag


# ---------------------------------------------------------------------------
# 与 pipeline.py 旧命名 checkpoint 的互转 / 生成器加载
# ---------------------------------------------------------------------------
# pipeline.py::make_model 的命名：e1.c / e1.bn / e1.film.g / d1.c / d1.bn / d5
_LEGACY_KEY_RE = re.compile(r"^([ed])(\d+)\.(.*)$")


def remap_legacy_unet_keys(state_dict) -> dict:
    """把 ``pipeline.py`` 版 U-Net 的层名转成 ``models.PPGDenoiserUNet`` 的层名。

    已经是新命名的键原样保留（因此本函数是幂等的，可安全重复调用）。

    ``e{i}.c.*``   -> ``conv{i}.conv.*``      ``d{i}.c.*``   -> ``deconv{i}.deconv.*``
    ``e{i}.bn.*``  -> ``conv{i}.bn.*``        ``d{i}.bn.*``  -> ``deconv{i}.bn.*``
    ``e{i}.film.g.*``->``conv{i}.film.gamma.*``（``film.b`` -> ``film.beta``）
    ``d5.*``       -> ``deconv5.*``（末层 ConvTranspose1d，pipeline 里没有 bn/film）
    """
    out = {}
    for k, v in state_dict.items():
        m = _LEGACY_KEY_RE.match(k)
        if m is None:
            out[k] = v
            continue
        side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        if side == "e":
            prefix = "conv"
            rest = "conv." + rest[2:] if rest.startswith("c.") else rest
        else:
            prefix = "deconv"
            rest = "deconv." + rest[2:] if rest.startswith("c.") else rest
        rest = rest.replace("film.g.", "film.gamma.").replace("film.b.", "film.beta.")
        out["%s%d.%s" % (prefix, idx, rest)] = v
    return out


def load_generator_checkpoint(path: str, device="cpu", strict: bool = True):
    """加载已有 U-Net checkpoint，构造 ``PPGDenoiserUNet`` 并灌入权重。

    返回 ``(model, meta)``；``meta`` 是 checkpoint 里的元信息字典（in_channels /
    cond_dim / xkey / cond / stats2_mean / stats2_std / stats3_mean / stats3_std ...），
    训练脚本用它来决定读哪个特征键、条件向量怎么标准化。

    只加载 ``PPGDenoiserUNet`` 的**结构超参**（in_channels / cond_dim / use_film），
    其余键（xkey/cond/rls/seg_len/fs）原样透出。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # ⚠️ 残差模型（out = 物理先验 + Δ）不能走这里：本函数只加载基础网络、会**静默丢掉先验项**。
    # 这是本项目早期的大勘误（reports/08/09：DR 残差被误判为灾难性），必须响亮失败而不是静默出错。
    if bool(ckpt.get("residual", False)):
        raise RuntimeError(
            "%s 是**残差模型**（out = 物理先验 + Δ）；`gan.load_generator_checkpoint` 只加载基础网络，"
            "会静默丢掉物理先验项（历史勘误见 reports/08、09）。请改用 `residual.load()`。" % path)
    if "model" not in ckpt:
        raise KeyError("checkpoint %s 不含 'model' 键，实际键：%s" % (path, list(ckpt.keys())))

    in_channels = int(ckpt.get("in_channels", 1))
    cond_dim = int(ckpt.get("cond_dim", 0))
    model = PPGDenoiserUNet(in_channels=in_channels, cond_dim=cond_dim,
                            use_film=cond_dim > 0, norm=str(ckpt.get("norm", "bn")))
    missing, unexpected = model.load_state_dict(
        remap_legacy_unet_keys(ckpt["model"]), strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError("权重与 PPGDenoiserUNet 不匹配：missing=%s unexpected=%s"
                           % (missing, unexpected))
    model.to(device)

    meta = {k: v for k, v in ckpt.items() if k != "model"}
    meta["in_channels"] = in_channels
    meta["cond_dim"] = cond_dim
    meta["init_path"] = path
    meta["missing_keys"] = list(missing)
    meta["unexpected_keys"] = list(unexpected)
    return model, meta


# ---------------------------------------------------------------------------
# checkpoint 写出（严格限制为 Tensor + 内建类型）
# ---------------------------------------------------------------------------
_PLAIN_TYPES = (torch.Tensor, int, float, bool, str, type(None))


def assert_plain_checkpoint(obj, path: str = "root") -> None:
    """递归校验 ``obj`` 只含 Tensor / 内建标量 / dict / list / tuple。

    PyTorch >= 2.6 的 ``torch.load`` 默认 ``weights_only=True``，
    ``np.ndarray``、``argparse.Namespace``、自定义对象都会让之后加载直接抛错，
    所以宁可在写盘前就炸掉。``torch.Tensor`` 之外的 ``np`` 类型也会被拦下。
    """
    if isinstance(obj, _PLAIN_TYPES):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, (str, int)):
                raise TypeError("checkpoint 的字典键必须是 str/int，%s 处为 %r" % (path, type(k)))
            assert_plain_checkpoint(v, "%s[%r]" % (path, k))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_plain_checkpoint(v, "%s[%d]" % (path, i))
        return
    raise TypeError("checkpoint 不允许出现 %s（%s 处）：weights_only=True 加载会失败"
                    % (type(obj), path))


def make_gan_checkpoint(model: nn.Module, disc_temporal: nn.Module,
                        disc_spectral: nn.Module, meta: dict, extra: dict) -> dict:
    """组装 GAN checkpoint（训练脚本与手工保存共用，保证结构一致）。

    ``meta`` 里预期有 in_channels/cond_dim/xkey/cond/stats*_mean/stats*_std
    （来自 ``--init`` 的 checkpoint）；张量原样带过去，缺省用空张量占位。
    """
    payload = {
        "model": model.state_dict(),
        "disc_temporal": disc_temporal.state_dict(),
        "disc_spectral": disc_spectral.state_dict(),
        "in_channels": int(meta.get("in_channels", 1)),
        "cond_dim": int(meta.get("cond_dim", 0)),
        "xkey": str(meta.get("xkey", "XrlsOnly")),
        "cond": str(meta.get("cond", "none")),
        "init_path": str(meta.get("init_path", "")),
    }
    for k in ("stats2_mean", "stats2_std", "stats3_mean", "stats3_std"):
        v = meta.get(k)
        payload[k] = v if isinstance(v, torch.Tensor) else torch.zeros(0, 0)
    # extra 里的标量先做一次"降级"：numpy 标量 -> python 内建，其余交给 assert 拦截
    clean = {}
    for k, v in extra.items():
        if hasattr(v, "item") and not isinstance(v, (torch.Tensor, str, bytes)):
            try:
                v = v.item()
            except Exception:
                pass
        clean[k] = v
    payload.update(clean)
    return payload


def save_gan_checkpoint(path: str, payload: dict) -> None:
    """校验后保存；目录不存在则创建。"""
    assert_plain_checkpoint(payload)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    torch.save(payload, path)

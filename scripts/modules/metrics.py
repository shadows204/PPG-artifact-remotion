"""评价指标：波形恢复（需 clean 参考）+ 伪影抑制 + 心率。"""
from __future__ import annotations

import numpy as np

__all__ = ["prd", "pearson", "cosine", "snr_db", "snr_improvement",
           "hr_from_spectrum", "hr_errors", "band_energy_ratio"]


def _flat(a):
    return np.asarray(a, dtype=np.float64).reshape(-1)


def prd(clean, pred):
    """Percentage Root-mean-square Difference，越小越好（%）。"""
    c, p = _flat(clean), _flat(pred)
    if c.shape != p.shape:
        raise ValueError("prd: clean %s 与 pred %s 长度不一致（禁止静默广播）" % (c.shape, p.shape))
    denom = np.linalg.norm(c)
    if denom == 0:
        return float("nan")
    return float(100.0 * np.linalg.norm(p - c) / denom)


def pearson(clean, pred):
    c, p = _flat(clean), _flat(pred)
    if c.std() == 0 or p.std() == 0:
        return float("nan")
    return float(np.corrcoef(c, p)[0, 1])


def cosine(clean, pred):
    c, p = _flat(clean), _flat(pred)
    nc, np_ = np.linalg.norm(c), np.linalg.norm(p)
    if nc == 0 or np_ == 0:
        return float("nan")
    return float(np.dot(c, p) / (nc * np_))


def snr_db(signal, clean):
    """把 clean 当参考，signal 相对它的信噪比（dB）。"""
    s, c = _flat(signal), _flat(clean)
    noise = s - c
    pn = np.mean(noise ** 2)
    ps = np.mean(c ** 2)
    if pn <= 0:
        return float("inf")
    return float(10.0 * np.log10(ps / pn))


def snr_improvement(clean, noisy, pred):
    return snr_db(pred, clean) - snr_db(noisy, clean)


def _psd(sig, fs):
    sig = _flat(sig)
    sig = sig - sig.mean()
    n = len(sig)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * win)) ** 2
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, spec


def hr_from_spectrum(sig, fs=100.0, band=(0.7, 4.0)):
    """频域峰值法估心率（bpm）。band 为心率搜索带（Hz）。"""
    sig = _flat(sig)
    if not np.isfinite(sig).all():
        return float("nan")
    if sig.std() <= 0:                     # 常数输入：频谱全 0，argmax 会给出带内第一个 bin（历史坑）
        return float("nan")
    freq, spec = _psd(sig, fs)
    m = (freq >= band[0]) & (freq <= band[1])
    if not m.any() or spec[m].max() <= 0:
        return float("nan")
    f_peak = freq[m][np.argmax(spec[m])]
    return float(f_peak * 60.0)


def band_energy_ratio(sig, fs=100.0, motion_band=(2.5, 10.0), hr_band=(0.7, 2.5)):
    """运动频带能量 / 心率频带能量，反映运动伪影的相对强弱，越小越好。"""
    freq, spec = _psd(sig, fs)
    mm = (freq >= motion_band[0]) & (freq <= motion_band[1])
    mh = (freq >= hr_band[0]) & (freq <= hr_band[1])
    eh = spec[mh].sum()
    if eh <= 0:
        return float("inf")
    return float(spec[mm].sum() / eh)


def hr_errors(hr_true, hr_pred):
    """-> (MAE, RMSE, 误差>5bpm 的比例)"""
    t = np.asarray(hr_true, dtype=np.float64)
    p = np.asarray(hr_pred, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if len(t) == 0:
        return float("nan"), float("nan"), float("nan")
    d = p - t
    return (float(np.mean(np.abs(d))),
            float(np.sqrt(np.mean(d ** 2))),
            float(np.mean(np.abs(d) > 5.0)))

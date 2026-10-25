#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成用于冒烟测试的合成输入（含参考真值），供打包/CI 冒烟测试用。

用法：python webapp/packaging/gen_test_npz.py [out.npz]
"""
import sys

import numpy as np


def main(path: str = "test_input.npz") -> str:
    fs, n = 100.0, 3000
    t = np.arange(n) / fs
    hr = 70.0 / 60.0
    clean = (80.0 * np.sin(2 * np.pi * hr * t)
             + 25.0 * np.sin(2 * np.pi * 2 * hr * t + 0.8))
    motion = 40.0 * np.sin(2 * np.pi * 3.0 * t) + 18.0 * np.sin(2 * np.pi * 1.5 * t + 0.4)
    rng = np.random.RandomState(0)
    ppg = clean + motion + 3.0 * rng.randn(n)
    np.savez_compressed(
        path,
        ppg=ppg,
        acc=(motion / 20.0)[None, :],
        gyro=np.roll(motion, 20)[None, :] / 50.0,
        reference=clean,
        fs=fs,
    )
    print("已生成 %s（%d 点 @ %g Hz，含 reference）" % (path, n, fs))
    return path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_input.npz")

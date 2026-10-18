#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按指定 α（RLS 收缩系数）重生成 CIME 特征（物理先验强度消融用）。

`ablation_cime` 显示 α=0.60（异质两路）时物理先验达 **79.00 PRD**，而主流程特征用的是 α=0.40（83.78）
→ 需要验证"更好的先验起点是否带来更好的最终结果"。

用法：PY=... $PY scripts/tools/make_features_alpha.py --alpha 0.6 --out outputs/alpha60
"""
import argparse, importlib.util, os, sys, time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); SCRIPTS=os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS); sys.path.insert(0, os.path.join(SCRIPTS,"modules"))
from _common import paths as P

def lm(n,p):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

ap=argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--taps", type=int, default=4)
ap.add_argument("--lam", type=float, default=0.995,
                    help="与权威特征一致（历史默认 0.99 会混入 λ 变化）")
ap.add_argument("--out", required=True)
a=ap.parse_args()
pl=lm("pl", os.path.join(SCRIPTS,"pipeline","pipeline.py"))
os.makedirs(os.path.join(a.out,"features"), exist_ok=True)
t0=time.time()
for name in ("train","test"):
    d=np.load(os.path.join(pl.DATA_DIR, "cime_%s.npz" % name))
    feats=pl.build_features(d["noisy"], d["clean"], a.alpha, a.taps, a.lam, verbose=True)
    feats["subject"]=pl.subject_ids(d, name)
    np.savez_compressed(os.path.join(a.out,"features","%s.npz"%name), **feats)
    y=feats["y"]; base=feats["XrlsOnly"][:,2]
    prd=float(np.mean([100*np.linalg.norm(base[i]-y[i])/np.linalg.norm(y[i]) for i in range(len(y))]))
    print("[feat] %-5s X=%s | 物理先验 PRD（本 α）%.2f" % (name, feats["X"].shape, prd))
print("[feat] α=%.2f 完成，用时 %.0f s -> %s" % (a.alpha, time.time()-t0, a.out))

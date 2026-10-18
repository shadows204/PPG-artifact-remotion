#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 30 s 特征按时间切成更短/拼接成更长的段（段长消融用）。

原理：特征各通道都是**逐采样**的（PPG / M̂ / PPG_RLS / ACC_x / Gyro_y 与真值 y），
所以在时间维切片是合法的；切完按"该段含噪 PPG 的 std"重新标准化（与 `build_features` 的
每段标准化约定一致）。PRD 对整体尺度不变，故缩放不影响指标。

用法：
    $PY scripts/tools/make_seg_features.py --seconds 15 --out outputs/seg15
"""
import argparse, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); SCRIPTS=os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
from _common import paths as P

ap=argparse.ArgumentParser()
ap.add_argument("--src", default=P.artifact("final_10seed/features"))
ap.add_argument("--seconds", type=float, required=True)
ap.add_argument("--fs", type=float, default=100.0)
ap.add_argument("--out", required=True)
a=ap.parse_args()
L=int(round(a.seconds*a.fs))
os.makedirs(os.path.join(a.out,"features"), exist_ok=True)
for split in ("train","test"):
    d=np.load(os.path.join(a.src,"%s.npz"%split))
    X,Y=d["X"],d["y"]; n_full=X.shape[-1]
    n_piece=n_full//L
    Xs=[];Ys=[];Ss=[];subj=[]
    for k in range(n_piece):
        s=k*L
        Xk=X[...,s:s+L].copy(); Yk=Y[...,s:s+L].copy()
        # 按该段含噪 PPG 的 std 重标准化（ch0 是含噪 PPG）
        sd=Xk[:,0,:].std(axis=1,keepdims=False).astype(np.float32)+1e-8
        Xk=Xk/sd[:,None,None]; Yk=Yk/sd[:,None]
        Xs.append(Xk); Ys.append(Yk)
        if "subject" in d.files: subj.append(d["subject"])
    Xo=np.concatenate(Xs,0); Yo=np.concatenate(Ys,0)
    out=dict(X=Xo, y=Yo)
    for key in ("Xppg","XnoRLS","XrlsOnly"):
        if key in d.files:
            v=d[key]; pieces=[]
            for k in range(n_piece):
                s=k*L; vk=v[...,s:s+L].copy()
                sd=X[...,s:s+L][:,0,:].std(axis=1)+1e-8
                pieces.append(vk/sd[:,None,None])
            out[key]=np.concatenate(pieces,0)
    for key in ("stats2","stats3","scale"):
        if key in d.files: out[key]=np.repeat(d[key], n_piece, axis=0)
    if subj: out["subject"]=np.concatenate(subj,0)
    np.savez_compressed(os.path.join(a.out,"features","%s.npz"%split), **out)
    # 物理先验 PRD 自检
    prd=float(np.mean([100*np.linalg.norm(out["XrlsOnly"][i,2]-out["y"][i])/np.linalg.norm(out["y"][i])
                       for i in range(0,len(out["y"]), max(1,len(out["y"])//200))]))
    print("[seg] %-5s %s → X=%s | 物理先验 PRD 抽样 %.2f" % (split, X.shape, out["X"].shape, prd))
print("[seg] %.0f s 段完成 -> %s" % (a.seconds, a.out))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记录级多进程工具（本机 2×EPYC 7T83 = 128 物理核 / 503 GB 内存）。

为什么必须用**进程**而不是线程
------------------------------
`reports/15` §5 实测：本项目最贵的算子是 `rls_decompose`（逐样本 Python 循环）——
OMP=1 与 OMP=8 耗时 0.9490 s vs 0.9478 s，**完全不缩放**；数据加载是 I/O，同样不缩放。
所以唯一有效的加速是**按记录切块、多进程**（记录之间天然独立、无共享状态）。

设计要点
--------
1. **按块（chunk）而非按条**分配：worker 内部把数据集缓存到模块全局，
   因此**加载只发生一次/进程**（而不是一条一次）；
2. `worker(chunk) -> list`，`chunk` 只需含可 pickle 的标识（数据集名 + 记录下标 + 参数），
   **由 worker 自己加载数据** —— 这顺带把加载也并行化了；
3. `threads=1`：worker 内把 OMP/MKL/torch 线程数压到 1，避免 128 进程 × 多线程互相踩；
4. 提供 `fork`（纯 CPU，最快）与 `spawn`（带 CUDA 的工具更安全）两种启动方式；
5. 任何时候都能退回串行（`workers=1`），便于对拍验证。

用法
----
    from _common import parallel as PAR
    chunks = PAR.split_chunks(list(range(53)), PAR.default_workers(53))
    res, dt = PAR.pmap(_worker, chunks, desc="bidmc 53 条")

    def _worker(idx_chunk):          # 必须是模块顶层函数（可 pickle）
        import sys; sys.path.insert(0, ...)
        ...                          # 自行 import + 加载需要的记录
        return [dict(...), ...]
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

__all__ = ["default_workers", "split_chunks", "pmap", "log_env"]


def log_env(log=print):
    log("[par] cpu=%s 内存可用=%s GB 已用负载=%s"
        % (os.cpu_count(),
           _mem_available_gb(),
           tuple(round(x, 2) for x in os.getloadavg())))


def _mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return round(int(line.split()[1]) / 1e6, 1)
    except Exception:
        pass
    return "?"


def default_workers(n_items: int, reserve: int = 2, cap: int | None = None) -> int:
    """默认 worker 数：留 reserve 个核给主进程/系统，且不超过块数。"""
    cpu = os.cpu_count() or 4
    w = max(1, min(int(n_items), cpu - reserve))
    return min(w, cap) if cap else w


def split_chunks(items, n_chunks: int):
    """把 items 尽量等分成 n_chunks 块。"""
    items = list(items)
    n_chunks = max(1, min(int(n_chunks), len(items) or 1))
    k, m = divmod(len(items), n_chunks)
    out, i = [], 0
    for j in range(n_chunks):
        sz = k + (1 if j < m else 0)
        if sz:
            out.append(items[i:i + sz])
        i += sz
    return out


def _init_worker(threads: int, nice: int = 0, use_torch: bool = False):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(threads)
    # 注意：torch 的 import 约 1.5 s，**只在确实需要时导入**——否则每个 worker 都白付这笔钱，
    # 会把并行加速比吃光（实测：16 进程自检 1.59 s vs 串行 1.60 s，就是这个原因）。
    if use_torch:
        try:
            import torch
            torch.set_num_threads(threads)
        except Exception:
            pass
    if nice:
        try:
            os.nice(nice)
        except Exception:
            pass


def _flatten_parts(parts):
    """把各块返回值合并成一条结果列表。

    ⚠️ 只展平 **list/tuple**；worker 若返回 **dict**（或其他对象）则原样作为**一条结果**保留。
    早期版本无条件 `for r in p`，导致返回 dict 时被拆成"键"（本项目多次踩坑），已修正。
    """
    out = []
    for p in parts:
        if p is None:
            continue
        if isinstance(p, (list, tuple)):
            out.extend(p)
        else:
            out.append(p)
    return out


def pmap(worker, chunks, workers: int | None = None, threads: int = 1,
         spawn: bool = False, desc: str = "", log=print, use_torch: bool = False):
    """并行执行 ``worker(chunk) -> list | dict | 任意对象``，返回 ``(合并结果, 秒数)``。

    worker 返回 **list/tuple** 时会被展平成一条条结果；返回 **dict** 或其他对象时**原样保留为一条**。

    spawn=False（默认）用 fork：启动快，适合纯 CPU 工具；
    spawn=True 用 spawn：子进程重新 import，适合内部建 CUDA 上下文的工具。
    """
    chunks = list(chunks)
    workers = int(workers or default_workers(len(chunks)))
    t0 = time.perf_counter()
    if workers <= 1 or len(chunks) <= 1:
        res = _flatten_parts(worker(ch) for ch in chunks)
        mode = "串行"
    else:
        ctx = mp.get_context("spawn" if spawn else "fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_init_worker,
                                 initargs=(threads, 0, use_torch)) as ex:
            parts = list(ex.map(worker, chunks))
        res = _flatten_parts(parts)
        mode = "%d 进程%s" % (workers, "（spawn）" if spawn else "（fork）")
    dt = time.perf_counter() - t0
    if desc:
        log("  [par] %-28s %s：%d 块 / %d 条结果 / %.1f s"
            % (desc, mode, len(chunks), len(res), dt))
    return res, dt


def _selfcheck_work(chunk):
    """自检 worker（必须是模块顶层函数才能 pickle）。"""
    import time as _t
    _t.sleep(0.004 * len(chunk))        # 单位成本正比于块长 → 总工作量与切块数无关
    return [(i, i * i) for i in chunk]


def bench_selfcheck():
    """自检：并行结果与串行逐位一致 + 加速比。"""
    work = _selfcheck_work
    n = 400
    items = list(range(n))
    ser, t_ser = pmap(work, split_chunks(items, 1), workers=1, desc="自检-串行")
    par, t_par = pmap(work, split_chunks(items, 16), workers=16, desc="自检-16 进程")
    assert sorted(ser) == sorted(par), "并行结果与串行不一致"
    # 回归：worker 返回 dict 时必须原样保留（不得被拆成键）
    dser, _ = pmap(lambda ch: dict(keys=list(ch)), split_chunks(items[:4], 1), workers=1)
    assert dser == [dict(keys=list(range(4)))], dser
    print("[par] 自检通过：结果逐位一致；串行 %.2fs → 16 进程 %.2fs（%.1f×）"
          % (t_ser, t_par, t_ser / max(t_par, 1e-9)))


if __name__ == "__main__":
    log_env()
    bench_selfcheck()

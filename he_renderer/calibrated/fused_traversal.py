"""Opt-in CUDA prototype; never selected by the runtime renderer automatically."""
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'runtime/python'))
os.environ.setdefault('CUPY_CACHE_DIR', str(ROOT.parent/'cache/cupy'))
os.environ.setdefault('CUDA_PATH', str(ROOT.parent/'runtime/cuda_headers/nvidia/cuda_runtime'))
_dll = os.add_dll_directory(str(Path(torch.__file__).parent/'lib')) if os.name == 'nt' else None
import cupy as cp

_kernel = None


def intersect(hull, origin, directions):
    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel((ROOT/'fused_traversal.cu').read_text(), 'march',
                              options=('--fmad=false',))
    # Synchronize explicitly across libraries; this prototype favors correctness.
    torch.cuda.synchronize()
    volume = cp.from_dlpack(hull.volume)
    rays = cp.asarray(directions.reshape(-1, 3), dtype=cp.float32)
    start = cp.asarray(origin, dtype=cp.float32)
    low, high = cp.asarray(hull.low, dtype=cp.float32), cp.asarray(hull.high, dtype=cp.float32)
    points = cp.empty_like(rays)
    hits = cp.empty(len(rays), dtype=cp.uint8)
    nz, ny, nx = hull.volume.shape[-3:]
    _kernel(((len(rays)+127)//128,), (128,), (volume, rays, start, low, high,
        np.float32(hull.spacing), np.int32(nx), np.int32(ny), np.int32(nz),
        np.int32(len(rays)), points, hits))
    return cp.asnumpy(points).reshape(directions.shape), cp.asnumpy(hits).astype(bool).reshape(directions.shape[:2])

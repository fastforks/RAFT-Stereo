#!/usr/bin/env python
"""RAFT-Stereo, general fast inference: two rectified images of any size in, disparity out. No per-size compilation,
autotuning or warm-up (see core/general.py).

    python raft_fast.py left.png right.png out.npy            # fp32 disparity [H, W] in pixels (left view)
    python raft_fast.py left.png right.png out.png            # 16-bit PNG, disparity * 256 (KITTI convention)
    python raft_fast.py left.png right.png out.pfm            # PFM (Middlebury convention, float32)
    options: --ckpt models/raftstereo-middlebury.pth  --iters 32

Python API:
    from raft_fast import load_model
    model = load_model("models/raftstereo-middlebury.pth")
    disp = model(left, right)        # uint8 [H, W, 3] numpy/torch -> torch fp32 [H, W] on the GPU
"""

import argparse
import os
import sys
import threading
import time

_T0 = time.perf_counter()
# fewer reserved-but-unused bytes when consecutive images have different sizes (set before torch initializes CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# v8: fused-epilogue convs (GRU + K>=128 conv/BN/act) from the shipped CUTLASS library; RAFT_FUSED=0 gives v7
os.environ.setdefault("RAFT_FUSED", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(_HERE, "models", "raftstereo-middlebury.pth")


def load_model(ckpt=DEFAULT_CKPT, iters=32):
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from core.general import GeneralRAFT

    return GeneralRAFT(ckpt, iters=iters)


def read_image(path):
    import numpy as np
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def write_disp(path, disp):
    import numpy as np

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, disp)
    elif ext == ".png":
        from PIL import Image

        Image.fromarray(np.clip(disp * 256.0 + 0.5, 0, 65535).astype(np.uint16)).save(path)
    elif ext == ".pfm":
        h, w = disp.shape
        with open(path, "wb") as f:
            f.write(f"Pf\n{w} {h}\n-1.0\n".encode())
            f.write(np.ascontiguousarray(disp[::-1], dtype="<f4").tobytes())
    else:
        raise SystemExit(f"unknown output format {ext} (use .npy, .png or .pfm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("out")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--timing", action="store_true", help="print phase times to stderr")
    a = ap.parse_args()

    imgs = {}
    reader = threading.Thread(
        target=lambda: imgs.update(l=read_image(a.left), r=read_image(a.right))
    )
    reader.start()  # PNG decoding overlaps torch import + CUDA init

    t_import = time.perf_counter()
    model = load_model(a.ckpt, a.iters)
    t_model = time.perf_counter()
    reader.join()
    disp = model(imgs["l"], imgs["r"]).cpu().numpy()
    t_run = time.perf_counter()
    write_disp(a.out, disp)
    if a.timing:
        print(
            f"import {t_import - _T0:.3f} s, model {t_model - t_import:.3f} s, inference {t_run - t_model:.3f} s, "
            f"write {time.perf_counter() - t_run:.3f} s, total {time.perf_counter() - _T0:.3f} s "
            f"({model.convs.n_built} conv plans built)",
            file=sys.stderr,
        )
    sys.stdout.flush()
    os._exit(0)  # skip interpreter / CUDA teardown


if __name__ == "__main__":
    main()

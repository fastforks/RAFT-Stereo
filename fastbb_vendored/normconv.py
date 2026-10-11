# vendored from fastforks/fastbb: fastbb/normconv.py @ aa2e3f3 -- do not edit here, see fastbb/VENDORED.md
"""Vendored CUTLASS convolution/BatchNorm epilogues for the RAFT context encoder."""
import hashlib
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "csrc", "norm_conv.cu")
CUTLASS = os.environ.get("CUTLASS_DIR", "")   # CUTLASS >= 4.x checkout (only needed to build the torch-op .so; see README)
_loaded = [False]


def load():
    if _loaded[0]:
        return True
    abi = f"{torch.__version__}:{torch.version.cuda}:{torch._C._GLIBCXX_USE_CXX11_ABI}"
    tag = hashlib.sha1(open(SRC, "rb").read() + abi.encode()).hexdigest()[:12]
    bdir = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "fastbb", f"normconv_{tag}")
    so = os.path.join(bdir, f"fastbb_normconv_{tag}.so")
    if os.path.exists(so):  # cached: no compiler / CUTLASS needed
        torch.ops.load_library(so)
        _loaded[0] = True
        return True
    if not CUTLASS or not os.path.isdir(os.path.join(CUTLASS, "include")):
        raise RuntimeError("no prebuilt torch-op library for this torch/CUDA build: set CUTLASS_DIR to a CUTLASS >= 4.x checkout and CUDA_HOME to a CUDA 12.8+/13 toolkit (nvcc) to build it, see README")
    os.makedirs(bdir, exist_ok=True)
    from torch.utils.cpp_extension import load as _load
    libdir = os.path.join(bdir, "lib")
    os.makedirs(libdir, exist_ok=True)
    cudart = os.path.join(os.environ.get("CUDA_HOME", ""), "lib", "libcudart.so.13")
    if os.path.exists(cudart) and not os.path.exists(os.path.join(libdir, "libcudart.so")):
        os.symlink(cudart, os.path.join(libdir, "libcudart.so"))
    arch = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in (80, 86, 89, 120)] + ["-gencode=arch=compute_80,code=compute_80"]
    _load(name=f"fastbb_normconv_{tag}", sources=[SRC], build_directory=bdir, verbose=False, is_python_module=False,
          extra_include_paths=[os.path.join(CUTLASS, "include")], extra_ldflags=[f"-L{libdir}"],
          extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-U__CUDA_NO_HALF_OPERATORS__",
                             "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", *arch])
    _loaded[0] = True
    return True


def ok(x, w):
    """Shapes the kernels take (else: caller's fallback)."""
    return (x.dtype == torch.half and w.dtype == torch.half and x.shape[1] % 8 == 0 and w.shape[0] % 8 == 0
            and w.shape[1] == x.shape[1])


def conv(x, w, scale=None, shift=None, res=None, act=0, res_mode=0, padding=None, stride=1, cfg=-1):
    load()
    R = w.shape[2]
    p = [R // 2, w.shape[3] // 2] if padding is None else ([padding] * 2 if isinstance(padding, int) else list(padding))
    s = [stride] * 2 if isinstance(stride, int) else list(stride)
    ps, pb = None, None
    return torch.ops.fastbb_nc.conv(x, w, scale, shift, res, ps, pb, p, s, act, res_mode, False, cfg)

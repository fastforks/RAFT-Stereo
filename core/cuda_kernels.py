"""The Rust engine's custom CUDA kernels (core/raft_kernels.cu) for PyTorch: compiled once to a cubin with the nvcc
of the active CUDA toolkit (CUDA_HOME), loaded through the driver API and launched on torch's current stream (so
they work under torch.cuda.graph capture like any other kernel).

    k = Kernels.get()
    k.launch("init_coords", k.linear(P), coords, W, P)          # argument types from SIGNATURES
"""

import ctypes
import hashlib
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raft_kernels.cu")
_PREBUILT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prebuilt"
)  # shipped cubins (no nvcc needed)
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "raft-stereo-torch")

# p: pointer (tensor or int address), i: int32, l: int64, f: float32, u: uint32
SIGNATURES = {
    "prep_image": "ppiiiiiii",
    "in_stats_partial": "pliip",
    "in_stats_final": "piilifp",
    "in_apply": "ppppplii",
    "corr_pool": "pplii",
    "corr_lookup": "ppppiiiippl",
    "pool2x_into": "piiipiiii",
    "interp_into": "piiipiiii",
    "write_flow": "pilpi",
    "convf1_direct": "piippp",
    "gru_gate1": "ppipl",
    "prep_context": "ppppppl",
    "init_coords": "pil",
    "upsample_out": "ppiiiiiipi",
    "flow_head_out": "piippp",
    "fill_rand": "plu",
    "corr_gemm": "pppif",
    "affine_act": "pipipipppilii",
    "gru_zr": "ppppil",
    "pool2x_dual": "piiippiiii",
    "interp_dual": "piiippiiii",
    "gru_update": "ppppipl",
}
_CTYPE = {"i": ctypes.c_int32, "l": ctypes.c_int64, "f": ctypes.c_float, "u": ctypes.c_uint32}


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"CUDA driver error {res} in {what}")


def _nvcc(arch_args, out):
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    subprocess.run(
        [
            os.path.join(cuda_home, "bin", "nvcc"),
            *arch_args,
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{cuda_home}/include",
            "-o",
            out + ".tmp",
            _SRC,
        ],
        check=True,
    )
    os.replace(out + ".tmp", out)


def build_prebuilt(archs=(80, 86, 89, 90, 120)):
    """Build the shipped fatbin for the current kernel source (run after editing raft_kernels.cu)."""
    src = open(_SRC, "rb").read()
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        if f.startswith("raft_kernels_"):
            os.remove(os.path.join(_PREBUILT, f))
    out = os.path.join(_PREBUILT, f"raft_kernels_{hashlib.sha1(src).hexdigest()[:16]}.fatbin")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + [
        "-gencode=arch=compute_80,code=compute_80"
    ]
    _nvcc(["-fatbin", *gencode], out)
    return out


class Kernels:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        src = open(_SRC, "rb").read()
        # shipped fatbin (sm_80..sm_120 SASS + compute_80 PTX), else a per-arch cubin compiled once with nvcc
        cubin = os.path.join(_PREBUILT, f"raft_kernels_{hashlib.sha1(src).hexdigest()[:16]}.fatbin")
        if not os.path.exists(cubin):
            tag = hashlib.sha1(src + f"sm_{major}{minor}".encode()).hexdigest()[:16]
            cubin = os.path.join(_CACHE, f"raft_kernels_{tag}.cubin")
        if not os.path.exists(cubin):
            os.makedirs(_CACHE, exist_ok=True)
            _nvcc(["-cubin", f"-arch=sm_{major}{minor}"], cubin)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_uint] * 7
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        )
        torch.zeros(1, device="cuda")  # make sure the primary context is current
        self.module = ctypes.c_void_p()
        self._image = open(cubin, "rb").read()
        _check(
            self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData"
        )
        self.funcs = {}
        for name in SIGNATURES:
            f = ctypes.c_void_p()
            _check(
                self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()),
                f"cuModuleGetFunction {name}",
            )
            self.funcs[name] = f

    @staticmethod
    def linear(n):
        """Rust's Launch::linear: 256 threads, grid-stride loops, at most 65535*8 blocks."""
        return (min(max((n + 255) // 256, 1), 65535 * 8), 1, 1), (256, 1, 1)

    def launch(self, name, cfg, *args):
        grid, block = cfg
        sig = SIGNATURES[name]
        assert len(sig) == len(args), f"{name}: expected {len(sig)} args, got {len(args)}"
        holders = []
        for t, a in zip(sig, args):
            if t == "p":
                holders.append(
                    ctypes.c_void_p(
                        a.data_ptr() if isinstance(a, torch.Tensor) else (int(a) or None)
                    )
                )
            else:
                holders.append(_CTYPE[t](a))
        arr = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
        )
        stream = torch.cuda.current_stream().cuda_stream
        _check(
            self.lib.cuLaunchKernel(self.funcs[name], *grid, *block, 0, stream, arr, None),
            f"launch {name}",
        )


if __name__ == "__main__":
    print(build_prebuilt())

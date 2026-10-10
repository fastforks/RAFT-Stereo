"""fp16 convolution with fp16 accumulation ("true half") through cuDNN graphs, conv + bias fused.

PyTorch always asks cuDNN for fp32 accumulation. On GeForce cards that runs the tensor cores at half rate
(RTX 5060 Ti: ~47 vs ~80 TFLOPS). RAFT-Stereo keeps AMP-level accuracy with fp16 accumulation (measured with
an im2col emulation), so this op builds one cuDNN graph per (shape, stride, padding) with
compute_data_type=HALF, times all execution plans once (plus torch's own conv as a candidate) and caches the
fastest. Registered as a torch custom op so torch.compile treats it as an opaque kernel.

    model.use_cudnn_half_acc_convs()   # see RAFTStereo
"""

import os
import re
import statistics
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cudnn
except ImportError:  # optional dependency (pip install nvidia-cudnn-frontend)
    cudnn = None

_handle = None
_cache = {}
_stats = {"cudnn": 0, "torch": 0, "rejected": 0}
# cuDNN 9.x on sm_120 (found while porting to Rust, DESIGN_FLAWS.md D9): engine 8 can crash the process, and some
# plans return wrong values without an error (engine 1 with fused epilogues). Never trust a plan only because it
# runs fastest: deny-list engine 8 and compare every candidate's output with an fp32 reference before timing it.
DENY_ENGINES = {8}
VERIFY_TOL = 0.05  # max |y - ref| / max |ref|; fp16 accumulation is ~1e-3, broken plans are O(1)


def _get_handle():
    global _handle
    if _handle is None:
        _handle = cudnn.create_handle()
    cudnn.set_stream(handle=_handle, stream=torch.cuda.current_stream().cuda_stream)
    return _handle


def _out_hw(x, w, padding, stride):
    return (
        (x.shape[2] + 2 * padding[0] - w.shape[2]) // stride[0] + 1,
        (x.shape[3] + 2 * padding[1] - w.shape[3]) // stride[1] + 1,
    )


def _time(fn, warmup=2, runs=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def _engine(g, i):
    m = re.match(r"eng(\d+)", g.get_plan_name_at_index(i))
    return int(m.group(1)) if m else -1


def _deviation(y, ref):
    d = (y.float() - ref).abs().max()
    return float("inf") if torch.isnan(d) else float(d / (ref.abs().max() + 1e-3))


def _build(x, w, b, padding, stride):
    """Returns ('cudnn', graph, tensors, plan, ws_size) or ('torch',) if torch's conv is faster or cuDNN fails."""
    H16 = cudnn.data_type.HALF
    N, K = x.shape[0], w.shape[0]
    Ho, Wo = _out_hw(x, w, padding, stride)
    heuristic = os.environ.get("RAFT_TUNE") == "heuristic"  # first verified cuDNN plan, no timing
    torch_ms = float("inf") if heuristic else _time(lambda: F.conv2d(x, w, b, stride, padding))
    try:
        h = _get_handle()
        g = cudnn.pygraph(
            io_data_type=H16, intermediate_data_type=H16, compute_data_type=H16, handle=h
        )
        X = g.tensor(name="X", dim=list(x.shape), stride=list(x.stride()), data_type=H16)
        W = g.tensor(name="W", dim=list(w.shape), stride=list(w.stride()), data_type=H16)
        Y = g.conv_fprop(
            image=X,
            weight=W,
            padding=list(padding),
            stride=list(stride),
            dilation=[1, 1],
            compute_data_type=H16,
        )
        B = None
        if b is not None:
            B = g.tensor(name="B", dim=[1, K, 1, 1], stride=[K, 1, K, K], data_type=H16)
            Y = g.bias(input=Y, bias=B)
        Y.set_output(True).set_data_type(H16).set_dim([N, K, Ho, Wo]).set_stride(
            [Ho * Wo * K, 1, Wo * K, K]
        )
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        if (
            os.environ.get("RAFT_TUNE") != "heuristic"
        ):  # heuristic mode builds plans lazily (runtime-compiled ones cost ~0.3 s each)
            g.build_plans(cudnn.build_plan_policy.ALL)
    except Exception:  # noqa: BLE001 - no cuDNN half plan for this shape (e.g. 2 or 3 input channels)
        return ("torch",)
    y = torch.empty(
        (N, K, Ho, Wo), device=x.device, dtype=x.dtype, memory_format=torch.channels_last
    )
    # verification input: random values in [-1, 1] (activations of the real call could be all zero in places)
    gen = torch.Generator(device=x.device).manual_seed(0)
    xr = (
        (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1)
        .to(x.dtype)
        .contiguous(memory_format=torch.channels_last)
    )
    ref = F.conv2d(xr.float(), w.float(), None if b is None else b.float(), stride, padding)
    best = None
    for i in range(g.get_execution_plan_count()):
        if _engine(g, i) in DENY_ENGINES:
            continue
        try:
            if heuristic:
                g.build_plan_at_index(i)
            ws = torch.empty(
                max(g.get_workspace_size_plan_at_index(i), 1), device=x.device, dtype=torch.uint8
            )
            y.fill_(float("nan"))
            g.execute_plan_at_index(
                {X: xr, W: w, Y: y} if B is None else {X: xr, W: w, B: b, Y: y}, ws, i, handle=h
            )
            dev = _deviation(y, ref)
            if dev > VERIFY_TOL:
                _stats["rejected"] += 1
                print(
                    f"[conv_half_acc] rejected plan {g.get_plan_name_at_index(i)} for x{tuple(x.shape)} w{tuple(w.shape)}: deviation {dev:.3g}"
                )
                continue
            pack = {X: x, W: w, Y: y} if B is None else {X: x, W: w, B: b, Y: y}
            ms = 0.0 if heuristic else _time(lambda: g.execute_plan_at_index(pack, ws, i, handle=h))  # noqa: B023 - _time invokes this callback synchronously
        except Exception:  # noqa: BLE001
            continue
        if best is None or ms < best[0]:
            best = (ms, i, g.get_workspace_size_plan_at_index(i))
        if heuristic:
            break
    if best is None or best[0] >= torch_ms:
        return ("torch",)
    return ("cudnn", g, X, W, B, Y, best[1], best[2])


# needs_exact_strides: under torch.compile, Inductor would otherwise pick its own (NCHW) layout for the inputs of
# this opaque op, and every call would pay an NCHW->NHWC copy (measured: 44 ms of copies at KITTI size)
@torch.library.custom_op(
    "raft::conv_half_acc", mutates_args=(), tags=(torch.Tag.needs_exact_strides,)
)
def conv_half_acc(
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    padding: List[int],
    stride: List[int],
) -> torch.Tensor:
    x = x.contiguous(memory_format=torch.channels_last)
    key = (
        tuple(x.shape),
        tuple(w.shape),
        tuple(w.stride()),
        b is not None,
        tuple(padding),
        tuple(stride),
        x.device.index,
    )
    entry = _cache.get(key)
    if entry is None:
        entry = _cache[key] = _build(x, w, b, padding, stride)
    if entry[0] == "torch":
        _stats["torch"] += 1
        return F.conv2d(x, w, b, stride, padding).contiguous(memory_format=torch.channels_last)
    _stats["cudnn"] += 1
    _, g, X, W, B, Y, plan, ws_size = entry
    N, K = x.shape[0], w.shape[0]
    Ho, Wo = _out_hw(x, w, padding, stride)
    y = torch.empty(
        (N, K, Ho, Wo), device=x.device, dtype=x.dtype, memory_format=torch.channels_last
    )
    ws = torch.empty(max(ws_size, 1), device=x.device, dtype=torch.uint8)
    pack = {X: x, W: w, Y: y} if B is None else {X: x, W: w, B: b, Y: y}
    g.execute_plan_at_index(pack, ws, plan, handle=_get_handle())
    return y


@conv_half_acc.register_fake
def _(x, w, b, padding, stride):
    Ho, Wo = _out_hw(x, w, padding, stride)
    return x.new_empty((x.shape[0], w.shape[0], Ho, Wo)).contiguous(
        memory_format=torch.channels_last
    )


def _forward(self, x):
    return torch.ops.raft.conv_half_acc(
        x, self.weight, self.bias, list(self.padding), list(self.stride)
    )


def patch_convs(module):
    """Route every fp16 nn.Conv2d (groups=1, dilation=1) of `module` through conv_half_acc. Returns the count."""
    assert cudnn is not None, "pip install nvidia-cudnn-frontend"
    n = 0
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            and m.groups == 1
            and m.dilation == (1, 1)
            and m.weight.dtype == torch.half
        ):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            m.forward = _forward.__get__(m)
            n += 1
    return n


def _instance_norm_nhwc(self, x):
    # same math as F.instance_norm (affine=False, no running stats), statistics in fp32; elementwise ops keep NHWC
    xf = x.float()
    var, mean = torch.var_mean(xf, dim=(2, 3), keepdim=True, unbiased=False)
    return ((xf - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)


def _batch_norm_eval_nhwc(self, x):
    # eval-mode BatchNorm = per-channel affine; computed from fp32 statistics, applied in x's dtype
    scale = self.weight.float() * torch.rsqrt(self.running_var.float() + self.eps)
    shift = self.bias.float() - self.running_mean.float() * scale
    return x * scale.to(x.dtype).view(1, -1, 1, 1) + shift.to(x.dtype).view(1, -1, 1, 1)


def patch_norms(module):
    """PyTorch's batch/instance norm kernels return NCHW even for NHWC input, so every following conv paid an
    NCHW->NHWC copy. Replace them (inference only) by layout-preserving elementwise implementations."""
    n = 0
    for m in module.modules():
        if isinstance(m, nn.InstanceNorm2d) and not m.affine and not m.track_running_stats:
            m.forward = _instance_norm_nhwc.__get__(m)
            n += 1
        elif isinstance(m, nn.BatchNorm2d) and not m.training:
            m.forward = _batch_norm_eval_nhwc.__get__(m)
            n += 1
    return n

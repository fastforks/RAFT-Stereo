"""cuDNN convolutions with fused epilogues on channel slices (port of the Rust engine's src/cudnn.rs).

A FusedConv computes   y = epi_n(... epi_1(conv(x, w)))   with fp16 I/O and fp16 accumulation, where x, y and the
epilogue operands are NHWC channel *slices* of wider buffers ([H, W, Ctot] tensors, a slice = channel offset + width).
Epilogue steps: ("bias", b) / ("scale", s) per-channel vectors, ("add"|"sub"|"mul", slice) tensors, ("relu",),
("sigmoid",), ("tanh",).

Writing into slices removes every torch.cat of the update loop, and the fused epilogues remove the elementwise
kernels around the convolutions (the whole GRU state update h + z(q - h) runs in the epilogue of the q-conv).

Plan selection = Rust's tuner: all plans of heuristics A + fallback; engine 8 is skipped (crashes on sm_120); every
candidate is run on random inputs and compared with an fp32 PyTorch reference first (cuDNN engine 1 silently returns
wrong values for long epilogues on sm_120, DESIGN_FLAWS.md D9); the fastest verified plan wins. Winners are cached
in ~/.cache/raft-stereo-torch/fused-plans.tsv (by plan name; the cached plan is re-verified on load).
The output must not alias an epilogue operand (cuDNN does not support it: DESIGN_FLAWS.md D8) -- asserted.
"""

import os
import re
import statistics

import cudnn
import torch
import torch.nn.functional as F

DENY_ENGINES = {8}
VERIFY_TOL = 0.05
_CACHE_FILE = os.path.join(
    os.path.expanduser("~"), ".cache", "raft-stereo-torch", "fused-plans.tsv"
)
_handle = None
_plan_cache = None
stats = {"built": 0, "rejected": 0, "cached": 0}


def _get_handle():
    global _handle
    if _handle is None:
        _handle = cudnn.create_handle()
    return _handle


def _load_cache():
    global _plan_cache
    if _plan_cache is None:
        _plan_cache = {}
        if os.path.exists(_CACHE_FILE):
            for line in open(_CACHE_FILE):
                k, _, v = line.rstrip("\n").rpartition("\t")
                if k:
                    _plan_cache[k] = v
    return _plan_cache


def _store_cache(key, name):
    _load_cache()[key] = name
    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    with open(_CACHE_FILE, "a") as fh:
        fh.write(f"{key}\t{name}\n")


class Slice:
    """Channel slice [c0, c0 + c) of an NHWC buffer `buf` of shape [H, W, Ctot] (batch 1)."""

    def __init__(self, buf, c0=0, c=None):
        assert buf.dim() == 3 and buf.is_contiguous() and buf.dtype == torch.half
        self.buf, self.c0 = buf, c0
        self.c = buf.shape[2] - c0 if c is None else c
        self.h, self.w, self.ctot = buf.shape
        assert c0 + self.c <= self.ctot

    @property
    def ptr(self):
        return self.buf.data_ptr() + 2 * self.c0

    def dims(self):
        return [1, self.c, self.h, self.w], [
            self.h * self.w * self.ctot,
            1,
            self.w * self.ctot,
            self.ctot,
        ]

    def view(self):
        """[1, C, H, W] torch view (channels-last strides) of the slice."""
        return self.buf[:, :, self.c0 : self.c0 + self.c].permute(2, 0, 1).unsqueeze(0)

    def overlaps(self, o):
        if self.buf.data_ptr() != o.buf.data_ptr():
            return False
        return self.c0 < o.c0 + o.c and o.c0 < self.c0 + self.c

    def key(self):
        return (
            f"{self.c}/{self.ctot}@{self.c0 % 8}"  # geometry only (alignment class of the offset)
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


def _engine(name):
    m = re.match(r"eng(\d+)", name)
    return int(m.group(1)) if m else -1


class FusedConv:
    def __init__(self, x: Slice, w: torch.Tensor, y: Slice, epi, stride=1, padding=None):
        """w: [K, C, R, S] fp16 with channels_last strides (physical KRSC)."""
        K, C, R, S = w.shape
        assert C == x.c and K == y.c, (w.shape, x.c, y.c)
        self.x, self.w, self.y, self.epi = (
            x,
            w.contiguous(memory_format=torch.channels_last),
            y,
            list(epi),
        )
        self.stride = stride
        self.padding = (R // 2, S // 2) if padding is None else padding
        for e in self.epi:
            if e[0] in ("add", "sub", "mul"):
                assert not y.overlaps(e[1]), (
                    "cuDNN: the output must not alias an epilogue operand (D8)"
                )
        ho = (x.h + 2 * self.padding[0] - R) // stride + 1
        wo = (x.w + 2 * self.padding[1] - S) // stride + 1
        assert (ho, wo) == (y.h, y.w), ((ho, wo), (y.h, y.w))
        self._build()

    # ---- graph
    def _graph(self, handle):
        H16 = cudnn.data_type.HALF
        F32 = cudnn.data_type.FLOAT
        # as in Rust: fp16 accumulation inside the conv, epilogue chain on fp32 virtual tensors (with fp16
        # intermediates no engine accepts the fused graphs)
        g = cudnn.pygraph(
            io_data_type=H16, intermediate_data_type=F32, compute_data_type=F32, handle=handle
        )
        xd, xs = self.x.dims()
        X = g.tensor(name="X", dim=xd, stride=xs, data_type=H16)
        W = g.tensor(name="W", dim=list(self.w.shape), stride=list(self.w.stride()), data_type=H16)
        t = g.conv_fprop(
            image=X,
            weight=W,
            padding=list(self.padding),
            stride=[self.stride] * 2,
            dilation=[1, 1],
            compute_data_type=H16,
        )
        K = self.w.shape[0]
        ins = []
        for i, e in enumerate(self.epi):
            op = e[0]
            if op in ("bias", "scale"):
                T = g.tensor(name=f"E{i}", dim=[1, K, 1, 1], stride=[K, 1, K, K], data_type=H16)
                t = g.bias(input=t, bias=T) if op == "bias" else g.mul(a=t, b=T)
                ins.append(T)
            elif op in ("add", "sub", "mul"):
                d, s = e[1].dims()
                T = g.tensor(name=f"E{i}", dim=d, stride=s, data_type=H16)
                t = {"add": g.add, "sub": g.sub, "mul": g.mul}[op](a=t, b=T)
                ins.append(T)
            else:
                t = {"relu": g.relu, "sigmoid": g.sigmoid, "tanh": g.tanh}[op](input=t)
                ins.append(None)
        yd, ys = self.y.dims()
        t.set_output(True).set_data_type(H16).set_dim(yd).set_stride(ys)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        if (
            os.environ.get("RAFT_TUNE") != "heuristic"
        ):  # heuristic mode builds plans lazily (runtime-compiled ones cost ~0.3 s each)
            g.build_plans(cudnn.build_plan_policy.ALL)
        return g, X, W, ins, t

    def key(self):
        e = ",".join(
            x[0] + (x[1].key() if len(x) > 1 and isinstance(x[1], Slice) else "") for x in self.epi
        )
        return f"t1 x{self.x.h}x{self.x.w}:{self.x.key()} w{tuple(self.w.shape)} s{self.stride} p{self.padding} y{self.y.key()} [{e}]"

    def _pack(self, xp, yp, eptrs):
        pack = {self.X: xp, self.W: self.w.data_ptr(), self.Y: yp}
        for T, p in zip(self.E, eptrs):
            if T is not None:
                pack[T] = p
        return pack

    def _operand_ptrs(self):
        return [
            None if len(e) == 1 else e[1].ptr if isinstance(e[1], Slice) else e[1].data_ptr()
            for e in self.epi
        ]

    def _reference(self, xs, ops):
        """fp32 PyTorch reference of the fused op on scratch inputs (views [1, C, H, W])."""
        r = F.conv2d(xs.float(), self.w.float(), None, self.stride, self.padding)
        for e, o in zip(self.epi, ops):
            k = e[0]
            if k == "bias":
                r = r + o.float().view(1, -1, 1, 1)
            elif k == "scale":
                r = r * o.float().view(1, -1, 1, 1)
            elif k == "add":
                r = r + o.float()
            elif k == "sub":
                r = r - o.float()
            elif k == "mul":
                r = r * o.float()
            elif k == "relu":
                r = torch.relu(r)
            elif k == "sigmoid":
                r = torch.sigmoid(r)
            elif k == "tanh":
                r = torch.tanh(r)
        return r

    def _build(self):
        handle = _get_handle()
        try:
            self.g, self.X, self.W, self.E, self.Y = self._graph(handle)
        except Exception as ex:  # noqa: BLE001 - no cuDNN support: PyTorch fallback
            print(f"[fused_conv] no cuDNN graph for {self.key()}: {ex}")
            self.g = None
            return
        stats["built"] += 1
        # scratch copies with the same geometry: tuning must not touch live buffers
        gen = torch.Generator(device="cuda").manual_seed(0)

        def scratch(s: Slice):
            b = (torch.rand(s.buf.shape, generator=gen, device="cuda") * 2 - 1).half()
            return Slice(b, s.c0, s.c)

        xs = scratch(self.x)
        ys = Slice(torch.empty_like(self.y.buf), self.y.c0, self.y.c)
        eops, eptrs = [], []
        for e in self.epi:
            if e[0] in ("add", "sub", "mul"):
                s = scratch(e[1])
                eops.append(s.view())
                eptrs.append(s.ptr)
            elif len(e) > 1:
                eops.append(e[1])
                eptrs.append(e[1].data_ptr())
            else:
                eops.append(None)
                eptrs.append(None)
        ref = self._reference(xs.view(), eops)
        n = self.g.get_execution_plan_count()
        names = [self.g.get_plan_name_at_index(i) for i in range(n)]
        cached = _load_cache().get(self.key())
        order = [names.index(cached)] if cached in names else range(n)
        heuristic = (
            os.environ.get("RAFT_TUNE") == "heuristic"
        )  # first verified plan in cuDNN's ranking, no timing
        best = None
        for i in order:
            if _engine(names[i]) in DENY_ENGINES:
                continue
            try:
                if heuristic or cached in names:
                    self.g.build_plan_at_index(i)
                ws = torch.empty(
                    max(self.g.get_workspace_size_plan_at_index(i), 1),
                    device="cuda",
                    dtype=torch.uint8,
                )
                ys.buf.fill_(float("nan"))
                pack = self._pack(xs.ptr, ys.ptr, eptrs)
                self.g.execute_plan_at_index(pack, ws, i, handle=handle)
                torch.cuda.synchronize()
                d = (ys.view().float() - ref).abs().max()
                dev = float("inf") if torch.isnan(d) else float(d / (ref.abs().max() + 1e-3))
                if dev > VERIFY_TOL:
                    stats["rejected"] += 1
                    print(
                        f"[fused_conv] rejected {names[i]} (deviation {dev:.3g}) for {self.key()}"
                    )
                    continue
                ms = (
                    0.0
                    if (cached in names or heuristic)
                    else _time(lambda: self.g.execute_plan_at_index(pack, ws, i, handle=handle))  # noqa: B023 - _time invokes this callback synchronously
                )
            except Exception:  # noqa: BLE001 - plan fails to run
                continue
            if best is None or ms < best[0]:
                best = (ms, i, ws)
            if heuristic:
                break
        if best is None and cached in names:  # cached plan no longer valid: full tuning
            _load_cache().pop(self.key(), None)
            return self._build()
        if best is None:
            print(f"[fused_conv] no valid plan for {self.key()}: PyTorch fallback")
            self.g = None
            return
        if cached in names:
            stats["cached"] += 1
        elif not heuristic:
            _store_cache(self.key(), names[best[1]])
        _, self.plan, self.ws = best
        self.pack = self._pack(self.x.ptr, self.y.ptr, self._operand_ptrs())

    def __call__(self):
        if self.g is None:
            ops = [
                None if len(e) == 1 else e[1].view() if isinstance(e[1], Slice) else e[1]
                for e in self.epi
            ]
            self.y.view().copy_(self._reference(self.x.view(), ops).half())
            return
        h = _get_handle()
        cudnn.set_stream(handle=h, stream=torch.cuda.current_stream().cuda_stream)
        self.g.execute_plan_at_index(self.pack, self.ws, self.plan, handle=h)

"""The Rust engine's update loop, re-hosted in PyTorch (inference, batch 1, default architecture, fp16 model).

    model.use_fast_loop()   # after model.half() (+ optionally use_cudnn_half_acc_convs / torch.compile of the encoders)

The encoders stay PyTorch modules. From the context outputs on, everything runs like the Rust engine:
- GRU inputs are persistent wide NHWC buffers [h | x-parts]; every producer (motion encoder, pool, interp) writes its
  channel slice directly: no torch.cat, no layout copies
- cuDNN convolutions with fp16 accumulation and fused epilogues (core/fused_conv.py): BN-free bias+ReLU, the z/r gates
  (+ context add, sigmoid, and r*h), and the whole GRU update h' = h + z(q - h) in the epilogue of the q-conv, written
  into the other h slot ("ping-pong": the slot that is dead after the GRU's convs, so the output never aliases an
  operand; odd iterations use z/r weights with rotated input channels)
- custom kernels (core/raft_kernels.cu, same as Rust): correlation pyramid + lookup, pool/bilinear into slices,
  7x7 flow conv on the x-channel only, flow-head output conv fused with the disparity update, convex upsampling
- the motion branch runs on a second stream concurrently with the coarse GRUs
- two iterations (even/odd parity) are captured as one CUDA graph and replayed iters/2 times
"""

import os
import statistics

import torch

from core.cuda_kernels import Kernels
from core.fused_conv import FusedConv, Slice

CX = (256, 256, 128)  # x-part channels of the GRU inputs at 1/4, 1/8, 1/16


def _cl(w):
    return w.detach().half().contiguous(memory_format=torch.channels_last)


class LoopWeights:
    """Update-block weights in the Rust export layout (rust/tools/export_weights.py), built from the model."""

    def __init__(self, model):
        u = model.update_block
        e = u.encoder
        self.gru = []
        for g in (u.gru08, u.gru16, u.gru32):
            wzr = torch.cat([g.convz.weight, g.convr.weight], 0).detach().half()
            self.gru.append(
                dict(
                    zr=(
                        _cl(wzr),
                        _cl(torch.cat([wzr[:, 128:], wzr[:, :128]], 1)),
                    ),  # [even, odd parity]
                    q_h=_cl(g.convq.weight[:, :128]),
                    q_x=_cl(g.convq.weight[:, 128:]),
                    bz=g.convz.bias.detach().half().contiguous(),
                    br=g.convr.bias.detach().half().contiguous(),
                    bq=g.convq.bias.detach().half().contiguous(),
                )
            )
        w = e.convc1.weight.detach().half()
        self.convc1 = (
            _cl(torch.cat([w, w.new_zeros(w.shape[0], 40 - w.shape[1], 1, 1)], 1)),
            e.convc1.bias.detach().half().contiguous(),
        )
        self.convc2 = (_cl(e.convc2.weight), e.convc2.bias.detach().half().contiguous())
        self.convf1 = (
            e.convf1.weight.detach()[:, 0].float().contiguous(),
            e.convf1.bias.detach().float().contiguous(),
        )
        self.convf2 = (_cl(e.convf2.weight), e.convf2.bias.detach().half().contiguous())
        w, b = e.conv.weight.detach().half(), e.conv.bias.detach().half()
        self.conv = (
            _cl(torch.cat([w, w.new_zeros(128 - w.shape[0], *w.shape[1:])], 0)),
            torch.cat([b, b.new_zeros(128 - b.shape[0])]).contiguous(),
        )
        self.fh1 = (
            _cl(u.flow_head.conv1.weight),
            u.flow_head.conv1.bias.detach().half().contiguous(),
        )
        self.fh2 = (
            _cl(u.flow_head.conv2.weight[:1]),
            u.flow_head.conv2.bias.detach().half()[:1].contiguous(),
        )
        self.mask1 = (_cl(u.mask[0].weight), u.mask[0].bias.detach().half().contiguous())
        self.mask2 = (_cl(u.mask[2].weight), u.mask[2].bias.detach().half().contiguous())


class LoopEngine:
    """Buffers, fused ops and the captured iteration graph for one resolution."""

    def __init__(self, wts: LoopWeights, sizes, split_zr=True, use_graph=True, concurrent=True):
        self.k = Kernels.get()
        self.wts, self.sizes, self.split_zr = wts, sizes, split_zr
        self.use_graph, self.concurrent = use_graph, concurrent
        dev = "cuda"
        half = dict(device=dev, dtype=torch.half)
        (h2, w2) = sizes[0]
        self.p2 = h2 * w2
        self.hx = [torch.zeros(h, w, 128 + cx, **half) for (h, w), cx in zip(sizes, CX)]
        self.z = [torch.empty(h, w, 128 if split_zr else 256, **half) for h, w in sizes]
        self.rh = [torch.empty(h, w, 128, **half) for h, w in sizes]
        self.qx = [torch.empty(h, w, 128, **half) for h, w in sizes]
        self.czr = [torch.empty(h, w, 256, **half) for h, w in sizes]
        self.cq = [torch.empty(h, w, 128, **half) for h, w in sizes]
        self.coords = torch.empty(self.p2, device=dev, dtype=torch.float32)
        self.corr_w = [w2 >> level for level in range(4)]
        self.corr = [torch.empty(self.p2 * cw, **half) for cw in self.corr_w]
        self.look = torch.empty(h2, w2, 40, **half)
        self.mc1 = torch.empty(h2, w2, 64, **half)
        self.mot_in = torch.empty(h2, w2, 128, **half)
        self.f1 = torch.empty(h2, w2, 64, **half)
        self.fh = torch.empty(h2, w2, 256, **half)
        self.m1 = torch.empty(h2, w2, 256, **half)
        self.m2 = torch.empty(h2, w2, 144, **half)
        self.side = torch.cuda.Stream()
        # step programs for the two iteration parities (each = one iteration + flow head)
        self.par = [0, 0, 0]
        self.body = [self._iteration_steps(), self._iteration_steps()]
        assert self.par == [0, 0, 0]
        self.mask_ops = [self._mask_ops(p) for p in (0, 1)]
        self.graph = None

    # ---- step construction (mirrors rust/src/model.rs)
    def h_off(self, lvl, p):
        return 0 if p == 0 else CX[lvl]

    @staticmethod
    def x_off(p):
        return 128 if p == 0 else 0

    def _kern(self, name, cfg, *args):
        return lambda: self.k.launch(name, cfg, *args)

    def _resample(self, kind, src, dst, slot):
        (hs, ws), (hd, wd) = self.sizes[src], self.sizes[dst]
        sb, db = self.hx[src], self.hx[dst]
        sp = sb.data_ptr() + 2 * self.h_off(src, self.par[src])
        dp = db.data_ptr() + 2 * (self.x_off(self.par[dst]) + slot)
        return self._kern(
            kind, self.k.linear(hd * wd * 16), sp, sb.shape[2], hs, ws, dp, db.shape[2], hd, wd, 128
        )

    def _gru(self, lvl):
        w = self.wts.gru[lvl]
        p = self.par[lvl]
        hx = self.hx[lvl]
        wide = hx.shape[2]
        h_cur, h_next, x_cur = self.h_off(lvl, p), self.h_off(lvl, 1 - p), self.x_off(p)
        wzr = w["zr"][p]
        h_sl = Slice(hx, h_cur, 128)
        steps = []
        if self.split_zr:
            steps.append(
                FusedConv(
                    Slice(hx),
                    wzr[:128],
                    Slice(self.z[lvl]),
                    [("add", Slice(self.czr[lvl], 0, 128)), ("sigmoid",)],
                )
            )
            steps.append(
                FusedConv(
                    Slice(hx),
                    wzr[128:],
                    Slice(self.rh[lvl]),
                    [("add", Slice(self.czr[lvl], 128, 128)), ("sigmoid",), ("mul", h_sl)],
                )
            )
            z_sl = Slice(self.z[lvl])
        else:
            steps.append(
                FusedConv(
                    Slice(hx),
                    wzr,
                    Slice(self.z[lvl]),
                    [("add", Slice(self.czr[lvl])), ("sigmoid",)],
                )
            )
            P = hx.shape[0] * hx.shape[1]
            steps.append(
                self._kern(
                    "gru_gate1", self.k.linear(P * 16), self.z[lvl], h_sl.ptr, wide, self.rh[lvl], P
                )
            )
            z_sl = Slice(self.z[lvl], 0, 128)
        steps.append(FusedConv(Slice(hx, x_cur, CX[lvl]), w["q_x"], Slice(self.qx[lvl]), []))
        steps.append(
            FusedConv(
                Slice(self.rh[lvl]),
                w["q_h"],
                Slice(hx, h_next, 128),
                [
                    ("add", Slice(self.qx[lvl])),
                    ("add", Slice(self.cq[lvl])),
                    ("tanh",),
                    ("sub", h_sl),
                    ("mul", z_sl),
                    ("add", h_sl),
                ],
            )
        )
        self.par[lvl] = 1 - p
        return steps

    def _motion(self):
        (h2, w2), P = self.sizes[0], self.p2
        W = self.wts
        steps = [
            self._kern(
                "corr_lookup", self.k.linear(P), *self.corr, *self.corr_w, self.coords, self.look, P
            ),
            FusedConv(
                Slice(self.look), W.convc1[0], Slice(self.mc1), [("bias", W.convc1[1]), ("relu",)]
            ),
            FusedConv(
                Slice(self.mc1),
                W.convc2[0],
                Slice(self.mot_in, 0, 64),
                [("bias", W.convc2[1]), ("relu",)],
            ),
            self._kern(
                "convf1_direct",
                (((w2 + 15) // 16, (h2 + 15) // 16, 1), (16, 16, 1)),
                self.coords,
                h2,
                w2,
                W.convf1[0],
                W.convf1[1],
                self.f1,
            ),
            FusedConv(
                Slice(self.f1),
                W.convf2[0],
                Slice(self.mot_in, 64, 64),
                [("bias", W.convf2[1]), ("relu",)],
            ),
        ]
        mot = self.x_off(self.par[0])
        hx0 = self.hx[0]
        steps.append(
            FusedConv(
                Slice(self.mot_in),
                W.conv[0],
                Slice(hx0, mot, 128),
                [("bias", W.conv[1]), ("relu",)],
            )
        )
        steps.append(
            self._kern(
                "write_flow",
                self.k.linear(P),
                self.coords,
                w2,
                P,
                hx0.data_ptr() + 2 * (mot + 126),
                hx0.shape[2],
            )
        )
        return steps

    def _flow_head(self):
        (h2, w2), W = self.sizes[0], self.wts
        hx0 = self.hx[0]
        return [
            FusedConv(
                Slice(hx0, self.h_off(0, self.par[0]), 128),
                W.fh1[0],
                Slice(self.fh),
                [("bias", W.fh1[1]), ("relu",)],
            ),
            lambda: self.k.launch(
                "flow_head_out",
                (((self.p2 * 32 + 255) // 256, 1, 1), (256, 1, 1)),
                self.fh,
                h2,
                w2,
                W.fh2[0],
                W.fh2[1],
                self.coords,
            ),
        ]

    def _iteration_steps(self):
        """[('side', steps), ('main', steps), ('join',), ('main', steps)] for one iteration + flow head."""
        side = self._motion()
        main = [self._resample("pool2x_into", 1, 2, 0)] + self._gru(2)
        main += [
            self._resample("pool2x_into", 0, 1, 0),
            self._resample("interp_into", 2, 1, 128),
        ] + self._gru(1)
        tail = [self._resample("interp_into", 1, 0, 128)] + self._gru(0) + self._flow_head()
        return [("fork", side), ("main", main), ("join",), ("main", tail)]

    def _mask_ops(self, p):
        W = self.wts
        return [
            FusedConv(
                Slice(self.hx[0], self.h_off(0, p), 128),
                W.mask1[0],
                Slice(self.m1),
                [("bias", W.mask1[1]), ("relu",)],
            ),
            FusedConv(Slice(self.m1), W.mask2[0], Slice(self.m2), [("bias", W.mask2[1])]),
        ]

    # ---- execution
    def _run_program(self, prog):
        cur = torch.cuda.current_stream()
        for item in prog:
            if item[0] == "fork":
                if self.concurrent:
                    self.side.wait_stream(cur)
                    with torch.cuda.stream(self.side):
                        for s in item[1]:
                            s()
                else:
                    for s in item[1]:
                        s()
            elif item[0] == "join":
                if self.concurrent:
                    cur.wait_stream(self.side)
            else:
                for s in item[1]:
                    s()

    def _pair(self):
        self._run_program(self.body[0])
        self._run_program(self.body[1])

    def setup(self, net_list, zqr_list, fmap1, fmap2):
        """Prologue: hidden states, context terms, correlation pyramid, disparity state."""
        k, P = self.k, self.p2
        (h2, w2) = self.sizes[0]
        for lvl in range(3):
            self.hx[lvl][:, :, :128].copy_(net_list[lvl][0].permute(1, 2, 0))
            z = zqr_list[lvl][0].permute(1, 2, 0).contiguous()  # [h, w, 384]
            g = self.wts.gru[lvl]
            Pl = z.shape[0] * z.shape[1]
            k.launch(
                "prep_context",
                k.linear(Pl * 128),
                z,
                g["bz"],
                g["br"],
                g["bq"],
                self.czr[lvl],
                self.cq[lvl],
                Pl,
            )
        f1 = fmap1[0].permute(1, 2, 0).contiguous()  # raw kernel expects packed NHWC
        f2 = fmap2[0].permute(1, 2, 0).contiguous()
        t = (
            w2 + 127
        ) // 128  # all-pairs correlation per row, fp32 accumulation (same kernel as Rust)
        k.launch("corr_gemm", ((t, t, h2), (256, 1, 1)), f1, f2, self.corr[0], w2, 1.0 / 16.0)
        for level in range(3):
            k.launch(
                "corr_pool",
                k.linear(P * self.corr_w[level + 1]),
                self.corr[level],
                self.corr[level + 1],
                P,
                self.corr_w[level],
                self.corr_w[level + 1],
            )
        k.launch("init_coords", k.linear(P), self.coords, w2, P)

    def iterate(self, iters):
        if self.use_graph and self.graph is None:
            # tune every op (eager pass on the live buffers), then capture two iterations; the eager warm-up
            # changes the state, so the caller's setup() must run again: done in run()
            raise RuntimeError("capture() first")
        for _ in range(iters // 2):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._pair()
        if iters % 2:
            self._run_program(self.body[0])
        return iters % 2

    def capture(self):
        self._pair()  # tunes all ops / warms up outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._pair()
        self.graph = g

    def finish(self, final_par, hp, wp):
        k = self.k
        for op in self.mask_ops[final_par]:
            op()
        (h2, w2) = self.sizes[0]
        out = torch.empty(1, 1, hp, wp, device="cuda", dtype=torch.float32)
        k.launch(
            "upsample_out", k.linear(hp * wp), self.coords, self.m2, h2, w2, 0, 0, hp, wp, out, 0
        )
        xs = torch.arange(w2, device="cuda", dtype=torch.float32)
        low = torch.zeros(1, 2, h2, w2, device="cuda", dtype=torch.float32)
        low[0, 0] = self.coords.view(h2, w2) - xs
        return low, out


class FastLoop:
    def __init__(self, model):
        self.wts = LoopWeights(model)
        self.version = self._version(model)
        self.engines = {}
        self.split_zr = os.environ.get("RAFT_SPLIT_ZR", "auto")
        self.use_graph = os.environ.get("RAFT_GRAPH", "1") == "1"
        self.concurrent = os.environ.get("RAFT_CONCURRENT", "1") == "1"

    @staticmethod
    def _version(model):
        params = tuple(model.update_block.parameters())
        if any(p.is_inference() for p in params):
            return None
        return tuple((id(p), p._version, p.dtype, p.device) for p in params)

    def _time_engine(self, eng, net_list, zqr_list, fmap1, fmap2, runs=5):
        ts = []
        for _ in range(runs + 1):
            eng.setup(net_list, zqr_list, fmap1, fmap2)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            eng.iterate(8)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts[1:])

    def _engine(self, sizes, net_list, zqr_list, fmap1, fmap2):
        key = tuple(sizes)
        if key not in self.engines:

            def mk(split):
                return LoopEngine(
                    self.wts,
                    sizes,
                    split_zr=split,
                    use_graph=self.use_graph,
                    concurrent=self.concurrent,
                )

            if self.split_zr == "auto":
                # z/r as two N=128 convs (r*h fused) vs one N=256 conv + gate kernel: depends on the resolution
                cands = []
                for split in (True, False):
                    eng = mk(split)
                    eng.setup(net_list, zqr_list, fmap1, fmap2)
                    if self.use_graph:
                        eng.capture()
                    cands.append(
                        (self._time_engine(eng, net_list, zqr_list, fmap1, fmap2), split, eng)
                    )
                    del eng  # (the loop variable would keep the last candidate alive)
                cands.sort(key=lambda c: c[0])
                print(
                    f"[fast_loop] {sizes[0]}: split {cands[0][1]} ({[round(c[0], 2) for c in cands]} ms / 8 iters)"
                )
                self.engines[key] = cands[0][2]
                del cands[1:]  # free the losing engine's buffers and graph
                torch.cuda.empty_cache()
            else:
                eng = mk(self.split_zr == "1")
                eng.setup(net_list, zqr_list, fmap1, fmap2)
                if self.use_graph:
                    eng.capture()
                self.engines[key] = eng
        return self.engines[key]

    def __call__(self, model, net_list, zqr_list, fmap1, fmap2, iters, hp, wp):
        version = self._version(model)
        if version is None or version != self.version:
            self.engines.clear()
            self.wts = LoopWeights(model)
            self.version = version
        sizes = [tuple(n.shape[2:]) for n in net_list]
        eng = self._engine(sizes, net_list, zqr_list, fmap1, fmap2)
        eng.setup(net_list, zqr_list, fmap1, fmap2)
        odd = eng.iterate(iters)
        return eng.finish(odd, hp, wp)


def engine_memory(eng):
    """Bytes held by an engine: (buffers, conv workspaces)."""
    bufs = [
        *eng.hx,
        *eng.z,
        *eng.rh,
        *eng.qx,
        *eng.czr,
        *eng.cq,
        eng.coords,
        *eng.corr,
        eng.look,
        eng.mc1,
        eng.mot_in,
        eng.f1,
        eng.fh,
        eng.m1,
        eng.m2,
    ]
    ops = [
        s
        for prog in eng.body
        for item in prog
        if item[0] != "join"
        for s in item[1]
        if hasattr(s, "ws")
    ]
    ops += [op for ops_ in eng.mask_ops for op in ops_ if hasattr(op, "ws")]
    return sum(b.numel() * b.element_size() for b in bufs), sum(o.ws.numel() for o in ops), len(ops)

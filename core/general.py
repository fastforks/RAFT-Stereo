"""General RAFT-Stereo inference engine: any input size, no per-size compilation, autotuning or CUDA graph.

Default architecture (3 GRU levels at 1/4, n_downsample 2, 32 iterations), batch 1, fp16 NHWC, fp32 disparity state.

- Convolutions: plain cuDNN convs (fp16 in/out, fp16 accumulation, no fused epilogue) through cudnn-frontend graphs,
  restricted to cuDNN's *precompiled* engines (plans with RUNTIME_COMPILATION are deselected). Fused-epilogue plans
  on sm_120 exist only as runtime-compiled (NVRTC) kernels: 20-800 ms to build per conv and per new size, which
  was the cold-start cost of v6. A plain conv plan costs ~3 ms (graph + heuristics query) and builds in ~0 ms; the
  first plan in cuDNN's heuristic order is used (no timing).
- Everything else (bias / BatchNorm affine + activations, InstanceNorm, GRU gates and update, correlation GEMM,
  pyramid, lookup, resampling, flow head output, convex upsampling) is a custom kernel from core/raft_kernels.cu,
  loaded from a shipped cubin (core/prebuilt/) -- no nvcc at run time.
- Buffers are ordinary PyTorch tensors allocated per call (caching allocator): the encoder activations are freed
  before the correlation volume and the loop buffers are allocated, so the peak is max(encoder, loop), not the sum.
- Every conv reads and writes *packed* NHWC tensors: with channel-slice operands the precompiled engines add an
  nhwcSliceC copy kernel (measured 10.7 ms per pass at KITTI). GRU inputs per level: A = [h | x] for the z|r conv
  and B = [r*h | x] for the q conv; x is written into both by its producers (resampling kernels, motion epilogue),
  r*h by the gate kernel; the state update runs in place on A's h (separate kernel, so no aliasing issue) and also
  writes a packed copy of the 1/4 h for the flow and mask heads.

    eng = GeneralRAFT("models/raftstereo-middlebury.pth")
    disp = eng(left_uint8_hwc, right_uint8_hwc)       # torch uint8 [H, W, 3] (CPU or GPU) -> fp32 [H, W] disparity (px)
"""

import os

import torch

from core.cuda_kernels import Kernels

try:
    import cudnn
except ImportError as e:  # pragma: no cover
    raise ImportError("core.general needs cudnn-frontend: pip install nvidia-cudnn-frontend") from e

CX = (256, 256, 128)  # x-part channels of the GRU inputs at 1/4, 1/8, 1/16
IN_CHUNK = 1024
VERIFY = (
    os.environ.get("RAFT_VERIFY") == "1"
)  # check every new conv plan against an fp32 reference (debug)
FUSED = (
    os.environ.get("RAFT_FUSED", "0") == "1"
)  # GRU convs with fused epilogues (core/cutlass_conv.cu, compiled once)
# Optional CUTLASS encoder convolutions. The context encoder folds BN/ReLU into the epilogue.
FASTBB_ENC = os.environ.get("FASTBB_ENC", "1") == "1"


def _normconv():
    """fastbb_vendored.normconv (vendored from fastforks/fastbb; CUTLASS torch-op library, built on first use: needs CUTLASS_DIR +
    nvcc). None when the library is unavailable: the previous (cuDNN + separate norm kernels) encoder path is used."""
    try:
        from fastbb_vendored import normconv

        normconv.load()
    except (ImportError, RuntimeError) as e:
        import warnings

        warnings.warn(
            f"fastbb_vendored.normconv unavailable ({e}); using the previous encoder path (FASTBB_ENC=0)",
            stacklevel=2,
        )
        return None
    return normconv


def fused_cfg_zr(P):
    """Fixed size rule (measured once on the RTX 5060 Ti, results/2026-10-09_v8_fused_gru.txt): CUTLASS tile config of
    the fused z|r conv, or None = plain cuDNN conv + gru_zr kernel (faster for 4000 <= P < 10000 pixels)."""
    return None if 4000 <= P < 10000 else 0


def fused_cfg_q(P):
    return 0 if P >= 10000 else 1


def fused_cfg_affine(P, K):
    """conv + bias/BN + activation: fused only for K >= 128 output channels; with 64/96 channels cuDNN's plain conv +
    the separate affine_act pass is as fast or faster (results/2026-10-09_v8_fused_gru.txt)."""
    return None if K < 128 else (0 if P >= 10000 else 1)


# ------------------------------------------------------------------------------------------------ convolutions
class S:
    """Channel slice [c0, c0 + c) of an NHWC buffer [H, W, Ctot] (fp16, contiguous)."""

    __slots__ = ("buf", "c0", "c", "h", "w", "ctot")

    def __init__(self, buf, c0=0, c=None):
        self.buf, self.c0 = buf, c0
        self.h, self.w, self.ctot = buf.shape
        self.c = self.ctot - c0 if c is None else c

    @property
    def ptr(self):
        return self.buf.data_ptr() + 2 * self.c0

    @property
    def p(self):
        return self.h * self.w


def _full(x):
    return x.c0 == 0 and x.c == x.ctot


def _t4(x):
    """S (whole buffer) -> channels-last [1, C, H, W] view."""
    return x.buf.permute(2, 0, 1)[None]


class Convs:
    """Plain fp16-accumulate cuDNN convs with precompiled engines, one plan per geometry (kept for the process)."""

    def __init__(self):
        self.handle = cudnn.create_handle()
        self.plans = {}
        self.ws = torch.empty(0, device="cuda", dtype=torch.uint8)
        self.n_built = 0

    def _build(self, x: S, w, y: S, stride, pad):
        H16 = cudnn.data_type.HALF
        K, R, Sz, C = w.shape
        g = cudnn.pygraph(
            io_data_type=H16, intermediate_data_type=H16, compute_data_type=H16, handle=self.handle
        )
        X = g.tensor(
            name="X",
            dim=[1, C, x.h, x.w],
            stride=[x.p * x.ctot, 1, x.w * x.ctot, x.ctot],
            data_type=H16,
        )
        W = g.tensor(name="W", dim=[K, C, R, Sz], stride=[C * R * Sz, 1, Sz * C, C], data_type=H16)
        Y = g.conv_fprop(
            image=X,
            weight=W,
            padding=[pad, pad],
            stride=[stride, stride],
            dilation=[1, 1],
            compute_data_type=H16,
        )
        Y.set_output(True).set_data_type(H16).set_dim([1, K, y.h, y.w]).set_stride(
            [y.p * y.ctot, 1, y.w * y.ctot, y.ctot]
        )
        g.validate()
        g.build_operation_graph()
        g.deselect_behavior_notes([cudnn.behavior_note.RUNTIME_COMPILATION])
        g.create_execution_plans([cudnn.heur_mode.A])
        g.check_support()
        last = None
        for i in range(g.get_execution_plan_count()):
            # eng0: reference engine (100x slower); eng8: crashes on sm_120 (DESIGN_FLAWS.md D9)
            if g.get_plan_name_at_index(i).split("_")[0] in ("eng0", "eng8"):
                continue
            try:
                g.build_plan_at_index(i)
            except Exception as ex:  # noqa: BLE001
                last = ex
                continue
            if VERIFY and not self._verify(g, X, W, Y, i, x, w, y, stride, pad):
                continue
            self.n_built += 1
            return g, X, W, Y, i, g.get_workspace_size_plan_at_index(i)
        raise RuntimeError(
            f"no precompiled cuDNN plan for conv x{(x.h, x.w, x.c, x.ctot)} w{tuple(w.shape)} s{stride}: {last}"
        )

    def _verify(self, g, X, W, Y, i, x, w, y, stride, pad):
        xr = (torch.rand(x.buf.shape, device="cuda") * 2 - 1).half()
        yr = torch.full(y.buf.shape, float("nan"), device="cuda", dtype=torch.half)
        ws = torch.empty(
            max(g.get_workspace_size_plan_at_index(i), 1), device="cuda", dtype=torch.uint8
        )
        g.execute_plan_at_index(
            {X: xr.data_ptr() + 2 * x.c0, W: w.data_ptr(), Y: yr.data_ptr() + 2 * y.c0},
            ws,
            i,
            handle=self.handle,
        )
        ref = torch.nn.functional.conv2d(
            xr[:, :, x.c0 : x.c0 + x.c].permute(2, 0, 1)[None].float(),
            w.permute(0, 3, 1, 2).float(),
            None,
            stride,
            pad,
        )
        out = yr[:, :, y.c0 : y.c0 + y.c].permute(2, 0, 1)[None].float()
        dev = float((out - ref).abs().max() / (ref.abs().max() + 1e-3))
        ok = dev == dev and dev < 0.05
        if not ok:
            print(f"[general] rejected {g.get_plan_name_at_index(i)}: deviation {dev:.3g}")
        return ok

    def __call__(self, x: S, w, y: S, stride=1):
        K, R, _, C = w.shape
        assert C == x.c and K == y.c, (tuple(w.shape), x.c, y.c)
        pad = R // 2
        key = (x.h, x.w, x.c, x.ctot, K, y.ctot, R, stride)
        e = self.plans.get(key)
        if e is None:
            e = self.plans[key] = self._build(x, w, y, stride, pad)
        g, X, W, Y, i, ws = e
        if ws > self.ws.numel():
            self.ws = torch.empty(ws, device="cuda", dtype=torch.uint8)
        cudnn.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        g.execute_plan_at_index(
            {X: x.ptr, W: w.data_ptr(), Y: y.ptr}, self.ws, i, handle=self.handle
        )


# ------------------------------------------------------------------------------------------------ weights
def _krsc(w):
    return w.detach().half().permute(0, 2, 3, 1).contiguous().cuda()


class Weights:
    """Inference weights from an upstream RAFT-Stereo state dict (default architecture)."""

    def __init__(self, sd):
        sd = {k.removeprefix("module."): v.float() for k, v in sd.items()}
        self.w, self.s, self.t = {}, {}, {}

        def conv(name, key):
            self.w[name] = _krsc(sd[key + ".weight"])
            return sd.get(key + ".bias")

        def bn(
            name, conv_key, norm_key
        ):  # BN(conv(x) + b) = conv(x) * s + ((b - mean) * s + beta), applied in fp32
            b = conv(name, conv_key)
            s = sd[norm_key + ".weight"] / torch.sqrt(sd[norm_key + ".running_var"] + 1e-5)
            self.s[name] = s.cuda().contiguous()
            self.t[name] = (
                ((b - sd[norm_key + ".running_mean"]) * s + sd[norm_key + ".bias"])
                .cuda()
                .contiguous()
            )

        def bias(name, key):
            self.t[name] = conv(name, key).cuda().contiguous()

        def resblock(pfx, key, batch, ds):
            if batch:
                bn(pfx + ".c1", key + ".conv1", key + ".norm1")
                bn(pfx + ".c2", key + ".conv2", key + ".norm2")
                if ds:
                    bn(pfx + ".ds", key + ".downsample.0", key + ".downsample.1")
            else:  # InstanceNorm: conv bias cancels (A8)
                conv(pfx + ".c1", key + ".conv1")
                conv(pfx + ".c2", key + ".conv2")
                if ds:
                    conv(pfx + ".ds", key + ".downsample.0")

        for net, batch in (("fnet", False), ("cnet", True)):
            if batch:
                bn("cnet.conv1", "cnet.conv1", "cnet.norm1")
            else:
                conv("fnet.conv1", "fnet.conv1")
            for li in (1, 2, 3) + ((4, 5) if batch else ()):
                resblock(f"{net}.layer{li}.0", f"{net}.layer{li}.0", batch, li > 1)
                resblock(f"{net}.layer{li}.1", f"{net}.layer{li}.1", batch, False)
        bias("fnet.conv2", "fnet.conv2")
        for _lvl, head in ((0, "outputs08"), (1, "outputs16")):
            for j in (0, 1):
                resblock(f"cnet.{head}.{j}.rb", f"cnet.{head}.{j}.0", True, False)
                bias(f"cnet.{head}.{j}.conv", f"cnet.{head}.{j}.1")
        for j in (0, 1):
            bias(f"cnet.outputs32.{j}.conv", f"cnet.outputs32.{j}")
        for lvl in range(3):
            bias(f"zqr.{lvl}", f"context_zqr_convs.{lvl}")
        u = "update_block"
        self.gru = []
        for _lvl, gname in enumerate(("gru08", "gru16", "gru32")):
            k = f"{u}.{gname}"
            wzr = torch.cat([sd[k + ".convz.weight"], sd[k + ".convr.weight"]], 0)
            wq = sd[k + ".convq.weight"]
            self.gru.append(
                dict(
                    zr=_krsc(wzr),
                    q=_krsc(wq),
                    bz=sd[k + ".convz.bias"].half().cuda(),
                    br=sd[k + ".convr.bias"].half().cuda(),
                    bq=sd[k + ".convq.bias"].half().cuda(),
                )
            )
        e = f"{u}.encoder"
        w = sd[e + ".convc1.weight"]
        self.w["convc1"] = _krsc(torch.cat([w, w.new_zeros(w.shape[0], 40 - w.shape[1], 1, 1)], 1))
        self.t["convc1"] = sd[e + ".convc1.bias"].cuda()
        bias("convc2", e + ".convc2")
        bias("convf2", e + ".convf2")
        self.convf1_w = sd[e + ".convf1.weight"][:, 0].contiguous().cuda()
        self.convf1_b = sd[e + ".convf1.bias"].contiguous().cuda()
        w, b = sd[e + ".conv.weight"], sd[e + ".conv.bias"]
        self.w["mconv"] = _krsc(torch.cat([w, w.new_zeros(128 - w.shape[0], *w.shape[1:])], 0))
        self.t["mconv"] = torch.cat([b, b.new_zeros(128 - b.shape[0])]).cuda()
        bias("fh1", f"{u}.flow_head.conv1")
        self.fh2_w = _krsc(sd[f"{u}.flow_head.conv2.weight"][:1])
        self.fh2_b = sd[f"{u}.flow_head.conv2.bias"][:1].half().cuda()
        bias("mask1", f"{u}.mask.0")
        bias("mask2", f"{u}.mask.2")


# ------------------------------------------------------------------------------------------------ engine
class GeneralRAFT:
    def __init__(self, ckpt_or_state_dict, iters=32):
        torch.cuda.init()
        sd = ckpt_or_state_dict
        if isinstance(sd, (str, os.PathLike)):
            sd = torch.load(sd, map_location="cpu", weights_only=True)
        self.W = Weights(sd)
        self.iters = iters
        self.k = Kernels.get()
        self.convs = Convs()
        self.concurrent = os.environ.get("RAFT_CONCURRENT", "1") == "1"
        self.side = torch.cuda.Stream()
        self.fused = None
        if FUSED:
            from core.cutlass_conv import EpiArgs, FusedConvs

            self.fused, self._epi = FusedConvs.get(), EpiArgs
        self._unfusable = set(
            os.environ.get("RAFT_FUSED_SKIP", "").split(",")
        )  # debug: conv names kept plain
        self.nc = None
        if FASTBB_ENC:
            self.nc = _normconv()
            # KRSC weights viewed as channels-last [K, C, R, S] (no copy)
            self.W4 = {
                n: w.permute(0, 3, 1, 2)
                for n, w in self.W.w.items()
                if n.startswith(("fnet.layer", "cnet."))
            }

    # ---- small helpers
    def _new(self, h, w, c):
        return torch.empty(h, w, c, device="cuda", dtype=torch.half)

    def conv(self, x: S, name, stride=1, out: S = None, w=None):
        w = self.W.w[name] if w is None else w
        if out is None:
            ho, wo = (x.h - 1) // stride + 1, (x.w - 1) // stride + 1
            out = S(self._new(ho, wo, w.shape[0]))
        self.convs(x, w, out, stride)
        return out

    def affine(self, y: S, name, mode, r: S = None, out: S = None, out2: S = None, flow=None):
        """Per-channel affine (BN scale/shift or bias) + activation of conv output y, in place or into out (+ out2)."""
        s, t = self.W.s.get(name), self.W.t.get(name)
        o = y if out is None else out
        k = self.k
        k.launch(
            "affine_act",
            k.linear(y.p * y.c // 8),
            y.ptr,
            y.ctot,
            o.ptr,
            o.ctot,
            0 if out2 is None else out2.ptr,
            0 if out2 is None else out2.ctot,
            0 if s is None else s,
            0 if t is None else t,
            0 if r is None else r.ptr,
            0 if r is None else r.ctot,
            y.p,
            y.c,
            mode,
            0 if flow is None else flow[0],
            0 if flow is None else flow[1],
        )
        return o

    def conv_act(
        self,
        x: S,
        name,
        mode,
        stride=1,
        out: S = None,
        out2: S = None,
        r: S = None,
        flow=None,
        tmp: S = None,
    ):
        """affine(conv(x)) -- as one fused-epilogue conv (RAFT_FUSED=1) when the operands allow it, else conv into
        tmp (or a new buffer when out is None) followed by the affine_act kernel."""
        w = self.W.w[name]
        K, R, _, C = w.shape
        ho, wo = (x.h - 1) // stride + 1, (x.w - 1) // stride + 1
        cfg = None if self.fused is None else fused_cfg_affine(ho * wo, K)
        if (
            cfg is None
            and self.nc is not None
            and name in self.W4
            and _full(x)
            and C % 8 == 0
            and K % 8 == 0
            and out is None
            and out2 is None
            and flow is None
            and mode in (0, 1, 3)
            and (r is None or _full(r))
        ):
            y4 = self.nc.conv(
                _t4(x),
                self.W4[name],
                self.W.s.get(name),
                self.W.t.get(name),
                res=None if r is None else _t4(r),
                act=int(mode in (1, 3)),
                res_mode=int(mode == 3),
                stride=stride,
            )[0]
            return S(y4.permute(0, 2, 3, 1)[0])
        if (
            cfg is not None
            and x.c0 == 0
            and x.c == x.ctot
            and C % 8 == 0
            and name not in self._unfusable
        ):
            o = S(self._new(ho, wo, K)) if out is None else out
            s, t = self.W.s.get(name), self.W.t.get(name)
            e = self._epi(
                2,
                mode,
                0 if r is None else r.ptr,
                0 if r is None else r.ctot,
                o.ptr,
                o.ctot,
                None,
                0 if out2 is None else out2.ctot,
                None if out2 is None else out2.ptr,
                None,
                None if s is None else s.data_ptr(),
                None if t is None else t.data_ptr(),
                None if flow is None else flow[0].data_ptr(),
                0 if flow is None else flow[1],
            )
            rc = self.fused(cfg, x.buf, w, x.h, x.w, C, K, e, stride=stride, check=False)
            if rc == 0:
                return o
            if rc != 1:  # 1 = geometry not supported by the kernel -> plain path from now on
                raise RuntimeError(f"fused conv {name} failed ({rc})")
            self._unfusable.add(name)
        y = self.conv(x, name, stride, out=tmp if out is not None else None)
        return self.affine(y, name, mode, r=r, out=out, out2=out2, flow=flow)

    def in_stats(self, y: S):
        assert y.c0 == 0 and y.c == y.ctot
        k, P, C = self.k, y.p, y.c
        nblk = (P + IN_CHUNK - 1) // IN_CHUNK
        part = torch.empty(nblk * 2 * C, device="cuda", dtype=torch.float32)
        st = torch.empty(2 * C, device="cuda", dtype=torch.float32)
        k.launch(
            "in_stats_partial",
            ((nblk, 1, 1), ((256 // (C // 8)) * (C // 8), 1, 1)),
            y.ptr,
            P,
            C,
            IN_CHUNK,
            part,
        )
        k.launch(
            "in_stats_final",
            ((C, 1, 1), (32, 1, 1)),
            part,
            nblk,
            C,
            P,
            IN_CHUNK,
            1e-5,
            st,
        )
        return st

    def in_apply(self, y: S, st, mode=0, r: S = None, st_r=None):
        k, n = self.k, y.p * y.c
        k.launch(
            "in_apply",
            k.linear(n // 8),
            y.ptr,
            st,
            0 if r is None else r.ptr,
            0 if st_r is None else st_r,
            y.ptr,
            n,
            y.c,
            mode,
        )

    # ---- encoders
    def res_block(self, x: S, name, stride, ds, batch):
        if batch:
            r = self.conv_act(x, name + ".ds", 0, stride) if ds else x
            y = self.conv_act(x, name + ".c1", 1, stride)
            return self.conv_act(y, name + ".c2", 3, r=r)
        d = st_d = None
        if ds:
            d = self.conv(x, name + ".ds", stride)
            st_d = self.in_stats(d)
        y = self.conv(x, name + ".c1", stride)
        self.in_apply(y, self.in_stats(y), 0)
        o = self.conv(y, name + ".c2")
        del y
        st = self.in_stats(o)
        if ds:
            self.in_apply(o, st, 1, d, st_d)
        else:
            self.in_apply(o, st, 2, x)
        return o

    def prep_image(self, img_u8, hp, wp, top, left):
        H, W = img_u8.shape[:2]
        out = self._new(hp, wp, 3)
        self.k.launch("prep_image", self.k.linear(hp * wp), img_u8, out, H, W, hp, wp, top, left, 0)
        return S(out)

    def fnet(self, img: S, out: S):
        x = self.conv(img, "fnet.conv1")
        self.in_apply(x, self.in_stats(x), 0)
        for li, stride in ((1, 1), (2, 2), (3, 2)):
            x = self.res_block(x, f"fnet.layer{li}.0", stride, stride > 1, False)
            x = self.res_block(x, f"fnet.layer{li}.1", 1, False, False)
        self.conv_act(x, "fnet.conv2", 0, out=out, tmp=out)

    def context_head(self, x: S, lvl, head, L):
        # the level's long-lived buffers are allocated here, after the full-resolution encoder stage is freed
        h, w, cx = x.h, x.w, CX[lvl]
        L["A"][lvl], L["czr"][lvl], L["cq"][lvl] = (
            self._new(h, w, 128 + cx),
            self._new(h, w, 256),
            self._new(h, w, 128),
        )
        if lvl == 0:
            L["h0"] = self._new(h, w, 128)
        h_sl = S(L["A"][lvl], 0, 128)
        h0 = S(L["h0"]) if lvl == 0 else None
        if head == "outputs32":
            self.conv_act(x, f"cnet.{head}.0.conv", 2, out=h_sl, out2=h0)
            inp = self.conv_act(x, f"cnet.{head}.1.conv", 1)
        else:
            o = self.res_block(x, f"cnet.{head}.0.rb", 1, False, True)
            self.conv_act(o, f"cnet.{head}.0.conv", 2, out=h_sl, out2=h0)
            o = self.res_block(x, f"cnet.{head}.1.rb", 1, False, True)
            inp = self.conv_act(o, f"cnet.{head}.1.conv", 1)
            del o
        z = self.conv_act(inp, f"zqr.{lvl}", 0)
        g = self.W.gru[lvl]
        k, P = self.k, z.p
        k.launch(
            "prep_context",
            k.linear(P * 128),
            z.buf,
            g["bz"],
            g["br"],
            g["bq"],
            L["czr"][lvl],
            L["cq"][lvl],
            P,
        )

    def cnet(self, img: S, L):
        x = self.conv_act(img, "cnet.conv1", 1)
        for li, stride in ((1, 1), (2, 2), (3, 2)):
            x = self.res_block(x, f"cnet.layer{li}.0", stride, stride > 1, True)
            x = self.res_block(x, f"cnet.layer{li}.1", 1, False, True)
        self.context_head(x, 0, "outputs08", L)
        x = self.res_block(x, "cnet.layer4.0", 2, True, True)
        x = self.res_block(x, "cnet.layer4.1", 1, False, True)
        self.context_head(x, 1, "outputs16", L)
        x = self.res_block(x, "cnet.layer5.0", 2, True, True)
        x = self.res_block(x, "cnet.layer5.1", 1, False, True)
        self.context_head(x, 2, "outputs32", L)

    # ---- update loop
    def resample(self, kind, L, src, dst, slot):
        """pool2x / bilinear of level src's h into x-slot `slot` of level dst (both its [h | x] and [rh | x] buffers)."""
        sb, da, db = L["A"][src], L["A"][dst], L["B"][dst]
        hs, ws, _ = sb.shape
        hd, wd, wide = da.shape
        off = 2 * (128 + slot)
        self.k.launch(
            kind,
            self.k.linear(hd * wd * 16),
            sb,
            sb.shape[2],
            hs,
            ws,
            da.data_ptr() + off,
            db.data_ptr() + off,
            wide,
            hd,
            wd,
            128,
        )

    def gru(self, L, lvl):
        A, B, g, k = L["A"][lvl], L["B"][lvl], self.W.gru[lvl], self.k
        h, w, wide = A.shape
        P = h * w
        zr, q = L["zr"][lvl], L["q"][lvl]
        if self.fused is not None:
            ep = L.setdefault("epi", {}).get(lvl)
            if ep is None:  # launch arguments are the same every iteration: built once per pass
                fc, E = self.fused, self._epi
                h0 = L["h0"].data_ptr() if lvl == 0 else None
                cfg = fused_cfg_zr(P)
                zr_call = (
                    None
                    if cfg is None
                    else fc.bind(  # z -> zr[:, :128], r*h -> B[:, :128]
                        cfg,
                        A,
                        g["zr"],
                        h,
                        w,
                        wide,
                        256,
                        E(
                            0,
                            0,
                            L["czr"][lvl].data_ptr(),
                            256,
                            zr.data_ptr(),
                            256,
                            A.data_ptr(),
                            wide,
                            B.data_ptr(),
                            None,
                            None,
                            None,
                        ),
                    )
                )
                q_call = fc.bind(  # h updated in place in A (not read by the q conv), + packed copy h0 at 1/4
                    fused_cfg_q(P),
                    B,
                    g["q"],
                    h,
                    w,
                    wide,
                    128,
                    E(
                        1,
                        0,
                        L["cq"][lvl].data_ptr(),
                        128,
                        zr.data_ptr(),
                        256,
                        A.data_ptr(),
                        wide,
                        None,
                        h0,
                        None,
                        None,
                    ),
                )
                ep = L["epi"][lvl] = (zr_call, q_call)
            zr_call, q_call = ep
            if zr_call is None:
                self.convs(S(A), g["zr"], S(zr))
                k.launch("gru_zr", k.linear(P * 16), zr, L["czr"][lvl], A, B, wide, P)
            else:
                zr_call()
            q_call()
            return
        self.convs(S(A), g["zr"], S(zr))  # z|r = conv([h | x])
        k.launch(
            "gru_zr", k.linear(P * 16), zr, L["czr"][lvl], A, B, wide, P
        )  # z; r*h -> B[:, :128]
        self.convs(S(B), g["q"], S(q))  # q = conv([r*h | x])
        k.launch(
            "gru_update",
            k.linear(P * 16),
            q,
            L["cq"][lvl],
            zr,
            A,
            wide,
            L["h0"] if lvl == 0 else 0,
            P,
        )

    def motion(self, L):
        k, W = self.k, self.W
        h2, w2, P = L["h2"], L["w2"], L["p2"]
        A0, B0, t64, t128 = L["A"][0], L["B"][0], S(L["t64"]), S(L["t128"])
        k.launch("corr_lookup", k.linear(P), *L["corr"], *L["corr_w"], L["coords"], L["look"], P)
        self.conv_act(S(L["look"]), "convc1", 1, out=S(L["mc1"]), tmp=S(L["mc1"]))
        self.conv_act(S(L["mc1"]), "convc2", 1, out=S(L["mot_in"], 0, 64), tmp=t64)
        k.launch(
            "convf1_direct",
            (((w2 + 15) // 16, (h2 + 15) // 16, 1), (16, 16, 1)),
            L["coords"],
            h2,
            w2,
            W.convf1_w,
            W.convf1_b,
            L["f1"],
        )
        self.conv_act(S(L["f1"]), "convf2", 1, out=S(L["mot_in"], 64, 64), tmp=t64)
        # motion features [relu(conv) (126) | flow_x, 0] into the x-slot of both level-0 GRU buffers
        self.conv_act(
            S(L["mot_in"]),
            "mconv",
            1,
            out=S(A0, 128, 128),
            out2=S(B0, 128, 128),
            flow=(L["coords"], w2),
            tmp=t128,
        )

    def flow_head(self, L):
        k, W = self.k, self.W
        self.conv(S(L["h0"]), "fh1", out=S(L["fh"]))  # bias + ReLU fused into the output kernel
        k.launch(
            "flow_head_out_br",
            (((L["p2"] * 32 + 255) // 256, 1, 1), (256, 1, 1)),
            L["fh"],
            W.t["fh1"],
            L["h2"],
            L["w2"],
            W.fh2_w,
            W.fh2_b,
            L["coords"],
        )

    def iteration(self, L):
        # the motion branch (needs only the disparity) runs on a side stream concurrently with the coarse GRUs
        main = torch.cuda.current_stream()
        if self.concurrent:
            self.side.wait_stream(main)
            with torch.cuda.stream(self.side):
                self.motion(L)
        self.resample("pool2x_dual", L, 1, 2, 0)
        self.gru(L, 2)
        self.resample("pool2x_dual", L, 0, 1, 0)
        self.resample("interp_dual", L, 2, 1, 128)
        self.gru(L, 1)
        if self.concurrent:
            main.wait_stream(self.side)
        else:
            self.motion(L)
        self.resample("interp_dual", L, 1, 0, 128)
        self.gru(L, 0)
        self.flow_head(L)

    # ---- full pass
    @torch.inference_mode()
    def __call__(self, left, right, iters=None):
        """left, right: uint8 [H, W, 3] RGB (numpy or torch, CPU or GPU). Returns fp32 [H, W] disparity (px, >= 0 = left view)."""
        return -self.flow(left, right, iters)

    @torch.inference_mode()
    def flow(self, left, right, iters=None):
        """As __call__, but returns upstream's convention: flow_x = -disparity."""
        iters = self.iters if iters is None else iters
        imgs = []
        for im in (left, right):
            im = torch.as_tensor(im)
            assert im.dtype == torch.uint8 and im.dim() == 3 and im.shape[2] == 3, (
                "expects uint8 [H, W, 3] RGB"
            )
            imgs.append(im.cuda(non_blocking=True).contiguous())
        H, W = imgs[0].shape[:2]
        assert imgs[1].shape[:2] == (H, W)
        pad_h, pad_w = (-H) % 32, (-W) % 32
        top, left_ = pad_h // 2, pad_w // 2  # upstream InputPadder ('sintel' mode, divis_by=32)
        hp, wp = H + pad_h, W + pad_w
        h2, w2 = hp // 4, wp // 4
        sizes = [(h2, w2), (h2 // 2, w2 // 2), (h2 // 4, w2 // 4)]
        P = h2 * w2
        k = self.k

        # feature encoder, one image at a time
        fmaps = [self._new(h2, w2, 256) for _ in range(2)]
        for i in range(2):
            self.fnet(self.prep_image(imgs[i], hp, wp, top, left_), S(fmaps[i]))
        L = dict(h2=h2, w2=w2, p2=P)
        # per level: A = [h | x] (+ packed copy h0 of the 1/4 h), context terms czr, cq: allocated in context_head
        L["A"], L["czr"], L["cq"] = [None] * 3, [None] * 3, [None] * 3
        self.cnet(self.prep_image(imgs[0], hp, wp, top, left_), L)
        del imgs

        # correlation pyramid
        L["corr_w"] = [w2 >> level for level in range(4)]
        L["corr"] = [torch.empty(P * cw, device="cuda", dtype=torch.half) for cw in L["corr_w"]]
        t = (w2 + 127) // 128
        k.launch(
            "corr_gemm", ((t, t, h2), (256, 1, 1)), fmaps[0], fmaps[1], L["corr"][0], w2, 1.0 / 16.0
        )
        del fmaps
        for level in range(3):
            k.launch(
                "corr_pool",
                k.linear(P * L["corr_w"][level + 1]),
                L["corr"][level],
                L["corr"][level + 1],
                P,
                L["corr_w"][level],
                L["corr_w"][level + 1],
            )
        L["coords"] = torch.empty(P, device="cuda", dtype=torch.float32)
        k.launch("init_coords", k.linear(P), L["coords"], w2, P)

        # loop buffers
        L["B"] = [self._new(h, w, 128 + cx) for (h, w), cx in zip(sizes, CX)]  # [r*h | x]
        L["zr"] = [self._new(h, w, 256) for h, w in sizes]
        L["q"] = [self._new(h, w, 128) for h, w in sizes]
        L["look"] = self._new(h2, w2, 40)
        L["mc1"] = self._new(h2, w2, 64)
        L["mot_in"] = self._new(h2, w2, 128)
        L["f1"] = self._new(h2, w2, 64)
        L["t64"] = self._new(h2, w2, 64)
        L["t128"] = L["q"][0]  # motion conv output: consumed before gru(0) writes q[0]
        L["fh"] = L["zr"][0]  # flow head hidden: zr[0] is dead after gru(0)
        for _ in range(iters):
            self.iteration(L)
        for name in (
            "zr",
            "q",
            "look",
            "mc1",
            "mot_in",
            "f1",
            "t64",
            "t128",
            "corr",
            "czr",
            "cq",
            "B",
        ):
            del L[name]

        # mask head + convex upsampling (unpadded output)
        m1 = self.conv_act(S(L["h0"]), "mask1", 1, out=S(L["fh"]), tmp=S(L["fh"]))
        m2 = self.conv_act(m1, "mask2", 0)
        out = torch.empty(H, W, device="cuda", dtype=torch.float32)
        k.launch(
            "upsample_out", k.linear(H * W), L["coords"], m2.buf, h2, w2, top, left_, H, W, out, 0
        )
        return out

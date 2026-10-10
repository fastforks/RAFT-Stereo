// Size-independent implicit-GEMM convolutions with fused epilogues (CUTLASS 2.x API, sm_80-style mma.sync with fp16
// accumulation), compiled ONCE ahead of time into core/prebuilt/libcutlass_conv_*.so -- no per-size compilation or
// tuning: the problem size is a runtime argument. Used by core/general.py (RAFT_FUSED=1) for the GRU convs.
//
// The epilogue is an EpilogueWithVisitor: the visitor gets each thread's 8-channel accumulator vectors with their
// (pixel, channel) coordinates and does the GRU gate / state update / bias+activation directly from registers, so
// the raw conv output never goes to memory.
//   mode 0 (GRU z|r conv, K = 256): s = acc + c[p, n] (c = czr, stride cs);
//           n < 128: z = sigmoid(s) -> z[p*zs + n];  n >= 128: rh = sigmoid(s) * h[p*hs + n-128] -> rh[p*hs + n-128]
//   mode 1 (GRU q conv, K = 128):   q = tanh(acc + c[p, n]) (c = cq); hn = h + z (q - h) with z = z[p*zs + n],
//           h = h[p*hs + n] -> written back to h (in place) and to h2[p*128 + n] if h2 != null
//   mode 2 (affine): v = act(acc * s[n] + t[n]) (s == null: 1, t == null: 0; act 0 none, 1 relu, 2 tanh,
//           3 relu(fp16(relu(v)) + r) with the residual r = c[p*cs + n]) -> z[p*zs + n], also -> rh[p*hs + n] if
//           rh != null; coords != null: the last two channels become [coords[p] - p % W, 0] (affine_act's flow input)
// Numerics match core/raft_kernels.cu's gru_zr / gru_update / affine_act applied to an fp16 conv output.
// From exp 003 S2M2 (s2m2-general bba633d): swizzle-8 rasterization (cfg 0/1; identity raster = cfg 2/3), noinline
// epilogue math.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

extern "C" struct EpiArgs {
    int mode, act;
    const __half* c; int cs;
    __half* z; int zs;
    __half* h; int hs;
    __half* rh;
    __half* h2;
    const float* s;
    const float* t;
    const float* coords; int W;
};


// Epilogue math of one 8-channel vector per mode. Not inlined (origin: exp 003 S2M2 s2m2_cutlass.cu): the epilogue is
// unrolled over all accumulator fragments, and inlining the expf / tanhf math into every copy thrashes the instruction
// cache (short-K convs +0.2 ms). -DEPI_INLINE restores inlining (A/B only).
#ifdef EPI_INLINE
#define EPI_FN __device__ __forceinline__
#else
#define EPI_FN __device__ __noinline__
#endif
#define H8(v) reinterpret_cast<const __half*>(&(v))

// mode 0, n < 128: z = sigmoid(acc + c)
EPI_FN uint4 epi_z8(uint4 acc, uint4 cv) {
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
        o[k] = __float2half(1.f / (1.f + __expf(-(__half2float(H8(acc)[k]) + __half2float(H8(cv)[k])))));
    return *reinterpret_cast<const uint4*>(o);
}

// mode 0, n >= 128: r * h with r = sigmoid(acc + c)
EPI_FN uint4 epi_rh8(uint4 acc, uint4 cv, uint4 hv) {
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float r = 1.f / (1.f + __expf(-(__half2float(H8(acc)[k]) + __half2float(H8(cv)[k]))));
        o[k] = __hmul(__float2half(r), H8(hv)[k]);
    }
    return *reinterpret_cast<const uint4*>(o);
}

// mode 1: h + z (tanh(acc + c) - h)
EPI_FN uint4 epi_q8(uint4 acc, uint4 cv, uint4 zv, uint4 hv) {
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float q = tanhf(__half2float(H8(acc)[k]) + __half2float(H8(cv)[k]));
        float hf = __half2float(H8(hv)[k]), z = __half2float(H8(zv)[k]);
        o[k] = __float2half(hf + z * (q - hf));
    }
    return *reinterpret_cast<const uint4*>(o);
}

// mode 2: act(acc * s + t) (s, t already offset by n; null: 1 / 0), act 3 with the residual rv
EPI_FN uint4 epi_affine8(uint4 acc, const float* s, const float* t, int act, uint4 rv) {
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float v = __half2float(H8(acc)[k]);
        if (s) v *= s[k];
        if (t) v += t[k];
        if (act == 1) v = fmaxf(v, 0.f);
        else if (act == 2) v = tanhf(v);
        __half hv = __float2half(act == 3 ? fmaxf(v, 0.f) : v);
        if (act == 3) { hv = __hadd(hv, H8(rv)[k]); hv = __hgt(hv, __float2half(0.f)) ? hv : __float2half(0.f); }
        o[k] = hv;
    }
    return *reinterpret_cast<const uint4*>(o);
}

template <class OutIt>
struct GruVisitor {
    static int const kIterations = OutIt::kIterations;
    static int const kElementsPerAccess = OutIt::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "8 channels per access");
    using ElementAccumulator = Hf;
    using AccumulatorFragment = cutlass::Array<Hf, kElementsPerAccess>;
    using ThreadMap = typename OutIt::ThreadMap;
    struct SharedStorage {};

    EpiArgs const& a;
    OutIt it;
    int M, N;
    // epilogue operands of the current step, loaded in begin_step (before the accumulator exchange through shared
    // memory) so their latency overlaps it: pc = c[p, n], px = h[p, n-128] (mode 0, n >= 128) or z (mode 1), ph = h
    static int const kFrags = ThreadMap::Iterations::kCount;
    uint4 pc[kFrags], px[kFrags], ph[kFrags];

    CUTLASS_DEVICE GruVisitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                              int thread_idx, cutlass::MatrixCoord tb_offset)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()) {}

    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
        if (a.mode > 1) return;
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(i);
            long long p = co.row();
            int n = co.column();
            if (p < M && n < N) {
                pc[i] = __ldg(reinterpret_cast<const uint4*>(a.c + p * a.cs + n));
                if (a.mode == 0) {
                    if (n >= 128) px[i] = *reinterpret_cast<const uint4*>(a.h + p * a.hs + n - 128);
                } else {
                    px[i] = *reinterpret_cast<const uint4*>(a.z + p * a.zs + n);
                    ph[i] = *reinterpret_cast<const uint4*>(a.h + p * a.hs + n);
                }
            }
        }
    }
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void end_step(int) { ++it; }
    CUTLASS_DEVICE void end_epilogue() {}

    CUTLASS_DEVICE void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
        cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(frag_idx);
        long long p = co.row();
        int n = co.column();
        if (p >= M || n >= N) return;
        uint4 av = *reinterpret_cast<const uint4*>(&acc);
        if (a.mode == 0) {
            if (n < 128) *reinterpret_cast<uint4*>(a.z + p * a.zs + n) = epi_z8(av, pc[frag_idx]);
            else *reinterpret_cast<uint4*>(a.rh + p * a.hs + n - 128) = epi_rh8(av, pc[frag_idx], px[frag_idx]);
        } else if (a.mode == 1) {
            uint4 o = epi_q8(av, pc[frag_idx], px[frag_idx], ph[frag_idx]);
            *reinterpret_cast<uint4*>(a.h + p * a.hs + n) = o;
            if (a.h2) *reinterpret_cast<uint4*>(a.h2 + p * 128 + n) = o;
        } else {
            uint4 rv = make_uint4(0, 0, 0, 0);
            if (a.act == 3) rv = *reinterpret_cast<const uint4*>(a.c + p * a.cs + n);
            __align__(16) __half o[8];
            *reinterpret_cast<uint4*>(o) = epi_affine8(av, a.s ? a.s + n : nullptr, a.t ? a.t + n : nullptr, a.act, rv);
            if (a.coords && n + 8 == N) {
                o[6] = __float2half(a.coords[p] - (float)(p % a.W));
                o[7] = __float2half(0.f);
            }
            *reinterpret_cast<uint4*>(a.z + p * a.zs + n) = *reinterpret_cast<const uint4*>(o);
            if (a.rh) *reinterpret_cast<uint4*>(a.rh + p * a.hs + n) = *reinterpret_cast<const uint4*>(o);
        }
    }
};

template <class TB, class WS, int Stages, int Swz = 1>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using Visitor = GruVisitor<OutIt>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        Visitor, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
};

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) fused_conv_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    cutlass::MatrixCoord extent(prm.problem_size.N * prm.problem_size.P * prm.problem_size.Q, prm.problem_size.K);
    typename F::Visitor v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run(const void* x, const void* w, int H, int W, int C, int K, int R, int stride, EpiArgs const& e,
               cudaStream_t s) {
    using Base = typename F::Base;
    int pad = R / 2;
    int Ho = (H + 2 * pad - R) / stride + 1, Wo = (W + 2 * pad - R) / stride + 1;
    cutlass::conv::Conv2dProblemSize p({1, H, W, C}, {K, R, R, C}, {pad, pad, pad, pad}, {stride, stride}, {1, 1},
                                       {1, Ho, Wo, K}, cutlass::conv::Mode::kCrossCorrelation, 1);
    typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({1, H, W, C})}, {(Hf*)w, NHWC::packed({K, R, R, C})},
                                  {nullptr, NHWC::packed({1, Ho, Wo, K})}, {nullptr, NHWC::packed({1, Ho, Wo, K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(fused_conv_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr_set = true;
    }
    fused_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

// tile configurations (compiled once; general.py picks one per call from the problem size by a fixed rule)
extern "C" int fused_conv(int cfg, const void* x, const void* w, int H, int W, int C, int K, int R, int stride,
                          const EpiArgs* e, void* stream) {
    cudaStream_t s = (cudaStream_t)stream;
    switch (cfg) {  // 128x128x32, 3 stages = the tile config cuDNN's own precompiled fprop kernel uses here
        // swizzle 8 (origin: exp 003 S2M2): tiles that share activation rows run together -> activations read from DRAM
        // once instead of once per 128-channel output tile
        case 0: return run<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 3, 8>>(x, w, H, W, C, K, R, stride, *e, s);
        case 1: return run<Fused<GS<64, 128, 64>, GS<32, 64, 64>, 4, 8>>(x, w, H, W, C, K, R, stride, *e, s);  // small P
        case 2: return run<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 3, 1>>(x, w, H, W, C, K, R, stride, *e, s);  // identity raster (before)
        case 3: return run<Fused<GS<64, 128, 64>, GS<32, 64, 64>, 4, 1>>(x, w, H, W, C, K, R, stride, *e, s);
    }
    return -1;
}

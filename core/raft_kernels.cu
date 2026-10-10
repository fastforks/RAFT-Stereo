// Copied from the Rust engine (experiments/002-raft-stereo/rust/kernels/raft.cu); used by core/fast_loop.py.
// RAFT-Stereo inference kernels (everything except convolutions and the correlation GEMM).
// Layout: fp16 NHWC everywhere; a "slice" is (pointer to first channel, pixel stride in elements).
// Numerics follow the PyTorch fp16 reference (pure .half() model, fp32 disparity state).
#include <cuda_fp16.h>
#include <stdint.h>

#define GRID_STRIDE(i, n) for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (long long)blockDim.x * gridDim.x)

// ---------------------------------------------------------------------------------------------------------------
// Input: uint8 HWC RGB -> fp16 NHWC with 3 channels (cuDNN's small-channel 7x7 kernels are fastest at C=3), replicate-padded
// to (Hp, Wp) with the top/left offsets of upstream's InputPadder ('sintel' mode), value = 2*(v/255)-1 (fp32).
// mirror != 0: the image is mirrored horizontally on load (right-view pass of the both-views mode).
extern "C" __global__ void prep_image(const uint8_t* __restrict__ src, __half* __restrict__ dst, int H, int W, int Hp,
                                      int Wp, int top, int left, int mirror) {
    GRID_STRIDE(i, (long long)Hp * Wp) {
        int y = (int)(i / Wp), x = (int)(i % Wp);
        int sy = min(max(y - top, 0), H - 1), sx = min(max(x - left, 0), W - 1);
        if (mirror) sx = W - 1 - sx;
        const uint8_t* p = src + ((long long)sy * W + sx) * 3;
#pragma unroll
        for (int c = 0; c < 3; ++c) dst[i * 3 + c] = __float2half(2.f * ((float)p[c] / 255.f) - 1.f);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// InstanceNorm statistics. Pass 1: per (pixel chunk, channel) fp32 sums of (x - a) and (x - a)^2 around a = the chunk's
// first pixel (a value of the same channel: no cancellation for large means) -> (mean, M2) per chunk; blockDim.x == C.
extern "C" __global__ void in_stats_partial(const __half* __restrict__ x, long long P, int C, int chunk,
                                            float* __restrict__ partial) {
    // 16-byte loads: thread (row, c8) accumulates channels [8*c8, 8*c8+8) over every rows-th pixel of the chunk;
    // rows = blockDim.x / (C/8) pixel rows in flight per block; then a shared-memory reduction over the rows.
    __shared__ float red[2][256 * 8];
    const int CV = C >> 3, rows = blockDim.x / CV;
    const int t = threadIdx.x, c8 = t % CV, row = t / CV;
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, ss[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    long long p0 = (long long)blockIdx.x * chunk, p1 = min(P, p0 + chunk);
    float a[8];
    {
        uint4 v = *reinterpret_cast<const uint4*>(x + p0 * C + c8 * 8);
        const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
        for (int k = 0; k < 4; ++k) { float2 f = __half22float2(h2[k]); a[2 * k] = f.x; a[2 * k + 1] = f.y; }
    }
    if (row < rows) {
        for (long long p = p0 + row; p < p1; p += rows) {
            uint4 v = *reinterpret_cast<const uint4*>(x + p * C + c8 * 8);
            const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __half22float2(h2[k]);
                float dx = f.x - a[2 * k], dy = f.y - a[2 * k + 1];
                s[2 * k] += dx; ss[2 * k] = fmaf(dx, dx, ss[2 * k]);
                s[2 * k + 1] += dy; ss[2 * k + 1] = fmaf(dy, dy, ss[2 * k + 1]);
            }
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            red[0][row * C + c8 * 8 + k] = s[k];
            red[1][row * C + c8 * 8 + k] = ss[k];
        }
    }
    __syncthreads();
    const float inv = 1.f / (float)(p1 - p0);
    for (int c = t; c < C; c += blockDim.x) {
        float sa = 0.f, sb = 0.f;
        for (int r = 0; r < rows; ++r) { sa += red[0][r * C + c]; sb += red[1][r * C + c]; }
        const float m = sa * inv;
        const float ac = __half2float(x[p0 * C + c]);  // the anchor of channel c
        partial[(long long)blockIdx.x * 2 * C + c] = ac + m;
        partial[(long long)blockIdx.x * 2 * C + C + c] = fmaxf(sb - sa * m, 0.f);
    }
}

// Pass 2: merge the chunks' (mean, M2) with Chan's formula in a fixed order (fp32 is enough: no cancellation) -> mean and
// 1/sqrt(var + eps) (biased variance, as InstanceNorm).
__device__ __forceinline__ void chan_merge(float& na, float& ma, float& qa, float nb, float mb, float qb) {
    if (nb == 0.f) return;
    const float n = na + nb, d = mb - ma, f = nb / n;
    ma += d * f;
    qa += qb + d * d * na * f;
    na = n;
}
extern "C" __global__ void in_stats_final(const float* __restrict__ partial, int nblk, int C, long long P, int chunk, float eps,
                                          float* __restrict__ mean_rstd) {
    // one warp per channel (grid = C, block = 32): lane l merges chunks l, l + 32, ... in order, then a shuffle tree
    int c = blockIdx.x, lane = threadIdx.x;
    float n = 0.f, m = 0.f, q = 0.f;
    for (int bk = lane; bk < nblk; bk += 32)
        chan_merge(n, m, q, (float)min((long long)chunk, P - (long long)bk * chunk), partial[(long long)bk * 2 * C + c],
                   partial[(long long)bk * 2 * C + C + c]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float nb = __shfl_down_sync(0xffffffffu, n, o), mb = __shfl_down_sync(0xffffffffu, m, o),
                    qb = __shfl_down_sync(0xffffffffu, q, o);
        if (lane < o) chan_merge(n, m, q, nb, mb, qb);
    }
    if (lane == 0) {
        mean_rstd[c] = m;
        mean_rstd[C + c] = rsqrtf(q / (float)P + eps);
    }
}

__device__ __forceinline__ __half hrelu(__half v) { return __hgt(v, __float2half(0.f)) ? v : __float2half(0.f); }

// out = relu(IN(x))                              mode 0
// out = relu(relu(IN(x)) + IN(r))                mode 1 (residual through a normalized 1x1 downsample conv)
// out = relu(relu(IN(x)) + r)                    mode 2 (identity residual)
// out may alias x (elementwise, in place).
extern "C" __global__ void in_apply(const __half* x, const float* __restrict__ st, const __half* __restrict__ r,
                                    const float* __restrict__ st_r, __half* out, long long n, int C, int mode) {
    // 8 channels per thread (C is a multiple of 8); 16-byte loads/stores
    GRID_STRIDE(v, n / 8) {
        long long i = v * 8;
        int c0 = (int)(i % C);
        uint4 xv = *reinterpret_cast<const uint4*>(x + i);
        const __half* xh = reinterpret_cast<const __half*>(&xv);
        uint4 rv = make_uint4(0, 0, 0, 0);
        if (mode != 0) rv = *reinterpret_cast<const uint4*>(r + i);
        const __half* rh = reinterpret_cast<const __half*>(&rv);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int c = c0 + k;
            __half a = hrelu(__float2half((__half2float(xh[k]) - st[c]) * st[C + c]));
            if (mode == 1) a = hrelu(__hadd(a, __float2half((__half2float(rh[k]) - st_r[c]) * st_r[C + c])));
            else if (mode == 2) a = hrelu(__hadd(a, rh[k]));
            o[k] = a;
        }
        *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(o);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Correlation pyramid: out[row][i] = mean(in[row][2i], in[row][2i+1])   (avg_pool2d [1,2])
extern "C" __global__ void corr_pool(const __half* __restrict__ in, __half* __restrict__ out, long long rows, int win,
                                     int wout) {
    GRID_STRIDE(i, rows * wout) {
        long long row = i / wout;
        int j = (int)(i % wout);
        const __half* p = in + row * win + 2 * j;
        out[i] = __float2half((__half2float(p[0]) + __half2float(p[1])) * 0.5f);
    }
}

// Correlation lookup (same arithmetic as upstream's CUDA sampler, fp16 accumulation): for each pixel and level l,
// 9 linearly interpolated taps around x/2^l. Output: 40 channels (36 used, 4 zero padding) NHWC.
extern "C" __global__ void corr_lookup(const __half* __restrict__ v0, const __half* __restrict__ v1,
                                       const __half* __restrict__ v2, const __half* __restrict__ v3, int w0, int w1,
                                       int w2, int w3, const float* __restrict__ coords_x, __half* __restrict__ out,
                                       long long P) {
    GRID_STRIDE(p, P) {
        const __half* vols[4] = {v0 + p * w0, v1 + p * w1, v2 + p * w2, v3 + p * w3};
        const int ws[4] = {w0, w1, w2, w3};
        float x = coords_x[p];
        __align__(16) __half o[40];
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            float xl = x / (float)(1 << l);
            float fl = floorf(xl);
            float dx = xl - fl;
            int x0 = (int)fl;
            __half hdx = __float2half(dx), h1dx = __float2half(1.f - dx);
            __half acc[9];
#pragma unroll
            for (int i = 0; i < 9; ++i) acc[i] = __float2half(0.f);
#pragma unroll
            for (int i = 0; i < 10; ++i) {
                int x1 = x0 - 4 + i;
                if (x1 >= 0 && x1 < ws[l]) {
                    __half s = vols[l][x1];
                    if (i > 0) acc[i - 1] = __hadd(acc[i - 1], __hmul(s, hdx));
                    if (i < 9) acc[i] = __hadd(acc[i], __hmul(s, h1dx));
                }
            }
#pragma unroll
            for (int i = 0; i < 9; ++i) o[l * 9 + i] = acc[i];
        }
#pragma unroll
        for (int c = 36; c < 40; ++c) o[c] = __float2half(0.f);
        uint4* dst = reinterpret_cast<uint4*>(out + p * 40);
#pragma unroll
        for (int k = 0; k < 5; ++k) dst[k] = reinterpret_cast<const uint4*>(o)[k];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// avg_pool2d(kernel 3, stride 2, padding 1, count_include_pad) of a C-channel slice into another slice.
__device__ __forceinline__ void pool2x_body(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                            int dstride, __half* dst2, int Hd, int Wd, int C) {
    const int CV = C / 8;  // 8 channels per thread
    GRID_STRIDE(i, (long long)Hd * Wd * CV) {
        int c = (int)(i % CV) * 8;
        long long pd = i / CV;
        int y = (int)(pd / Wd), x = (int)(pd % Wd);
        float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            int sy = 2 * y - 1 + dy;
            if (sy < 0 || sy >= Hs) continue;
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                int sx = 2 * x - 1 + dx;
                if (sx < 0 || sx >= Ws) continue;
                uint4 v = *reinterpret_cast<const uint4*>(src + ((long long)sy * Ws + sx) * sstride + c);
                const __half* h = reinterpret_cast<const __half*>(&v);
#pragma unroll
                for (int k = 0; k < 8; ++k) s[k] += __half2float(h[k]);
            }
        }
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(s[k] / 9.f);
        *reinterpret_cast<uint4*>(dst + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
        if (dst2) *reinterpret_cast<uint4*>(dst2 + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
    }
}
extern "C" __global__ void pool2x_into(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                       int dstride, int Hd, int Wd, int C) {
    pool2x_body(src, sstride, Hs, Ws, dst, dstride, nullptr, Hd, Wd, C);
}
// general path: the result goes to two slices with the same pixel stride (the [h | x] and [r*h | x] buffers)
extern "C" __global__ void pool2x_dual(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                       __half* dst2, int dstride, int Hd, int Wd, int C) {
    pool2x_body(src, sstride, Hs, Ws, dst, dstride, dst2, Hd, Wd, C);
}

// Bilinear upsampling with align_corners=True (PyTorch formula) of a C-channel slice into another slice.
__device__ __forceinline__ void interp_body(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                            int dstride, __half* dst2, int Hd, int Wd, int C) {
    float rh = Hd > 1 ? (float)(Hs - 1) / (float)(Hd - 1) : 0.f;
    float rw = Wd > 1 ? (float)(Ws - 1) / (float)(Wd - 1) : 0.f;
    const int CV = C / 8;  // 8 channels per thread
    GRID_STRIDE(i, (long long)Hd * Wd * CV) {
        int c = (int)(i % CV) * 8;
        long long pd = i / CV;
        int y = (int)(pd / Wd), x = (int)(pd % Wd);
        float h1r = rh * (float)y, w1r = rw * (float)x;
        int h1 = (int)h1r, w1 = (int)w1r;
        int h1p = h1 < Hs - 1 ? 1 : 0, w1p = w1 < Ws - 1 ? 1 : 0;
        float h1l = h1r - (float)h1, w1l = w1r - (float)w1;
        float h0l = 1.f - h1l, w0l = 1.f - w1l;
        uint4 a = *reinterpret_cast<const uint4*>(src + ((long long)h1 * Ws + w1) * sstride + c);
        uint4 b = *reinterpret_cast<const uint4*>(src + ((long long)h1 * Ws + w1 + w1p) * sstride + c);
        uint4 cc = *reinterpret_cast<const uint4*>(src + ((long long)(h1 + h1p) * Ws + w1) * sstride + c);
        uint4 d = *reinterpret_cast<const uint4*>(src + ((long long)(h1 + h1p) * Ws + w1 + w1p) * sstride + c);
        const __half *v00 = reinterpret_cast<const __half*>(&a), *v01 = reinterpret_cast<const __half*>(&b);
        const __half *v10 = reinterpret_cast<const __half*>(&cc), *v11 = reinterpret_cast<const __half*>(&d);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o[k] = __float2half(h0l * (w0l * __half2float(v00[k]) + w1l * __half2float(v01[k])) +
                                h1l * (w0l * __half2float(v10[k]) + w1l * __half2float(v11[k])));
        *reinterpret_cast<uint4*>(dst + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
        if (dst2) *reinterpret_cast<uint4*>(dst2 + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
    }
}
extern "C" __global__ void interp_into(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                       int dstride, int Hd, int Wd, int C) {
    interp_body(src, sstride, Hs, Ws, dst, dstride, nullptr, Hd, Wd, C);
}
extern "C" __global__ void interp_dual(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                       __half* dst2, int dstride, int Hd, int Wd, int C) {
    interp_body(src, sstride, Hs, Ws, dst, dstride, dst2, Hd, Wd, C);
}

// ---------------------------------------------------------------------------------------------------------------
// Flow (disparity) features. flow_x = coords_x - x in fp32, rounded to fp16 as the network input.
// Writes channels [flow_x, 0] into a slice (the last two motion-feature channels of the gru08 input).
extern "C" __global__ void write_flow(const float* __restrict__ coords_x, int W, long long P, __half* dst, int dstride) {
    GRID_STRIDE(p, P) {
        float fx = coords_x[p] - (float)(p % W);
        __half2 v = __halves2half2(__float2half(fx), __float2half(0.f));
        *reinterpret_cast<__half2*>(dst + p * dstride) = v;
    }
}

// convf1: 7x7 conv on the 2-channel flow whose y-channel is always 0 -> only the x-channel weights matter.
// out[p][k] = relu(b[k] + sum w[k][ky][kx] * flow_x), 64 output channels, fp32 accumulation, weights rounded to fp16.
#define F1_TILE 16
extern "C" __global__ void convf1_direct(const float* __restrict__ coords_x, int H, int W, const float* __restrict__ wx,
                                         const float* __restrict__ bias, __half* __restrict__ out) {
    __shared__ float tile[F1_TILE + 6][F1_TILE + 6];
    __shared__ float wsh[49][64];  // [tap][out channel]: one broadcast float4 read serves 4 outputs
    __shared__ float bsh[64];
    int tx = threadIdx.x, ty = threadIdx.y, t = ty * F1_TILE + tx;
    int x0 = blockIdx.x * F1_TILE, y0 = blockIdx.y * F1_TILE;
    for (int i = t; i < 64 * 49; i += F1_TILE * F1_TILE) wsh[i % 49][i / 49] = __half2float(__float2half(wx[i]));
    if (t < 64) bsh[t] = __half2float(__float2half(bias[t]));
    for (int i = t; i < (F1_TILE + 6) * (F1_TILE + 6); i += F1_TILE * F1_TILE) {
        int ly = i / (F1_TILE + 6), lx = i % (F1_TILE + 6);
        int gy = y0 + ly - 3, gx = x0 + lx - 3;
        float v = 0.f;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W)
            v = __half2float(__float2half(coords_x[(long long)gy * W + gx] - (float)gx));
        tile[ly][lx] = v;
    }
    __syncthreads();
    int y = y0 + ty, x = x0 + tx;
    if (y >= H || x >= W) return;
    float in[49];
#pragma unroll
    for (int ky = 0; ky < 7; ++ky)
#pragma unroll
        for (int kx = 0; kx < 7; ++kx) in[ky * 7 + kx] = tile[ty + ky][tx + kx];
    uint4* dst = reinterpret_cast<uint4*>(out + ((long long)y * W + x) * 64);
#pragma unroll 1
    for (int kb = 0; kb < 64; kb += 8) {
        float acc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = bsh[kb + k];
#pragma unroll
        for (int j = 0; j < 49; ++j) {
            float4 w0 = *reinterpret_cast<const float4*>(&wsh[j][kb]);
            float4 w1 = *reinterpret_cast<const float4*>(&wsh[j][kb + 4]);
            acc[0] += w0.x * in[j]; acc[1] += w0.y * in[j]; acc[2] += w0.z * in[j]; acc[3] += w0.w * in[j];
            acc[4] += w1.x * in[j]; acc[5] += w1.y * in[j]; acc[6] += w1.z * in[j]; acc[7] += w1.w * in[j];
        }
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(fmaxf(acc[k], 0.f));
        dst[kb / 8] = *reinterpret_cast<const uint4*>(o);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// ConvGRU pieces. zr: [P][256] = sigmoid(z) | sigmoid(r) (from the fused cuDNN graph); h lives in a slice.
// gate1: rh = r * h
extern "C" __global__ void gru_gate1(const __half* __restrict__ zr, const __half* __restrict__ h, int hstride,
                                     __half* __restrict__ rh, long long P) {
    GRID_STRIDE(i, P * 16) {  // 8 channels per thread
        long long p = i / 16;
        int c = (int)(i % 16) * 8;
        uint4 rv = *reinterpret_cast<const uint4*>(zr + p * 256 + 128 + c);
        uint4 hv = *reinterpret_cast<const uint4*>(h + p * hstride + c);
        uint4 o;
        const __half2 *r2 = reinterpret_cast<const __half2*>(&rv), *h2 = reinterpret_cast<const __half2*>(&hv);
        __half2* o2 = reinterpret_cast<__half2*>(&o);
#pragma unroll
        for (int k = 0; k < 4; ++k) o2[k] = __hmul2(r2[k], h2[k]);
        *reinterpret_cast<uint4*>(rh + p * 128 + c) = o;
    }
}

// Context terms: zqr [P][384] = cz | cr | cq  ->  czr [P][256] = (cz + bz) | (cr + br),  cq [P][128] = cq + bq
extern "C" __global__ void prep_context(const __half* __restrict__ zqr, const __half* __restrict__ bz,
                                        const __half* __restrict__ br, const __half* __restrict__ bq,
                                        __half* __restrict__ czr, __half* __restrict__ cq, long long P) {
    GRID_STRIDE(i, P * 128) {
        long long p = i / 128;
        int c = (int)(i % 128);
        const __half* s = zqr + p * 384;
        czr[p * 256 + c] = __hadd(s[c], bz[c]);
        czr[p * 256 + 128 + c] = __hadd(s[128 + c], br[c]);
        cq[p * 128 + c] = __hadd(s[256 + c], bq[c]);
    }
}

extern "C" __global__ void init_coords(float* __restrict__ coords_x, int W, long long P) {
    GRID_STRIDE(p, P) { coords_x[p] = (float)(p % W); }
}

// ---------------------------------------------------------------------------------------------------------------
// Convex 4x upsampling of the disparity with the learned mask (softmax over the 3x3 neighbourhood), written
// unpadded as fp32 [H][W] (same sign convention as upstream: flow_x, negative for left-image disparity).
// mask: [h4*w4][144] fp16 raw head output (scaled by 0.25 here), channel = k*16 + a*4 + b.
// mirror != 0: the result is written mirrored back (right-view pass of the both-views mode).
extern "C" __global__ void upsample_out(const float* __restrict__ coords_x, const __half* __restrict__ mask, int h4,
                                        int w4, int top, int left, int H, int W, float* __restrict__ out, int mirror) {
    GRID_STRIDE(i, (long long)H * W) {
        int yo = (int)(i / W), xo = (int)(i % W);
        int Y = yo + top, X = xo + left;
        int h = Y >> 2, a = Y & 3, w = X >> 2, b = X & 3;
        const __half* m = mask + ((long long)h * w4 + w) * 144 + a * 4 + b;
        float mk[9], mx = -1e30f;
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            mk[k] = __half2float(__hmul(__float2half(0.25f), m[k * 16]));
            mx = fmaxf(mx, mk[k]);
        }
        float den = 0.f;
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            mk[k] = expf(mk[k] - mx);
            den += mk[k];
        }
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            int ny = h + k / 3 - 1, nx = w + k % 3 - 1;
            float f = 0.f;
            if (ny >= 0 && ny < h4 && nx >= 0 && nx < w4) f = 4.f * (coords_x[(long long)ny * w4 + nx] - (float)nx);
            acc += __half2float(__float2half(mk[k] / den)) * f;
        }
        out[(long long)yo * W + (mirror ? W - 1 - xo : xo)] = acc;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Flow head output conv (3x3, 256 -> 1: stereo only uses the x channel) fused with the disparity update:
// coords_x[p] += fp16(b + sum_{tap, c} w[tap][c] * fh[neighbour][c]). One warp per pixel; each lane covers 8 of the
// 256 channels with 16-byte loads, then a warp reduction. w: KRSC fp16 [K>=1][3][3][256], channel 0 used.
extern "C" __global__ void flow_head_out(const __half* __restrict__ fh, int H, int W, const __half* __restrict__ w,
                                         const __half* __restrict__ b, float* __restrict__ coords_x) {
    int lane = threadIdx.x & 31;
    long long p = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (p >= (long long)H * W) return;
    int y = (int)(p / W), x = (int)(p % W);
    float acc = 0.f;
#pragma unroll
    for (int t = 0; t < 9; ++t) {
        int ny = y + t / 3 - 1, nx = x + t % 3 - 1;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        uint4 fv = *reinterpret_cast<const uint4*>(fh + ((long long)ny * W + nx) * 256 + lane * 8);
        uint4 wv = *reinterpret_cast<const uint4*>(w + t * 256 + lane * 8);
        const __half2 *f2 = reinterpret_cast<const __half2*>(&fv), *w2 = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 a = __half22float2(f2[k]), c = __half22float2(w2[k]);
            acc += a.x * c.x + a.y * c.y;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) coords_x[p] += __half2float(__float2half(acc + __half2float(b[0])));
}


// ---------------------------------------------------------------------------------------------------------------
// Autotuner support: fills n fp16 values with deterministic pseudo-random numbers in [-1, 1) (hash of index + seed).
extern "C" __global__ void fill_rand(__half* __restrict__ p, long long n, unsigned seed) {
    GRID_STRIDE(i, n) {
        unsigned x = (unsigned)i * 2654435761u ^ seed;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
        p[i] = __float2half((float)(x & 0xffffu) / 32768.f - 1.f);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Correlation volume (replaces the cuBLAS GEMM: no library load / handle / first-call kernel loading at startup):
// out[y][i][j] = fp16(alpha * sum_d f1[y][i][d] * f2[y][j][d]), D = 256, fp16 inputs, fp32 accumulation (tensor
// cores via WMMA). Block = 8 warps computing a 128 (i) x 128 (j) tile of one image row y; K in chunks of 32 with a
// double-buffered shared-memory pipeline (cp.async). grid = (ceil(W/128) over j, ceil(W/128) over i, H).
#include <mma.h>
#define CG_T 128
#define CG_K 32
#define CG_LD (CG_K + 8)  // padded row (halves) against bank conflicts
extern "C" __global__ void __launch_bounds__(256) corr_gemm(const __half* __restrict__ f1, const __half* __restrict__ f2,
                                                             __half* __restrict__ out, int W, float alpha) {
    using namespace nvcuda;
    __shared__ __align__(128) __half sa[2][CG_T * CG_LD];
    __shared__ __align__(128) __half sb[2][CG_T * CG_LD];
    const int y = blockIdx.z, i0 = blockIdx.y * CG_T, j0 = blockIdx.x * CG_T;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const __half* A = f1 + (long long)y * W * 256;
    const __half* B = f2 + (long long)y * W * 256;
    // each thread copies 2 x 16 bytes of A and of B per K chunk: rows t / 4 and t / 4 + 64, 8 halves at col (t % 4) * 8
    auto load = [&](int buf, int k0) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int row = (t >> 2) + r * 64, col = (t & 3) * 8;
            __half* da = &sa[buf][row * CG_LD + col];
            __half* db = &sb[buf][row * CG_LD + col];
            int ia = i0 + row, jb = j0 + row;
            unsigned sda = (unsigned)__cvta_generic_to_shared(da), sdb = (unsigned)__cvta_generic_to_shared(db);
            const __half* ga = A + (long long)min(ia, W - 1) * 256 + k0 + col;
            const __half* gb = B + (long long)min(jb, W - 1) * 256 + k0 + col;
            int na = ia < W ? 16 : 0, nb = jb < W ? 16 : 0;  // zero-fill rows past the image edge
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sda), "l"(ga), "r"(na));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sdb), "l"(gb), "r"(nb));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    // warp tile: 64 (i) x 32 (j) -> 4 x 2 fragments of 16 x 16; warps arranged 2 (i) x 4 (j)
    const int wi = (warp >> 2) * 64, wj = (warp & 3) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) wmma::fill_fragment(acc[a][b], 0.f);
    load(0, 0);
    constexpr int NK = 256 / CG_K;
#pragma unroll 1
    for (int kc = 0; kc < NK; ++kc) {
        int buf = kc & 1;
        if (kc + 1 < NK) {
            load(buf ^ 1, (kc + 1) * CG_K);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < CG_K; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
#pragma unroll
            for (int a = 0; a < 4; ++a) wmma::load_matrix_sync(fa[a], &sa[buf][(wi + a * 16) * CG_LD + kk], CG_LD);
#pragma unroll
            for (int b = 0; b < 2; ++b) wmma::load_matrix_sync(fb[b], &sb[buf][(wj + b * 16) * CG_LD + kk], CG_LD);
#pragma unroll
            for (int a = 0; a < 4; ++a)
#pragma unroll
                for (int b = 0; b < 2; ++b) wmma::mma_sync(acc[a][b], fa[a], fb[b], acc[a][b]);
        }
        __syncthreads();
    }
    // epilogue: each warp stages its fragments through shared memory (reusing sa) and writes fp16 with bounds checks
    float* stage = reinterpret_cast<float*>(&sa[0][0]) + warp * 256;  // 16 x 16 floats per warp (8 KB total)
#pragma unroll
    for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            wmma::store_matrix_sync(stage, acc[a][b], 16, wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int e = lane; e < 256; e += 32) {
                int r = e >> 4, cidx = e & 15;
                int i = i0 + wi + a * 16 + r, j = j0 + wj + b * 16 + cidx;
                if (i < W && j < W) out[((long long)y * W + i) * W + j] = __float2half(alpha * stage[e]);
            }
            __syncwarp();
        }
}

// ---------------------------------------------------------------------------------------------------------------
// General (size-independent) path, core/general.py: plain precompiled cuDNN convs + these elementwise epilogues.
// Per-channel affine + activation of an NHWC slice y (C % 8 == 0, 16-byte aligned), written to slice o and
// optionally also to slice o2 (o may equal y: in place):
//   v = y * s[c] + t[c]   (s == nullptr: scale 1; t == nullptr: shift 0), fp32 math, rounded to fp16
//   mode 0: v      1: relu(v)      2: tanh(v)      3: relu(fp16(relu(v)) + r)   (residual r: a slice, fp16 add)
// coords_x != nullptr: channels C-2, C-1 become [coords_x[p] - p % W, 0] (flow input of the GRU, see write_flow).
extern "C" __global__ void affine_act(const __half* y, int ystride, __half* o, int ostride, __half* o2, int o2stride,
                                      const float* __restrict__ s, const float* __restrict__ t,
                                      const __half* __restrict__ r, int rstride, long long P, int C, int mode,
                                      const float* __restrict__ coords_x, int W) {
    const int CV = C / 8;
    GRID_STRIDE(i, P * CV) {
        long long p = i / CV;
        int c0 = (int)(i % CV) * 8;
        uint4 yv = *reinterpret_cast<const uint4*>(y + p * ystride + c0);
        const __half* yh = reinterpret_cast<const __half*>(&yv);
        uint4 rv = make_uint4(0, 0, 0, 0);
        if (mode == 3) rv = *reinterpret_cast<const uint4*>(r + p * rstride + c0);
        const __half* rh = reinterpret_cast<const __half*>(&rv);
        __align__(16) __half out[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int c = c0 + k;
            float v = __half2float(yh[k]);
            if (s) v *= s[c];
            if (t) v += t[c];
            if (mode == 1) v = fmaxf(v, 0.f);
            else if (mode == 2) v = tanhf(v);
            __half hv = __float2half(mode == 3 ? fmaxf(v, 0.f) : v);
            if (mode == 3) hv = hrelu(__hadd(hv, rh[k]));
            out[k] = hv;
        }
        if (coords_x && c0 + 8 == C) {  // motion features: last two channels = [flow_x, 0] (replaces write_flow)
            out[6] = __float2half(coords_x[p] - (float)(p % W));
            out[7] = __float2half(0.f);
        }
        *reinterpret_cast<uint4*>(o + p * ostride + c0) = *reinterpret_cast<const uint4*>(out);
        if (o2) *reinterpret_cast<uint4*>(o2 + p * o2stride + c0) = *reinterpret_cast<const uint4*>(out);
    }
}

// GRU gates after the z|r conv (zr: [P][256] raw conv output, czr: [P][256] context terms incl. biases):
// z = sigmoid(zr[:128] + cz) written back into zr[:128]; rh = sigmoid(zr[128:] + cr) * h written into the rh slice
// (h and rh slices have the same pixel stride).
extern "C" __global__ void gru_zr(__half* zr, const __half* __restrict__ czr, const __half* __restrict__ h,
                                  __half* __restrict__ rh, int stride, long long P) {
    GRID_STRIDE(i, P * 16) {  // 8 channels per thread
        long long p = i / 16;
        int c = (int)(i % 16) * 8;
        uint4 zv = *reinterpret_cast<const uint4*>(zr + p * 256 + c), rv = *reinterpret_cast<const uint4*>(zr + p * 256 + 128 + c);
        uint4 czv = *reinterpret_cast<const uint4*>(czr + p * 256 + c), crv = *reinterpret_cast<const uint4*>(czr + p * 256 + 128 + c);
        uint4 hv = *reinterpret_cast<const uint4*>(h + p * stride + c);
        const __half *zh = reinterpret_cast<const __half*>(&zv), *rr = reinterpret_cast<const __half*>(&rv);
        const __half *cz = reinterpret_cast<const __half*>(&czv), *cr = reinterpret_cast<const __half*>(&crv);
        const __half* hh = reinterpret_cast<const __half*>(&hv);
        __align__(16) __half oz[8], orh[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float z = 1.f / (1.f + __expf(-(__half2float(zh[k]) + __half2float(cz[k]))));
            float r = 1.f / (1.f + __expf(-(__half2float(rr[k]) + __half2float(cr[k]))));
            oz[k] = __float2half(z);
            orh[k] = __hmul(__float2half(r), hh[k]);
        }
        *reinterpret_cast<uint4*>(zr + p * 256 + c) = *reinterpret_cast<const uint4*>(oz);
        *reinterpret_cast<uint4*>(rh + p * stride + c) = *reinterpret_cast<const uint4*>(orh);
    }
}

// GRU state update after the q conv (q: [P][128] raw conv output, cq: [P][128] context term incl. bias, z at zr[:128]):
// h = (1 - z) h + z tanh(q + cq), in place in the h slice; optionally also written to a packed copy h2 ([P][128]).
extern "C" __global__ void gru_update(const __half* __restrict__ q, const __half* __restrict__ cq, const __half* __restrict__ zr,
                                      __half* h, int stride, __half* h2, long long P) {
    GRID_STRIDE(i, P * 16) {
        long long p = i / 16;
        int c = (int)(i % 16) * 8;
        uint4 qv = *reinterpret_cast<const uint4*>(q + p * 128 + c), cv = *reinterpret_cast<const uint4*>(cq + p * 128 + c);
        uint4 zv = *reinterpret_cast<const uint4*>(zr + p * 256 + c), hv = *reinterpret_cast<const uint4*>(h + p * stride + c);
        const __half *qh = reinterpret_cast<const __half*>(&qv), *ch = reinterpret_cast<const __half*>(&cv);
        const __half *zh = reinterpret_cast<const __half*>(&zv), *hh = reinterpret_cast<const __half*>(&hv);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float qq = tanhf(__half2float(qh[k]) + __half2float(ch[k]));
            float hf = __half2float(hh[k]), z = __half2float(zh[k]);
            o[k] = __float2half(hf + z * (qq - hf));
        }
        *reinterpret_cast<uint4*>(h + p * stride + c) = *reinterpret_cast<const uint4*>(o);
        if (h2) *reinterpret_cast<uint4*>(h2 + p * 128 + c) = *reinterpret_cast<const uint4*>(o);
    }
}

// General path: flow_head_out with the hidden layer's bias + ReLU fused (fh = raw conv output, b1: fp32 [256]).
extern "C" __global__ void flow_head_out_br(const __half* __restrict__ fh, const float* __restrict__ b1, int H, int W,
                                            const __half* __restrict__ w, const __half* __restrict__ b,
                                            float* __restrict__ coords_x) {
    int lane = threadIdx.x & 31;
    long long p = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (p >= (long long)H * W) return;
    int y = (int)(p / W), x = (int)(p % W);
    float bb[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) bb[k] = b1[lane * 8 + k];
    float acc = 0.f;
#pragma unroll
    for (int t = 0; t < 9; ++t) {
        int ny = y + t / 3 - 1, nx = x + t % 3 - 1;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
        uint4 fv = *reinterpret_cast<const uint4*>(fh + ((long long)ny * W + nx) * 256 + lane * 8);
        uint4 wv = *reinterpret_cast<const uint4*>(w + t * 256 + lane * 8);
        const __half *f8 = reinterpret_cast<const __half*>(&fv), *w8 = reinterpret_cast<const __half*>(&wv);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            acc += __half2float(__float2half(fmaxf(__half2float(f8[k]) + bb[k], 0.f))) * __half2float(w8[k]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) coords_x[p] += __half2float(__float2half(acc + __half2float(b[0])));
}

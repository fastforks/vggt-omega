// fastbb: prebuilt CUDA kernels for the shared backbones (DINOv2-style ViTs first).
// Compiled once into a fatbin (SASS sm_80/86/89/90/120 + compute_80 PTX), loaded through the driver API (kernels.py).
// Nothing here depends on the input size.
#include <cuda_fp16.h>
#include <stdint.h>

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    int n = valid ? 16 : 0;  // src-size 0 -> zero fill
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// D(f16) = A * B + C(f16)   m16n8k16
__device__ __forceinline__ void mma_f16(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// D(f32) = A * B + C(f32)
__device__ __forceinline__ void mma_f32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 unpack_h2(uint32_t u) { return __half22float2(*reinterpret_cast<__half2*>(&u)); }
__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// swizzled [rows][64] fp16 tile: 8 chunks of 16 B per row, chunk index XOR (row & 7) -> conflict-free ldmatrix / cp.async
__device__ __forceinline__ int swz(int row, int chunk) { return row * 64 + ((chunk ^ (row & 7)) << 3); }

// ------------------------------------------------------------------------------------------------ flash attention
// O = softmax(scale * Q K^T) V, head_dim 64, no mask, any N (tails masked), fp16 in / out.
// Q, K, V, O: element (b, n, h, d) at ptr[b * sb + n * sn + h * sh + d] (d contiguous; strides in elements, multiples of 8,
// e.g. straight views into the packed [B, N, 3, H, 64] qkv GEMM output, and O written as [B, N, H*64] for the projection).
// FlashAttention-2 schedule: CTA = 4 warps x (16 * RM) query rows; K/V tiles of 64 keys double-buffered with cp.async.
// QK16 / PV16: fp16-accumulate mma for S = Q K^T / for each 64-key block of P V (the running O, the row max and the row sums
// stay fp32; an fp16 accumulation never spans more than 64 products: head_dim for S, one key block for P V).
template <int RM, bool QK16, bool PV16>
__device__ __forceinline__ void flash_fwd_impl(const __half* __restrict__ q, const __half* __restrict__ k,
                                               const __half* __restrict__ v, __half* __restrict__ o, int N, int H,
                                               long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh,
                                               long long vsb, int vsn, int vsh, long long osb, int osn, int osh,
                                               float scale_log2) {
    constexpr int BR = 64 * RM, BC = 64;
    __shared__ __align__(128) __half sQ[BR * 64];
    __shared__ __align__(128) __half sK[2][BC * 64];
    __shared__ __align__(128) __half sV[2][BC * 64];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int bh = blockIdx.y, b = bh / H, h = bh - b * H;
    const int q0 = blockIdx.x * BR;
    const __half* qb = q + b * qsb + (long long)h * qsh;
    const __half* kb = k + b * ksb + (long long)h * ksh;
    const __half* vb = v + b * vsb + (long long)h * vsh;

    // Q tile -> smem
#pragma unroll
    for (int i = 0; i < BR * 8 / 128; ++i) {
        int c = tid + i * 128, row = c >> 3, ch = c & 7;
        int n = q0 + row;
        cp_async16(smem_u32(&sQ[swz(row, ch)]), qb + (long long)min(n, N - 1) * qsn + ch * 8, n < N);
    }
    auto load_kv = [&](int buf, int k0) {
#pragma unroll
        for (int i = 0; i < BC * 8 / 128; ++i) {
            int c = tid + i * 128, row = c >> 3, ch = c & 7;
            int n = k0 + row;
            bool ok = n < N;
            int nn = min(n, N - 1);
            cp_async16(smem_u32(&sK[buf][swz(row, ch)]), kb + (long long)nn * ksn + ch * 8, ok);
            cp_async16(smem_u32(&sV[buf][swz(row, ch)]), vb + (long long)nn * vsn + ch * 8, ok);
        }
    };
    load_kv(0, 0);
    cp_async_commit();

    // running state: rows (lane>>2) and (lane>>2)+8 of each of the RM m-tiles of this warp
    float m_r[RM][2], l_r[RM][2];
    float oacc[RM][8][4];
#pragma unroll
    for (int mi = 0; mi < RM; ++mi) {
        m_r[mi][0] = m_r[mi][1] = -INFINITY;
        l_r[mi][0] = l_r[mi][1] = 0.f;
#pragma unroll
        for (int u = 0; u < 8; ++u) oacc[mi][u][0] = oacc[mi][u][1] = oacc[mi][u][2] = oacc[mi][u][3] = 0.f;
    }
    uint32_t qf[RM][4][4];  // Q A-fragments: [m-tile][k-step of 16 dims][4]

    const int nblk = (N + BC - 1) / BC;
    for (int j = 0; j < nblk; ++j) {
        cp_async_wait<0>();
        __syncthreads();
        if (j == 0) {
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    int row = warp * 16 * RM + mi * 16 + (lane & 15);
                    int ch = kk * 2 + (lane >> 4);
                    ldsm_x4(smem_u32(&sQ[swz(row, ch)]), qf[mi][kk][0], qf[mi][kk][1], qf[mi][kk][2], qf[mi][kk][3]);
                }
        }
        if (j + 1 < nblk) load_kv((j + 1) & 1, (j + 1) * BC);
        cp_async_commit();
        const __half* K_ = sK[j & 1];
        const __half* V_ = sV[j & 1];

        // ---- S = Q K^T for 64 keys: 8 n-tiles of 8 keys
        float s[RM][8][4];
        if (QK16) {
            uint32_t sh[RM][8][2];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) sh[mi][t][0] = sh[mi][t][1] = 0u;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int t = 0; t < 8; t += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = (t + (lane >> 4)) * 8 + (lane & 7);
                    int ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm_x4(smem_u32(&K_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f16(sh[mi][t], qf[mi][kk], b0, b1);
                        mma_f16(sh[mi][t + 1], qf[mi][kk], b2, b3);
                    }
                }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    float2 a = unpack_h2(sh[mi][t][0]), c = unpack_h2(sh[mi][t][1]);
                    s[mi][t][0] = a.x; s[mi][t][1] = a.y; s[mi][t][2] = c.x; s[mi][t][3] = c.y;
                }
        } else {
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) s[mi][t][0] = s[mi][t][1] = s[mi][t][2] = s[mi][t][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int t = 0; t < 8; t += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = (t + (lane >> 4)) * 8 + (lane & 7);
                    int ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm_x4(smem_u32(&K_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f32(s[mi][t], qf[mi][kk], b0, b1);
                        mma_f32(s[mi][t + 1], qf[mi][kk], b2, b3);
                    }
                }
        }
        // ---- tail mask (last block only)
        if ((j + 1) * BC > N) {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                int key = j * BC + t * 8 + (lane & 3) * 2;
#pragma unroll
                for (int mi = 0; mi < RM; ++mi) {
                    if (key >= N) s[mi][t][0] = s[mi][t][2] = -INFINITY;
                    if (key + 1 >= N) s[mi][t][1] = s[mi][t][3] = -INFINITY;
                }
            }
        }
        // ---- online softmax (base 2), P packed as fp16 A-fragments
        uint32_t pf[RM][4][4];  // [m-tile][k-chunk of 16 keys][4]
#pragma unroll
        for (int mi = 0; mi < RM; ++mi) {
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                mx0 = fmaxf(mx0, fmaxf(s[mi][t][0], s[mi][t][1]));
                mx1 = fmaxf(mx1, fmaxf(s[mi][t][2], s[mi][t][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
            float mn0 = fmaxf(m_r[mi][0], mx0 * scale_log2), mn1 = fmaxf(m_r[mi][1], mx1 * scale_log2);
            float c0 = fast_exp2(m_r[mi][0] - mn0), c1 = fast_exp2(m_r[mi][1] - mn1);
            m_r[mi][0] = mn0;
            m_r[mi][1] = mn1;
            float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                float p0 = fast_exp2(fmaf(s[mi][t][0], scale_log2, -mn0));
                float p1 = fast_exp2(fmaf(s[mi][t][1], scale_log2, -mn0));
                float p2 = fast_exp2(fmaf(s[mi][t][2], scale_log2, -mn1));
                float p3 = fast_exp2(fmaf(s[mi][t][3], scale_log2, -mn1));
                sum0 += p0 + p1;
                sum1 += p2 + p3;
                // n-tile t -> k-chunk t/2: even t = a0 (row) / a1 (row+8), odd t = a2 / a3
                pf[mi][t >> 1][(t & 1) * 2 + 0] = pack_h2(p0, p1);
                pf[mi][t >> 1][(t & 1) * 2 + 1] = pack_h2(p2, p3);
            }
            l_r[mi][0] = l_r[mi][0] * c0 + sum0;
            l_r[mi][1] = l_r[mi][1] * c1 + sum1;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                oacc[mi][u][0] *= c0; oacc[mi][u][1] *= c0;
                oacc[mi][u][2] *= c1; oacc[mi][u][3] *= c1;
            }
        }
        // ---- O += P V
        if (PV16) {
            uint32_t ot[RM][8][2];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int u = 0; u < 8; ++u) ot[mi][u][0] = ot[mi][u][1] = 0u;
#pragma unroll
            for (int kc = 0; kc < 4; ++kc)
#pragma unroll
                for (int u = 0; u < 8; u += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int ch = u + (lane >> 4);
                    ldsm_x4_t(smem_u32(&V_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f16(ot[mi][u], pf[mi][kc], b0, b1);
                        mma_f16(ot[mi][u + 1], pf[mi][kc], b2, b3);
                    }
                }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float2 a = unpack_h2(ot[mi][u][0]), c = unpack_h2(ot[mi][u][1]);
                    oacc[mi][u][0] += a.x; oacc[mi][u][1] += a.y; oacc[mi][u][2] += c.x; oacc[mi][u][3] += c.y;
                }
        } else {
#pragma unroll
            for (int kc = 0; kc < 4; ++kc)
#pragma unroll
                for (int u = 0; u < 8; u += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int ch = u + (lane >> 4);
                    ldsm_x4_t(smem_u32(&V_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f32(oacc[mi][u], pf[mi][kc], b0, b1);
                        mma_f32(oacc[mi][u + 1], pf[mi][kc], b2, b3);
                    }
                }
        }
    }
    // ---- epilogue: O / l -> fp16 (staged through smem for 16-byte stores)
    __syncthreads();
#pragma unroll
    for (int mi = 0; mi < RM; ++mi) {
        float l0 = l_r[mi][0], l1 = l_r[mi][1];  // per-thread partial row sums -> quad total
        l0 += __shfl_xor_sync(0xffffffff, l0, 1);
        l0 += __shfl_xor_sync(0xffffffff, l0, 2);
        l1 += __shfl_xor_sync(0xffffffff, l1, 1);
        l1 += __shfl_xor_sync(0xffffffff, l1, 2);
        float i0 = 1.f / l0, i1 = 1.f / l1;
        int r0 = warp * 16 * RM + mi * 16 + (lane >> 2);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            int col = u * 8 + (lane & 3) * 2;
            *reinterpret_cast<uint32_t*>(&sQ[swz(r0, col >> 3) + (col & 7)]) = pack_h2(oacc[mi][u][0] * i0, oacc[mi][u][1] * i0);
            *reinterpret_cast<uint32_t*>(&sQ[swz(r0 + 8, col >> 3) + (col & 7)]) = pack_h2(oacc[mi][u][2] * i1, oacc[mi][u][3] * i1);
        }
    }
    __syncthreads();
    __half* ob = o + b * osb + (long long)h * osh;
#pragma unroll
    for (int i = 0; i < BR * 8 / 128; ++i) {
        int c = tid + i * 128, row = c >> 3, ch = c & 7;
        int n = q0 + row;
        if (n < N) *reinterpret_cast<uint4*>(ob + (long long)n * osn + ch * 8) = *reinterpret_cast<const uint4*>(&sQ[swz(row, ch)]);
    }
}

#define FLASH_ARGS                                                                                                      \
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o,  \
        int N, int H, long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh, long long vsb, int vsn, int vsh, \
        long long osb, int osn, int osh, float scale_log2
#define FLASH_CALL q, k, v, o, N, H, qsb, qsn, qsh, ksb, ksn, ksh, vsb, vsn, vsh, osb, osn, osh, scale_log2

extern "C" __global__ void __launch_bounds__(128) flash64_r2_hh(FLASH_ARGS) { flash_fwd_impl<2, true, true>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r2_fh(FLASH_ARGS) { flash_fwd_impl<2, false, true>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r2_ff(FLASH_ARGS) { flash_fwd_impl<2, false, false>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_hh(FLASH_ARGS) { flash_fwd_impl<1, true, true>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_fh(FLASH_ARGS) { flash_fwd_impl<1, false, true>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_ff(FLASH_ARGS) { flash_fwd_impl<1, false, false>(FLASH_CALL); }

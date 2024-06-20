#include "sgemm_lora.cuh"
#include "exllama/exllama_liteqwen_ext.h"

// ================ inline functions ===============
__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__ void lgd32to32(float &reg, const void *ptr, bool guard) {
    if (guard) {
        reg = ((float*)ptr)[0];
    }
}

__device__ __forceinline__ void lgd16to32(float &reg, const void *ptr, bool guard) {
    if (guard) {
        reg = __half2float(((__half*)ptr)[0]);
    }
}

__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void stg32to16(const float &reg, void *ptr, bool guard) {
    if (guard) {
        ((__half*)ptr)[0] = __float2half(reg);
    }
}

__device__ __forceinline__
void stg32to16_inplace_add(const float &reg, void *ptr, float orig_val, float lora_scale, bool guard) {
    if (guard) {
        // printf("orig_fp32=%f\n", orig_val);
        float new_val = orig_val + lora_scale * reg;
        ((__half*)ptr)[0] = __float2half(new_val);
    }
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}


__device__ __noinline__
void C_tile_wb_2fp16(StgFrag C_frag,
               __half *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32to16(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n); // direct output to fp16 tensors
    }
}

__device__ __noinline__
void C_tile_wb_2fp16_lora_fuse(StgFrag C_frag,
               __half *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx,
               float lora_scale) {
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        // stg32to16(C_lds_ptr[i * 32],
        //       C_stg_ptr + i * n,
        //       i < m_guard && n_idx < n); // direct output to fp16 tensors
        
        float orig_val;
        if (i < m_guard && n_idx < n) {
            orig_val = __half2float(C_stg_ptr[i*n]);
        } else {
            orig_val = 0.0f;
        }
        stg32to16_inplace_add(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              orig_val,
              lora_scale,
              i < m_guard && n_idx < n); // adding result to an existing fp16 tensor (lora_B add)
    }
}


// ========== fp32 inp, weight & out linear kernel, x=inp=A, W=weight transposed=B. ================

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n128k8
 * warp tile: m32n64k8
 * thread tile: m8n8k8
 * thread fragment:
 *     matrixA: 8x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                128
 *                    --|---------------------|
 *             B_tile  8|                     |
 *                    --|---------------------|
 *
 *  A_tile   | 8 |      |    64    |
 *         --|---|    --|----------|----------|
 *           |   |    32|  warp_0  |  warp_1  |
 *           |   |    --|----------|----------|
 *           |   |      |  warp_2  |  warp_3  |
 *        128|   |      |----------|----------|
 *           |   |      |  warp_4  |  warp_5  |
 *           |   |      |----------|----------|
 *           |   |      |  warp_6  |  warp_7  |
 *         --|---|      |----------|----------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */
__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel(const float *A,
                            const float *B,
                            float *C,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k,
                            uint32_t A_ldg_step,    // k * sizeof(float)
                            uint32_t B_ldg_step    // n * sizeof(float) * 8
    ) {  
    /*
     * matrix A & B thread block tile shared memory (double buffer)
     * matrix A: 132 * 8 * 4Byte/item * double buffer = 4.125KB * 2
     * matrix B: 128 * 8 * 4Byte/item * double buffer = 8KB
     *
     * for double buffer faster switch, A_smem requires 8KB * 2 shared memory
     * and 16KB aligned, B_smem should be 8KB aligned, then the double buffer
     * can be switched by only 1 xor instruction:
     *     (uint32_t &)A_smem ^= 0x2000;
     *     (uint32_t &)B_smem ^= 0x1000;
     */
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);

    // A, B and C register fragment
    float A_frag[2][8];
    float B_frag[2][8];
    float C_frag[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C_frag[i][j] = 0;
        }
    }

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    
    // 4x8 threads each warp for FFMA
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // A_tile & B_tile ldg pointer
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8);
    // const char *B_ldg_ptr = (const char *)(
    //     B + (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32); // *** not B transposed version
    const char *B_ldg_ptr = (const char *)(
        B + (blockIdx.x * 128 + threadIdx.x % 32) * k + threadIdx.x / 32);

    // A_tile & B_tile sts/lds pointer
    // using uint32_t pointer for faster double buffer switch
    uint32_t A_sts_addr = smem_u32addr(
        A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));

    uint32_t A_lds_addr = smem_u32addr(
        A_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + (warp_id % 2) * 64 + mma_tid_x * 4);

    // ldg_guard to avoid LDG out of bound
    uint32_t A_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }

    uint32_t B_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int n_idx = blockIdx.x * 128 + threadIdx.x % 32 + i * 32;
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    float A_ldg_reg[4];
    float B_ldg_reg[4];

    // 1'st A&B tile loaded before the k_tile loop
    uint32_t k_tiles = (k + 7) / 8 - 1;

    // load 1'st tile to shared memory
    {
        uint32_t first_k_tile = k - k_tiles * 8;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            ldg32_nc_0(A_ldg_reg[i],
                       A_ldg_ptr + i * A_ldg_step,
                       guard); // *** fp32 version
        }

        sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
               A_sts_addr);

        #pragma unroll
        // for (int i = 0; i < 4; ++i) {
        //     bool guard = (B_ldg_guard & (1u << i)) != 0 &&
        //                  threadIdx.x / 32 < first_k_tile;
        //     ldg32_nc_0(B_ldg_reg[i],
        //                B_ldg_ptr + i * 32 * sizeof(float),
        //                guard);
        // } // *** not B transposed version
        for (int i = 0; i < 4; ++i) {
            bool guard = (B_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / 32 < first_k_tile;
            ldg32_nc_0(B_ldg_reg[i],
                       B_ldg_ptr + i * 32 * A_ldg_step,
                       guard);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
        }

        __syncthreads();

        // switch double buffer
        A_sts_addr ^= 0x2000;
        B_sts_addr ^= 0x1000;

        // ldg pointer for next tile
        A_ldg_ptr += first_k_tile * sizeof(float); //*** fp32 version
        // B_ldg_ptr += n * first_k_tile * sizeof(float); // *** not B transposed version
        B_ldg_ptr += first_k_tile * sizeof(float);
    }

    // load 1'st fragment
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 16 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 32 * sizeof(float));

    // k_tiles loop
    for (; k_tiles > 0; --k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            if (k_frag == 7) {
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                       A_sts_addr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                }

                __syncthreads();

                // switch double buffer
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x1000;
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x1000;

                // ldg pointer for next tile
                A_ldg_ptr += 8 * sizeof(float); // *** fp32 version
                // B_ldg_ptr += B_ldg_step; // *** not B transposed version
                B_ldg_ptr += 8 * sizeof(float);
            }

            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));

            // load next A&B tile
            if (k_frag == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(A_ldg_reg[i],
                             A_ldg_ptr + i * A_ldg_step,
                             (A_ldg_guard & (1u << i)) != 0); // *** fp32 version
                }

                #pragma unroll
                // for (int i = 0; i < 4; ++i) {
                //     ldg32_nc(B_ldg_reg[i],
                //              B_ldg_ptr + i * 32 * sizeof(float),
                //              (B_ldg_guard & (1u << i)) != 0);
                // } // *** not B transposed version
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * 32 * A_ldg_step,
                             (B_ldg_guard & (1u << i)) != 0);
                }
            }

            // FFMA loop
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
        }

        // FFMA loop
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C_tile write back, reuse A&B tile shared memory buffer
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

    float *C_stg_ptr = C + m_idx * n + n_idx; // *** fp32 version

    if (m_idx >= m) {
        return;
    } else if (m_idx + 32 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 8 * sizeof(float4));
                }

                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 16; ++p) {
                    stg32(C_lds_ptr[p * 32],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32); // fp32 out version
            }
        }
    }
}


// ========= lora A version, fp16 input A, fp32 loraA_W, fp32 out ===============
__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel_fp16x(const __half *A,
                            const float *B,
                            float *C,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k,
                            uint32_t A_ldg_step,    // k * sizeof(float)
                            uint32_t B_ldg_step    // n * sizeof(float) * 8
    ) {  
    /*
     * matrix A & B thread block tile shared memory (double buffer)
     * matrix A: 132 * 8 * 4Byte/item * double buffer = 4.125KB * 2
     * matrix B: 128 * 8 * 4Byte/item * double buffer = 8KB
     *
     * for double buffer faster switch, A_smem requires 8KB * 2 shared memory
     * and 16KB aligned, B_smem should be 8KB aligned, then the double buffer
     * can be switched by only 1 xor instruction:
     *     (uint32_t &)A_smem ^= 0x2000;
     *     (uint32_t &)B_smem ^= 0x1000;
     */
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);

    // A, B and C register fragment
    float A_frag[2][8];
    float B_frag[2][8];
    float C_frag[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C_frag[i][j] = 0;
        }
    }

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    uint32_t hA_ldg_step = A_ldg_step / 2;  // k * sizeof(half)

    // 4x8 threads each warp for FFMA
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // A_tile & B_tile ldg pointer
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8);
    // const char *B_ldg_ptr = (const char *)(
    //     B + (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32); // *** not B transposed version
    const char *B_ldg_ptr = (const char *)(
        B + (blockIdx.x * 128 + threadIdx.x % 32) * k + threadIdx.x / 32);

    // A_tile & B_tile sts/lds pointer
    // using uint32_t pointer for faster double buffer switch
    uint32_t A_sts_addr = smem_u32addr(
        A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));

    uint32_t A_lds_addr = smem_u32addr(
        A_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + (warp_id % 2) * 64 + mma_tid_x * 4);

    // ldg_guard to avoid LDG out of bound
    uint32_t A_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }

    uint32_t B_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int n_idx = blockIdx.x * 128 + threadIdx.x % 32 + i * 32;
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    float A_ldg_reg[4];
    float B_ldg_reg[4];

    // 1'st A&B tile loaded before the k_tile loop
    uint32_t k_tiles = (k + 7) / 8 - 1;

    // load 1'st tile to shared memory
    {
        uint32_t first_k_tile = k - k_tiles * 8;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            // ldg32_nc_0(A_ldg_reg[i],
            //            A_ldg_ptr + i * A_ldg_step,
            //            guard); // *** fp32 version
            lgd16to32(A_ldg_reg[i], A_ldg_ptr + i * hA_ldg_step, guard);
        }

        sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
               A_sts_addr);

        #pragma unroll
        // for (int i = 0; i < 4; ++i) {
        //     bool guard = (B_ldg_guard & (1u << i)) != 0 &&
        //                  threadIdx.x / 32 < first_k_tile;
        //     ldg32_nc_0(B_ldg_reg[i],
        //                B_ldg_ptr + i * 32 * sizeof(float),
        //                guard);
        // } // *** not B transposed version
        for (int i = 0; i < 4; ++i) {
            bool guard = (B_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / 32 < first_k_tile;
            ldg32_nc_0(B_ldg_reg[i],
                       B_ldg_ptr + i * 32 * A_ldg_step,
                       guard);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
        }

        __syncthreads();

        // switch double buffer
        A_sts_addr ^= 0x2000;
        B_sts_addr ^= 0x1000;

        // ldg pointer for next tile
        // A_ldg_ptr += first_k_tile * sizeof(float); //*** fp32 version
        A_ldg_ptr += first_k_tile * sizeof(__half);
        // B_ldg_ptr += n * first_k_tile * sizeof(float); // *** not B transposed version
        B_ldg_ptr += first_k_tile * sizeof(float);
    }

    // load 1'st fragment
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 16 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 32 * sizeof(float));

    // k_tiles loop
    for (; k_tiles > 0; --k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            if (k_frag == 7) {
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                       A_sts_addr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                }

                __syncthreads();

                // switch double buffer
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x1000;
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x1000;

                // ldg pointer for next tile
                // A_ldg_ptr += 8 * sizeof(float); // *** fp32 version
                A_ldg_ptr += 8 * sizeof(__half);
                // B_ldg_ptr += B_ldg_step; // *** not B transposed version
                B_ldg_ptr += 8 * sizeof(float);
            }

            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));

            // load next A&B tile
            if (k_frag == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    // ldg32_nc(A_ldg_reg[i],
                    //          A_ldg_ptr + i * A_ldg_step,
                    //          (A_ldg_guard & (1u << i)) != 0); // *** fp32 version
                    lgd16to32(A_ldg_reg[i], A_ldg_ptr + i * hA_ldg_step, (A_ldg_guard & (1u << i)) != 0);
                }

                #pragma unroll
                // for (int i = 0; i < 4; ++i) {
                //     ldg32_nc(B_ldg_reg[i],
                //              B_ldg_ptr + i * 32 * sizeof(float),
                //              (B_ldg_guard & (1u << i)) != 0);
                // } // *** not B transposed version
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * 32 * A_ldg_step,
                             (B_ldg_guard & (1u << i)) != 0);
                }
            }

            // FFMA loop
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
        }

        // FFMA loop
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C_tile write back, reuse A&B tile shared memory buffer
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

    float *C_stg_ptr = C + m_idx * n + n_idx;

    if (m_idx >= m) {
        return;
    } else if (m_idx + 32 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 8 * sizeof(float4));
                }

                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 16; ++p) {
                    stg32(C_lds_ptr[p * 32],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32);
            }
        }
    }
}

// ============== loraB version, fp32 inp & fp32 loraB_W, fp16 out C=C+x@WT ==========
__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel_fp16o_add(const float *A,
                            const float *B,
                            __half *C,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k,
                            uint32_t A_ldg_step,    // k * sizeof(float)
                            uint32_t B_ldg_step,    // n * sizeof(float) * 8
                            float lora_scaling
    ) {  
    /*
     * matrix A & B thread block tile shared memory (double buffer)
     * matrix A: 132 * 8 * 4Byte/item * double buffer = 4.125KB * 2
     * matrix B: 128 * 8 * 4Byte/item * double buffer = 8KB
     *
     * for double buffer faster switch, A_smem requires 8KB * 2 shared memory
     * and 16KB aligned, B_smem should be 8KB aligned, then the double buffer
     * can be switched by only 1 xor instruction:
     *     (uint32_t &)A_smem ^= 0x2000;
     *     (uint32_t &)B_smem ^= 0x1000;
     */
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 16 * 1024);

    // A, B and C register fragment
    float A_frag[2][8];
    float B_frag[2][8];
    float C_frag[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            C_frag[i][j] = 0;
        }
    }

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;
    
    // 4x8 threads each warp for FFMA
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // A_tile & B_tile ldg pointer
    const char *A_ldg_ptr = (const char *)(
        A + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8);
    // const char *B_ldg_ptr = (const char *)(
    //     B + (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32); // *** not B transposed version
    const char *B_ldg_ptr = (const char *)(
        B + (blockIdx.x * 128 + threadIdx.x % 32) * k + threadIdx.x / 32);

    // A_tile & B_tile sts/lds pointer
    // using uint32_t pointer for faster double buffer switch
    uint32_t A_sts_addr = smem_u32addr(
        A_smem + (threadIdx.x % 8) * 132 + (threadIdx.x / 8) * 4);
    uint32_t B_sts_addr = smem_u32addr(
        B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));

    uint32_t A_lds_addr = smem_u32addr(
        A_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(
        B_smem + (warp_id % 2) * 64 + mma_tid_x * 4);

    // ldg_guard to avoid LDG out of bound
    uint32_t A_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }

    uint32_t B_ldg_guard = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int n_idx = blockIdx.x * 128 + threadIdx.x % 32 + i * 32;
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    float A_ldg_reg[4];
    float B_ldg_reg[4];

    // 1'st A&B tile loaded before the k_tile loop
    uint32_t k_tiles = (k + 7) / 8 - 1;

    // load 1'st tile to shared memory
    {
        uint32_t first_k_tile = k - k_tiles * 8;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            ldg32_nc_0(A_ldg_reg[i],
                       A_ldg_ptr + i * A_ldg_step,
                       guard); // *** fp32 version
        }

        sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
               A_sts_addr);

        #pragma unroll
        // for (int i = 0; i < 4; ++i) {
        //     bool guard = (B_ldg_guard & (1u << i)) != 0 &&
        //                  threadIdx.x / 32 < first_k_tile;
        //     ldg32_nc_0(B_ldg_reg[i],
        //                B_ldg_ptr + i * 32 * sizeof(float),
        //                guard);
        // } // *** not B transposed version
        for (int i = 0; i < 4; ++i) {
            bool guard = (B_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / 32 < first_k_tile;
            ldg32_nc_0(B_ldg_reg[i],
                       B_ldg_ptr + i * 32 * A_ldg_step,
                       guard);
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
        }

        __syncthreads();

        // switch double buffer
        A_sts_addr ^= 0x2000;
        B_sts_addr ^= 0x1000;

        // ldg pointer for next tile
        A_ldg_ptr += first_k_tile * sizeof(float); //*** fp32 version
        // B_ldg_ptr += n * first_k_tile * sizeof(float); // *** not B transposed version
        B_ldg_ptr += first_k_tile * sizeof(float);
    }

    // load 1'st fragment
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3],
           A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7],
           A_lds_addr + 16 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3],
           B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7],
           B_lds_addr + 32 * sizeof(float));

    // k_tiles loop
    for (; k_tiles > 0; --k_tiles) {
        #pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            if (k_frag == 7) {
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                       A_sts_addr);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                }

                __syncthreads();

                // switch double buffer
                A_lds_addr ^= 0x2000;
                B_lds_addr ^= 0x1000;
                A_sts_addr ^= 0x2000;
                B_sts_addr ^= 0x1000;

                // ldg pointer for next tile
                A_ldg_ptr += 8 * sizeof(float); // *** fp32 version
                // B_ldg_ptr += B_ldg_step; // *** not B transposed version
                B_ldg_ptr += 8 * sizeof(float);
            }

            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));

            // load next A&B tile
            if (k_frag == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(A_ldg_reg[i],
                             A_ldg_ptr + i * A_ldg_step,
                             (A_ldg_guard & (1u << i)) != 0); // *** fp32 version
                }

                #pragma unroll
                // for (int i = 0; i < 4; ++i) {
                //     ldg32_nc(B_ldg_reg[i],
                //              B_ldg_ptr + i * 32 * sizeof(float),
                //              (B_ldg_guard & (1u << i)) != 0);
                // } // *** not B transposed version
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * 32 * A_ldg_step,
                             (B_ldg_guard & (1u << i)) != 0);
                }
            }

            // FFMA loop
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
    #pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 132 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 132 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
        }

        // FFMA loop
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C_tile write back, reuse A&B tile shared memory buffer
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

    // float *C_stg_ptr = C + m_idx * n + n_idx; // *** fp32 version
    __half *C_stg_ptr = C + m_idx * n + n_idx; // *** fp16 inplace add version

    if (m_idx >= m) {
        return;
    } else if (m_idx + 32 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 8 * sizeof(float4));
                }

                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 16; ++p) {
                    // stg32to16(C_lds_ptr[p * 32],
                    //       C_stg_ptr + (i * 16 + p) * n + j * 32,
                    //       j * 32 < n_guard); // direct output to fp16 tensor
                    
                    
                    float orig_val;
                    if (i*16+p < m && j*32 < n) {
                        orig_val = __half2float(C_stg_ptr[(i * 16 + p) * n + j * 32]);
                    } else {
                        orig_val = 0.0f;
                    }
                    stg32to16_inplace_add(C_lds_ptr[p * 32],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          orig_val,
                          lora_scaling,
                          j * 32 < n_guard); // adding result to existing fp16 tensor (lora_B add)
                }
            }
        }
    } else {        
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                // C_tile_wb(stg_frag,
                //           C_stg_ptr + i * 16 * n + j * 32,
                //           C_lds_ptr,
                //           C_sts_addr,
                //           m,
                //           n,
                //           m_idx + i * 16,
                //           n_idx + j * 32); // fp32 out version

                C_tile_wb_2fp16_lora_fuse(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32,
                          lora_scaling);
            }
        }
    }
}

// =============== small batch, simple reduce gemm =========
template <int THREAD_PER_BLOCK, int gridx_bound>
__global__ void GemvFp16Fp32Kernel2NoBias(__half *A, float *B, float *C, int m, int n, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    int tid = threadIdx.x;
    size_t p = blockIdx.y;
    size_t bm_thread = blockIdx.x;

    for (int m_start=0; m_start< m; m_start+=gridx_bound) {
        size_t bm = m_start+bm_thread;
        // 1. 计算
        sdata[tid] = 0;
        if (bm < m) {
            for (int i = tid; i < k; i += THREAD_PER_BLOCK) {
                sdata[tid] += __half2float(A[bm * k + i]) * B[p * k + i];
            }
        }
        __syncthreads();
        
        for (int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s && bm < m) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0 && bm<m) {
            C[bm * n + p] = sdata[0];
        }
        __syncthreads();
    }
 
}



// ========== other kernels ======
template <int block_size>
__global__ void tileAddKernel2(__half* outs, __half* inp, int repeat_num, int stride) {
    __shared__ __half sdata[block_size];
    int tid = blockIdx.x*block_size + threadIdx.x;
    if (tid < stride) {
        sdata[threadIdx.x] = inp[tid];
    }
    __syncthreads();

    int sdata_tile_rounds = (int)(ceil((float)(repeat_num) / block_size));
    int max_pos = repeat_num * stride;
    for (int r=0; r<sdata_tile_rounds; r++) {
        int row_id = r * block_size + threadIdx.x;
        if (row_id < repeat_num) {
            int writing_offset = row_id * stride + blockIdx.x*block_size;
            for (int j=0; j<block_size;j++) {
                int writing_pos = writing_offset+j;
                if (blockIdx.x*block_size+j < stride && writing_pos < max_pos) {
                    outs[writing_pos] += sdata[j];
                }
            }
        }
    }
}

void quant4_lora_linear_fused(const liteqwen::Data& out_tensor, const liteqwen::Data& x, uintptr_t w_ref, const liteqwen::Data& bias, bool use_bias, const liteqwen::Data& loraA_out, const liteqwen::Data& loraA_W, const liteqwen::Data& loraB_W, int r, float lora_scaling) {
    
    int dim_num = out_tensor.shape.size();
    int n;
    if (dim_num == 4) {
        n = out_tensor.shape[dim_num-2] * out_tensor.shape[dim_num-1];
    } else {
        n = out_tensor.shape[dim_num-1];
    }
    int m = (int)(out_tensor.numel()) / n;
    int inp_dim_num = x.shape.size();
    int k;
    if (inp_dim_num == 4) {
        k =  x.shape[inp_dim_num-2] * x.shape[inp_dim_num-1];
    } else {
        k = x.shape[inp_dim_num-1];
    }
    cublasHandle_t handle = get_cublas_handler(x.gpu_id);
    __half* out_data = (__half*)(out_tensor.cudaData);

    float* loraA_W_data = (float*)(loraA_W.cudaData);
    float* loraB_W_data = (float*)(loraB_W.cudaData);
    float* loraA_out_data = (float*)(loraA_out.cudaData);

    // gptq linear x @ WT + tiled(b)
    q4_matmul(x, w_ref, out_tensor, handle);
    if (use_bias) {
        dim3 dimBlock(32);
        dim3 dimGrid((int)(ceil((float)(n) / 32)));
        __half* bias_data = (__half*)(bias.cudaData);
        tileAddKernel2<32><<<dimGrid, dimBlock>>>(out_data, bias_data, m, n); //使用bias repeat进行初始化。
    }

    // loraA_out = fp16x @ loraB_wT, hidden=k
    __half* x_data = (__half*)x.cudaData;
    if (m < 128 && r < 128) {
        // loraA使用sgemm基本只有1个block在运算，并行度不够，所以使用二分reduce方法算matmul
        dim3 dimBlockLoraA(256);
        if (m>256) {
            dim3 dimGridLoraA(256, r);
            GemvFp16Fp32Kernel2NoBias<256, 256><<<dimGridLoraA, dimBlockLoraA>>>(x_data, loraA_W_data, loraA_out_data, m, r, k);
        } else {
            dim3 dimGridLoraA(m, r);
            GemvFp16Fp32Kernel2NoBias<256, 256><<<dimGridLoraA, dimBlockLoraA>>>(x_data, loraA_W_data, loraA_out_data, m, r, k);
        }
    } else {
        dim3 grid_loraA((r + 127) / 128, (m + 127) / 128);
        sgemm_128x128x8_kernel_fp16x<<<grid_loraA, 256>>>(x_data, loraA_W_data, loraA_out_data, m, r, k, k*sizeof(float), r*sizeof(float)*8);
    }

    // out += (loraA_out @ loraB_wT), hidden=r
    dim3 grid_loraB((n + 127) / 128, (m + 127) / 128);
    sgemm_128x128x8_kernel_fp16o_add<<<grid_loraB, 256>>>(loraA_out_data, loraB_W_data, out_data, m, n, r, r*sizeof(float), n*sizeof(float)*8, lora_scaling);
}
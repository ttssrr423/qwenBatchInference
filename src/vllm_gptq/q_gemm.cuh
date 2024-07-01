#include "core_cpu.h"
#include <cuda_fp16.h>
#include "core_gpu.cuh"

void gptq_gemm(const liteqwen::Data& c, const liteqwen::Data& a, uint32_t* b_q_weight,
                        uint32_t* b_gptq_qzeros,
                        half* b_gptq_scales, uint32_t* b_g_idx,
                        bool use_exllama, int64_t bit, int groups, cublasHandle_t handle);

void gptq_shuffle(const liteqwen::Data& q_weight, const liteqwen::Data& q_perm, bool desc_act, int64_t bit);
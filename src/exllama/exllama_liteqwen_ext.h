#include <cublas_v2.h>
#include "util.cuh"
#include "tuning.h"
#include "core_gpu.cuh"
#include "core_cpu.h"
#include "cuda_func/q4_matrix.cuh"
#include "cuda_func/q4_matmul.cuh"
#include "cuda_func/column_remap.cuh"
#include "cuda_buffers.cuh"

void prepare_buffers
(
    int device_index,
    const liteqwen::Data& temp_state,
    const liteqwen::Data& temp_dq
);

uintptr_t make_q4
(
    const liteqwen::Data& qweight,
    const liteqwen::Data& qzeros,
    const liteqwen::Data& scales,
    const liteqwen::Data& g_idx,
    int device
);


// q4_matmul的推理速度落后vllm_gptq，所以弃用exllama的q4_matmul,只保留make_q4和prepare_buffers。
// q4_matmul改用vllm_gptq/q_gemm.cuh下的gptq_gemm方法。
void q4_matmul
(
    const liteqwen::Data& x,
    uintptr_t w,
    const liteqwen::Data& out,
    cublasHandle_t handle
);

__half* get_temp_dq(int gpu_id);
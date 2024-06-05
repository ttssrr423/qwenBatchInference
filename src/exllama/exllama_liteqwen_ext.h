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

void q4_matmul
(
    const liteqwen::Data& x,
    uintptr_t w,
    const liteqwen::Data& out,
    cublasHandle_t handle
);
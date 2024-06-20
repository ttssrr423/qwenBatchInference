#include <cuda.h>
#include <cuda_fp16.h>
#include "core_gpu.cuh"
#include <cuda_runtime.h>

void quant4_lora_linear_fused(const liteqwen::Data& out_tensor, const liteqwen::Data& x, uintptr_t w_ref, const liteqwen::Data& bias, bool use_bias, const liteqwen::Data& loraA_out, const liteqwen::Data& loraA_W, const liteqwen::Data& loraB_W, int r, float lora_scaling);
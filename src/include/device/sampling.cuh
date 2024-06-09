#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "core_gpu.cuh"
#include "entities.h"

void gpu_curand_init(int gpu_id, int world_size, int data_size, int handle_id, const liteqwen::Data& seeds, int* cpu_seeds);
void gpu_curand_init(int gpu_id, int world_size, int data_size, int handle_id, const liteqwen::Data& seeds);
void filterInvalidApplyTemperature(const liteqwen::Data& logitsFp32, const liteqwen::Data& logits, const liteqwen::Data& temperature_tensor, int vocab_size, int dynamic_bsz, int masking_eos_id);
void topk_sampling(const liteqwen::Data& out_id, int gpu_id, int world_size, int handle_id, const liteqwen::Data& logits, int channel, int top_k, const liteqwen::Data& top_p, const liteqwen::Data& _1_pass_result, const liteqwen::Data& _1_psss_indices, const liteqwen::Data& gpu_top_logits, const liteqwen::Data& gpu_top_indices, const liteqwen::Data& sample_softmax_out, int dynamic_bsz);
liteqwen::BatchGeneratedRes download_sampled(const liteqwen::Data& sampled_id, int* cpu_sampled_id, int* eos_ids, int eos_num, int batch_size);
void batch_download_logits(int* top_batch_idx, float* top_batch_lgts, const liteqwen::Data& gpu_top_logits, const liteqwen::Data& gpu_top_indices, int dynamic_bsz, int top_k);
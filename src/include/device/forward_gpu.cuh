#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <vector>
#include <chrono>

#include "core_gpu.cuh"
#include "kv_cache.h"

void cpu_embedding_fwd(int gpu_id, const liteqwen::Data& out_tensor, int* cpu_input_ids, const liteqwen::Data& embedding_weights, int input_offset, int lookup_len, int channel);
void apply_rotary_embeddings(const liteqwen::Data& query, const liteqwen::Data& key, int dynamic_bsz, int dynamic_l, int hidden_size, int head_num, const liteqwen::Data& cos, const liteqwen::Data& sin);
void rotary_lookup(bool is_prefill, int gpu_id, const liteqwen::Data& out_cos, const liteqwen::Data& out_sin, const liteqwen::Data& batch_bids, const liteqwen::Data& batch_starts, const liteqwen::Data& cos_gpu, const liteqwen::Data& sin_gpu, int input_offset, int lookup_len, int channel, int dynamic_bsz);
void decode_attention(const liteqwen::Data& attended_out, const liteqwen::Data& scores, liteqwen::StringArray input_request_ids, int bl_bound, int batch_maxt, int layer_id, const liteqwen::Data& query, const liteqwen::Data& bl_batch_ids, const liteqwen::Data& start_positions, int max_dynamic_bsz, liteqwen::PipelineKVPool* kv_cache_ref, int kv_heads, int channel);
void rms_norm(const liteqwen::Data& out_tensor, const liteqwen::Data& hidden_in, const liteqwen::Data& norm_w, int sequence_length, int channel);
void quant4_linear_fwd(const liteqwen::Data& out_tensor, const liteqwen::Data& x, uintptr_t w_ref, const liteqwen::Data& bias, bool use_bias);
void quant4_lora_linear_fwd(const liteqwen::Data& out_tensor, const liteqwen::Data& x, uintptr_t w_ref, const liteqwen::Data& bias, bool use_bias, const liteqwen::Data& fp32_x, const liteqwen::Data& loraA_out, const liteqwen::Data& loraB_out, const liteqwen::Data& loraA_W, const liteqwen::Data& loraB_W, int r, float lora_scaling);
void inplace_add_half(const liteqwen::Data& a, const liteqwen::Data& b, size_t boundary);
void qwen_silu(const liteqwen::Data& out_tensor, const liteqwen::Data& x, const liteqwen::Data& up, int row_num, int ffn_hidden_size);
void dynamic_slice_last(const liteqwen::Data& final_step_hidden, const liteqwen::Data& hidden_state, const liteqwen::Data& local_qstarts, int dynamic_bsz, int hidden_size);
void linear_fwd(const liteqwen::Data& out_tensor, const liteqwen::Data& x, const liteqwen::Data& W, const liteqwen::Data& b, int m, int n, int k, bool use_bias);
void dynamic_gqa_tile(const liteqwen::Data& k_proj_layer_tiled, const liteqwen::Data& v_proj_layer_tiled, const liteqwen::Data& k_proj_layer, const liteqwen::Data& v_proj_layer, int dynamic_bl, int attention_heads, int kv_heads, int channels);
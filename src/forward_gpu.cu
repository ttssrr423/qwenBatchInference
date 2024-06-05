
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <vector>
#include <chrono>

#include "forward_gpu.cuh"
#include "core_gpu.cuh"
#include "kv_cache.h"
#include "exllama/exllama_liteqwen_ext.h"

std::map<int, std::pair<int, int>> reducePartitionLookup;

// ========== KERNELS =============
__global__ void embeddingDynamicLookupKernel(__half* outs, __half* embedding_weights, uint8_t* batch_ids, int* batch_starts, bool is_prefill, int lookup_len, int channel, int bsz) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int pos = tid / channel;
    int cid = tid % channel;
    int bid;
    if (is_prefill) {
        bid = static_cast<int>(batch_ids[pos]);
    } else {
        bid = pos;
    }
    
    if (bid >= bsz) {
        if (cid == 0) {
            printf("tid=%i, pos=%i batch id looked up in batch_ids is not valid (%i >= %i), should be less than dynamic_bsz\n", tid, pos, bid, bsz);
        }
        return;
    }

    
    int batch_start_pos = batch_starts[bid];

    int step_id;
    if (is_prefill) {
        step_id = pos - batch_start_pos;
    } else {
        int batch_nest_pos = batch_starts[bid+1];
        step_id = batch_nest_pos - batch_start_pos - 1;
    }

    if (cid < channel && pos < lookup_len) {
        // printf("%i(pos=%i,cid=%i,id=%i)|", tid, pos, cid, id);
        unsigned long reading_pos = static_cast<unsigned long>(step_id) * channel + cid;
        outs[pos * channel + cid] = embedding_weights[reading_pos];
    }
}

template<int channel, int grid_bound>
__global__ void copy_kv_tile_kernel(__half* out_k_data, __half* out_v_data, __half* in_k_data, __half* in_v_data, int query_heads, int kv_heads, size_t bound) {
    // dimBlock = 128
    int tid = threadIdx.x;
    
    for (int blkid=blockIdx.x; blkid < bound; blkid+=grid_bound) {
        int bl_id = blkid / kv_heads;
        int kv_head_id = blkid % kv_heads;
        int head_num_ratio = query_heads / kv_heads;
        int q_head_bound = (kv_head_id+1) * head_num_ratio;

        size_t read_idx = static_cast<size_t>(channel) * kv_heads * bl_id + kv_head_id * channel + tid;
        size_t write_idx_base = static_cast<size_t>(channel) * query_heads * bl_id;

        __half h_k = in_k_data[read_idx];
        __half h_v = in_v_data[read_idx];

        for (int q_head_id=kv_head_id*head_num_ratio; q_head_id < q_head_bound; q_head_id++) {
            size_t write_idx = write_idx_base + q_head_id * channel + tid;
            out_k_data[write_idx] = h_k;
            out_v_data[write_idx] = h_v;
        }
    }
}


template <int channel, int grid_bound>
__global__ void batch_gqa_decode_score(int block_bound, float* scores, __half* query, void** cache_start_ptrs, int dynamic_bsz, uint8_t* bl_batch_ids, int* batch_starts, int query_heads, int kv_heads, int batch_maxt, float dim_sqrt) {
    // dimGrid(dynamic_bl * H_q), dimBlock(128)
    // channel = dimBlock = 128
    // if block_bound > grid_bound, loop process all the blocks, else only a single loop run.

    __shared__ float sdata[channel];
    __shared__ int sbatch_starts[2]; // position_start, position_end
    __shared__ __half* sbatch_kv_ptrs[2]; // &key[0] and &value[0]
    int tid = threadIdx.x;

    for (int blkid=blockIdx.x; blkid<block_bound; blkid +=grid_bound) {
        int bl_id = blkid / query_heads;
        int q_head_id = blkid % query_heads;
        int head_num_ratio = query_heads / kv_heads;
        int kv_head_id = q_head_id / head_num_ratio;
        int batch_id = static_cast<int>(bl_batch_ids[bl_id]); // bounded by dynamic_bsz

        // starts缓存加速。
        if (tid < 2) {
            sbatch_starts[tid] = batch_starts[batch_id + tid];
            sbatch_kv_ptrs[tid] = reinterpret_cast<__half*>(cache_start_ptrs[batch_id*2+tid]);
        }

        sdata[tid] = 0.0f;
        __syncthreads();

        int end_pos = sbatch_starts[1];
        int start_pos = sbatch_starts[0];
        if (bl_id >= end_pos || end_pos == 0) { // length bound by batch_starts[dynamic_bsz]
            return;
        }

        // int data_len = end_pos - start_pos;
        int step_id = bl_id - start_pos;
        // if (step_id < 0 && tid == 0) {
        //     printf("KERNEL ERROR: bl_id(%i)=(blkid(%i) / H) looking up bl_batch_ids to get batch_id=%i, getting batch_starts[batch_id]=start=%i should be less than batch_starts[batch_id+1]=end(%i), something wrong with data prepare.\n", bl_id, blkid, batch_id, start_pos, end_pos);
        // }

        __half* key_data_start = sbatch_kv_ptrs[0]; // batch_id的t=0的key-cache指针。time stride = H_kv * D
        // __half* value_data_start = sbatch_kv_ptrs[1]; // batch_id的t=0的val-cache指针。time stride = H_kv * D

        int kv_read_shift = (bl_id - start_pos) * channel * kv_heads + kv_head_id * channel + tid;
        int q_read_shift = batch_id * channel * query_heads + q_head_id * channel + tid;
        
        float prod = __half2float(key_data_start[kv_read_shift]) * __half2float(query[q_read_shift]);
        sdata[tid] = prod;
        __syncthreads();

        // if (tid == 0) {
        //     printf("bid=%i, q_head_id=%i, step_id=bl_id(%i)-start(%i)=%i(max=%i), chn=0, q_read_shift=%i, kv_read_shift=%i\n", batch_id, q_head_id, bl_id, start_pos, step_id, batch_maxt, q_read_shift, kv_read_shift);
        // }

        for (int pow=channel/2; pow>0; pow>>=1){
            if (tid<pow) {
                sdata[tid] = sdata[tid] + sdata[tid+pow];
            }
            __syncthreads();
        }
        
        if (tid==0) {
            // int out_score_shift = batch_id * query_heads * batch_maxt + q_head_id * batch_maxt + step_id; // [B, H, 1, batch_maxt]
            int out_score_shift = blkid; // [BL, H]
            float score = (sdata[0] / dim_sqrt);
            scores[out_score_shift] = score;
        }
    }
}

template<int steps_per_fold, int channel, int block_size, int block_channel_ratio>
__global__ void SoftmaxFuseDecodeAttnKernel(__half* outs, float* scores, void** cache_start_ptrs, int* batch_starts, int query_heads, int kv_heads) {
    // dimGrid(dynamic_bsz * H_q), dimBlock(block_size). steps_per_fold根据batch_maxt计算，因为最慢计算一般出现在最长的example上。
    // block_size >= steps_per_fold, block_size > channel
    // out_parallel_num = max(channel, step_per_fold), small_block=True if step_per_fold < channel
    __shared__ float sdata[steps_per_fold];
    __shared__ float out_accu[block_size];
    __shared__ float prev_o[channel];

    __shared__ float m; // 截至当前patch的max_score
    __shared__ float l; // 截至当前patch的Z累计
    __shared__ int sbatch_starts[2]; // position_start, position_end
    __shared__ __half* sbatch_kv_ptrs[2]; // &key[0] and &value[0]

    __shared__ float rescale_factor;
    __shared__ float prev_m;

    int tid = threadIdx.x;
    int batch_id = blockIdx.x / query_heads;
    int head_id = blockIdx.x % query_heads;
    int head_num_ratio = query_heads / kv_heads;
    int kv_head_id = head_id / head_num_ratio;
    
    // if (head_id != kv_head_id) {
    //     printf("warning: attn head_id should == kv_head_id, ratio=%i\n", head_num_ratio);
    // }
    unsigned long hidden_size = static_cast<unsigned long>(channel) * query_heads;
    unsigned long cache_size = static_cast<unsigned long>(channel) * kv_heads;
    int out_accu_t_offset = tid / channel;
    int out_accu_chn_id = tid % channel;

    // starts缓存加速。
    if (tid < 2) {
        sbatch_starts[tid] = batch_starts[batch_id + tid];
        sbatch_kv_ptrs[tid] = reinterpret_cast<__half*>(cache_start_ptrs[batch_id*2+tid]);
    }
    if (tid == 0) {
        m = -8192.0f;
        l = 0.0f;
        prev_m = 0.0f;
    }

    if (tid < steps_per_fold) {
        sdata[tid] = 0.0f;
    }
    for (int chn_id=tid; chn_id<channel; chn_id += block_size) {
        prev_o[chn_id] = 0.0f;
    }
    __syncthreads();

    for (int patch_left=sbatch_starts[0]; patch_left < sbatch_starts[1]; patch_left+=steps_per_fold) {
        // 每个example的长度右边界判定，以及屏蔽过剩线程tid。
        bool before_end = (patch_left + tid < sbatch_starts[1]) && (tid < steps_per_fold); 
        int step_id = patch_left + tid - sbatch_starts[0];

        // 1. 计算patch max score
        float original_score;
        if (before_end) {
            original_score = scores[(patch_left + tid) * query_heads + head_id];
        } else {
            original_score = -8192.0f;
        }
        if (tid < steps_per_fold) {
            sdata[tid] = original_score;
        }

        __syncthreads();
        for (int pow=steps_per_fold/2; pow>0; pow>>=1){
            if (tid < pow && before_end){
                if (sdata[tid] < sdata[tid+pow]) {
                    sdata[tid] = sdata[tid+pow];
                }
            }
            __syncthreads();
        }
        if (tid==0){
            m = max(m, sdata[0]);
            // if (head_id == 0) {
            //     printf("batch=%i, head=%i, patch=[%i, %i), end=%i, max=max(sdata0=%f, m=%f)\n", batch_id, head_id, patch_left, patch_left+steps_per_fold, sbatch_starts[1], sdata[0], m);
            // }
        }
        __syncthreads();

        // 2. 计算patch partition， (sdata = local_partition_reducer)
        float local_exp_score = exp(original_score - m);
        if (tid < steps_per_fold) {
            sdata[tid] = local_exp_score;
        }
        __syncthreads();
        for (int pow=steps_per_fold/2; pow>0; pow>>=1){
            if (tid<pow && before_end) {
                sdata[tid] = sdata[tid] + sdata[tid+pow];
            }
            __syncthreads();
        }
        
        // 3. 更新rescale factor以及截至当前的Z
        if (tid == 0) {
            rescale_factor = exp(prev_m - m);
            // if (head_id == 0) {
            //     printf("batch=%i, head=%i, patch=[%i, %i), end=%i, prev_partition=%f, rescale_factor=%f, local_partition=%f, \n", batch_id, head_id, patch_left, patch_left+steps_per_fold, sbatch_starts[1], l, rescale_factor, sdata[0]);
            // }
            l = l * rescale_factor + sdata[0];
        }

        if (tid < steps_per_fold) {
            sdata[tid] = local_exp_score;
        }
        __syncthreads();

        // 4. prev_o根据当前patch的rescale factor进行更新
        for (int chn_id = tid; chn_id < channel; chn_id += block_size) {
            prev_o[chn_id] = prev_o[chn_id] * rescale_factor;
        }
        __syncthreads();

        // 5. 计算当前patch内的matmul: score[1, steps_per_fold] @ values[steps_per_fold, channel] using parallel_num=steps_per_fold
        // 之后累加当前的o到prev_o. (sdata = local_exp_score)
        // 这一步开放全部threads运算，不再限制 tid < steps_per_fold
        int valid_length = sbatch_starts[1] - patch_left;
        int patch_start_step = patch_left - sbatch_starts[0];
        if (valid_length > 0) {
            __half* value_data_start = sbatch_kv_ptrs[1]; // [L, H, channel]
            int task_len = valid_length < steps_per_fold ? valid_length : steps_per_fold;
            
            // // if (tid == 0) {
            // //     __half value_read0 = value_data_start[cache_size * 0 + kv_head_id * channel];
            // //     __half value_read_last = value_data_start[cache_size * (task_len-1) + kv_head_id * channel + channel-1];
            // //     printf("final matmul patch: [b=%i, hq=%i, hk=%i], patch=[%i, %i), end=%i, task_len=%i, value[0]=%f, value[-1]=%f\n", batch_id, head_id, kv_head_id, patch_left, patch_left+steps_per_fold, sbatch_starts[1], task_len, __half2float(value_read0), __half2float(value_read_last));
            // // }

            float accumulator = 0.0f;
            // 间隔dt=4累加，每个block处理 512=128(channel)* 4(step)个数据。
            for (int t=out_accu_t_offset; t<task_len; t+=block_channel_ratio) {
                int step_id = patch_start_step + t;
                __half value_read = value_data_start[(cache_size * step_id) + kv_head_id * channel + out_accu_chn_id];
                accumulator += (sdata[t] * __half2float(value_read));
            }
            // out_accu 存储数据是 [block_channel_ratio, channel], 需要在block_channel_ratio维度上继续reduce_sum
            out_accu[tid] = accumulator; 
            __syncthreads();
            if (tid < channel) {
                float accumulator2 = 0.0f;
                for (int i=0; i<block_channel_ratio; i++) {
                    accumulator2 += out_accu[i * channel + tid];
                }
                prev_o[tid] = prev_o[tid] + accumulator2;
            }
        }
        __syncthreads();

        if (tid==0) {
            prev_m = m;
        }
    }

    for (int chn_i=tid; chn_i<channel; chn_i+=block_size) {
        // if (tid==0 && head_id == 0) {
        //     printf("o[%lu]=%f/%f\n", (batch_id * hidden_size) + head_id * channel + chn_i, prev_o[chn_i], l);
        // }
        outs[(batch_id * hidden_size) + head_id * channel + chn_i] = __float2half(prev_o[chn_i] / l);
    }
}


template <int powers>
__global__ void RMSNormKernel(__half* out, __half* in, __half* norm_w, int rows, int padded_cols, int cols, int folds, float eps)
{
    __shared__ float sdata2[powers];
    __shared__ float var;

    int tid = threadIdx.x;
    int row_id = blockIdx.x;

    float sum2 = 0;
    for (int fi=folds-1; fi>=0; fi-=1) {
        int block_pos = threadIdx.x + (row_id * folds + fi) * blockDim.x;
        int block_col = block_pos % padded_cols;
        int source_pos;
        if (block_col < cols) {
            source_pos = (block_pos / padded_cols) * cols + block_col;
        } else {
            source_pos = -1;
        }
        if (source_pos >= 0){
            __half original_val = in[source_pos];
            float hid_val = __half2float(original_val);
            out[source_pos] = original_val;
            sum2 += (hid_val * hid_val);
        }
    }
    
    sdata2[tid] = sum2;
    __syncthreads();

    for (int pow=powers/2; pow>0; pow>>=1){
        if (tid < pow){
            sdata2[tid] = sdata2[tid] + sdata2[tid+pow];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // mean = sdata[0] / cols;
        // var = (sdata2[0] - mean * sdata[0]) / (cols-1); // corrected variance calculation by pytorch, see pytorch official document.
        var = sdata2[0]/ cols; // uses modulus length of a vector to normalize, not std.
        var = sqrt(var + eps);
    }
    __syncthreads();

    for (int fi=folds-1; fi>=0; fi-=1) {
        int block_pos = threadIdx.x + (row_id * folds + fi) * blockDim.x;
        int block_col = block_pos % padded_cols;
        int source_pos;
        if (block_col < cols) {
            source_pos = (block_pos / padded_cols) * cols + block_col;
        } else {
            source_pos = -1;
        }

        if (source_pos >= 0) {
            out[source_pos] = __float2half(__half2float(out[source_pos]) / var * __half2float(norm_w[block_col]));
        }
    }
}

template <int block_size>
__global__ void tileAddKernel(__half* outs, __half* inp, int repeat_num, int stride) {
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

__global__ void addFp32ToFp16KernelWithScaling(__half* out, float* in, int boundary, float scaling) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < boundary) {
        out[tid] += __float2half(in[tid] * scaling);
    }
}

__global__ void copyFp16ToFp32Kernel(float* out, __half* in, int numel) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id<numel) {
        out[id] = __half2float(in[id]);
    }
}

template<int head_num, int channel>
__global__ void RotaryHeadedKernel(__half* q_data, __half* k_data, __half* cos, __half* sin, int dynamic_bsz, int dynamic_l)
{
    // dimGrid(dynamic_l*H), dimBlock(64)
    int half_chn_id = threadIdx.x;
    // int head_id = blockIdx.x % head_num;
    int bkl_id = blockIdx.x;
    int pos =  bkl_id / head_num;
    int half_chn = channel / 2;
    int paring_chn_id = half_chn + half_chn_id;

    __half cos_val_1 = cos[pos*channel + half_chn_id];
    __half sin_val_1 = sin[pos*channel + half_chn_id];
    __half cos_val_2 = cos[pos*channel + paring_chn_id];
    __half sin_val_2 = sin[pos*channel + paring_chn_id];

    int row_head_offset = bkl_id * channel;

    __half orig_q_data_1 = q_data[row_head_offset + half_chn_id];
    __half orig_k_data_1 = k_data[row_head_offset + half_chn_id];
    __half orig_q_data_2 = q_data[row_head_offset + paring_chn_id];
    __half orig_k_data_2 = k_data[row_head_offset + paring_chn_id];

    __half q_embed_1 = orig_q_data_1 * cos_val_1 - orig_q_data_2 * sin_val_1;
    __half q_embed_2 = orig_q_data_2 * cos_val_2 + orig_q_data_1 * sin_val_2;
    __half k_embed_1 = orig_k_data_1 * cos_val_1 - orig_k_data_2 * sin_val_1;
    __half k_embed_2 = orig_k_data_2 * cos_val_2 + orig_k_data_1 * sin_val_2;


    q_data[row_head_offset + half_chn_id] = q_embed_1;
    q_data[row_head_offset + paring_chn_id] = q_embed_2;
    k_data[row_head_offset + half_chn_id] = k_embed_1;
    k_data[row_head_offset + paring_chn_id] = k_embed_2;
}

__global__ void addKernel(__half* out, __half* in, size_t boundary) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid < boundary) {
        out[tid] += in[tid];
    }
}

template<typename index_type>
__global__ void siluMulKernel(__half* out, __half* in, __half* up, int row_num, int out_hidden_size, int padded_hidden_size) {
    // in.shape = [row_num, out_hidden_size], out.shape=[row_num, out_hidden_size]
    index_type tid = threadIdx.x + blockIdx.x * blockDim.x;
    index_type rid = tid / padded_hidden_size;
    index_type cid = tid % padded_hidden_size;
    if (rid < row_num && cid<out_hidden_size) {
        index_type read_pos = rid * out_hidden_size + cid;
        float x = __half2float(in[read_pos]);
        float silux = x / (1.0f + exp(-x));
        float up_val = __half2float(up[read_pos]);
        out[rid*out_hidden_size+cid] =  __float2half(silux * up_val);
    }
}

__global__ void copyLastStepKernel(__half* outs, __half* inputs, int* seq_starts, int channel, int tile_num) {
    // dimGrid = (dynamic_bsz*(ceil(channels/block_size))
    __shared__ __half* tile_data;
    int block_size = blockDim.x;
    int batch_id = blockIdx.x / tile_num;
    int tile_id = blockIdx.x % tile_num;
    int thread_id = threadIdx.x;

    if (thread_id == 0) {
        size_t last_shift = static_cast<size_t>(seq_starts[batch_id+1] - 1) * channel;
        tile_data = inputs + (last_shift + tile_id*block_size);
    }
    __syncthreads();

    int chn_id = tile_id*block_size + thread_id;
    if (chn_id < channel) {
        outs[batch_id * channel + chn_id] = tile_data[thread_id]; 
    }
}

template <int block_size>
__global__ void tileCopyKernel(__half* outs, __half* inp, int repeat_num, int stride) {
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
                    outs[writing_pos] = sdata[j];
                }
            }
        }
    }
}



template <int THREAD_PER_BLOCK>
__global__ void GemvFp16Fp16Kernel2(__half *A, __half *B, __half *C, __half *bias, int k) {
    __shared__ __half sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // 1. 计算
    int st = blockIdx.x * 1;
    int end = st + 1;
    // for (int p = st; p < end; p++) {
    int p = blockIdx.x;

    sdata[tid] = 0;
    for (int i = tid; i < k; i += THREAD_PER_BLOCK) {
        sdata[tid] += A[i] * B[p * k + i];
    }

    __syncthreads();
    for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
        if ((tid & (2 * s - 1)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[p] = sdata[0] + bias[p];
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK>
__global__ void GemvFp16Fp16Kernel2NoBias(__half *A, __half *B, __half *C, int k) {
   __shared__ __half sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    int p = blockIdx.x;
    // 1. 计算
    int st = blockIdx.x * 1;
    int end = st + 1;
    // for (int p = st; p < end; p++) {
    sdata[tid] = 0;
    for (int i = tid; i < k; i += THREAD_PER_BLOCK) {
        sdata[tid] += A[i] * B[p * k + i];
    }
    __syncthreads();
    for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
        if ((tid & (2 * s - 1)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[p] = sdata[0];
    }
    __syncthreads();
}


// ============ Host Functions ==================

std::pair<int, int> get_reduce_partitions(int channel) {
    // ----------------------------
    // 分解 hidden_size 为 hidden_size <= folds * powers, where folds是任意整数，powers是2的任意整数次幂。
    // 分解后最小化 cost = folds + log2(powers) 
    // ----------------------------

    auto reduce_partition_cached = reducePartitionLookup.find(channel);
    if (reduce_partition_cached != reducePartitionLookup.end()) {
        return reduce_partition_cached->second;
    }

    const int max_block_size = 512;
    const int min_block_size = 32;

    if (channel >= max_block_size && (channel % max_block_size)==0) {
        return std::pair<int, int>(max_block_size, channel/max_block_size);
    }

    int trial_bundle_size = max_block_size;
    int min_cost = 10000;
    int optimum_powers = max_block_size;
    int optimum_folds = (int)(ceil((float)channel / max_block_size));
    int powers;
    int folds;
    int k;
    // int optimum_k;
    while (trial_bundle_size >= min_block_size) {
        int k;
        if (channel % trial_bundle_size == 0)
        {
            k = channel;
        }
        else
        {
            k = (int)(ceil((float)channel / trial_bundle_size)) * trial_bundle_size;
        }

        int folds = k;
        int powers = k/folds;
        int pow_num = 0;
        while (folds % 2 == 0 && powers<max_block_size) {
            folds >>=1;
            powers = k/folds;
            pow_num += 1;
        }
        int cost_expected = folds + pow_num;
        // printf("trial_bundle=%d, powers=%d, fold=%d, k=%d, cost=%d\n", trial_bundle_size, powers, folds, k, cost_expected);
        if (cost_expected < min_cost) {
            min_cost = cost_expected;
            optimum_folds = folds;
            optimum_powers = powers;
            // optimum_k = k;
        }
        trial_bundle_size /= 2;
    }

    std::pair<int, int> result_pair = std::pair<int, int>(optimum_powers, optimum_folds);
    reducePartitionLookup[channel] = result_pair;
    return result_pair;
}


void decode_attention(const liteqwen::Data& attended_out, const liteqwen::Data& scores, liteqwen::StringArray input_request_ids, int bl_bound, int batch_maxt, int layer_id, const liteqwen::Data& query, const liteqwen::Data& bl_batch_ids, const liteqwen::Data& cache_start_ptrs, const liteqwen::Data& cache_start_offsets, const liteqwen::Data& start_positions, int max_dynamic_bsz, liteqwen::PipelineKVPool* kv_cache_ref, int kv_heads, int channel, bool cache_ptr_loaded) {
    // attended_out, query: [B, H*D], 有效输出只有[dynamic_bsz, 1, H, D]. B = max_dynamic_bsz, 真实dynamic_bsz=len(input_request_ids)<=B
    // scores: 预分配shape=[max_BL, H], 不会全部被使用, 只使用紧凑排列的bl部分。
    // bl_batch_ids: [BL], dtype=uint8, 存储每个动态bl位置对应的batch_id
    // cache_start_ptrs: [2* sizeof(void*) * B / sizeof(uint8_t)], dtype=uint8_t。存储2*bsz个__half*（强转void*后），[k1_ptr, v1_ptr, k2_ptr, v2_ptr, ...]
    // start_positions: [B+1], dtype=int32, 动态batch拼接后的张量（attended_out）的紧凑排布位置，从pos=0开始。详细见BatchInputPreparer::BatchUpdate方法。

    int dynamic_bsz = (int)(input_request_ids.size());
    if (channel != 128) {
        printf("attention should use hidden_per_head=128, check model config or implement attention using other hidden_per_head.\n");
        throw("");
    }
    float dim_sqrt = sqrt(128.0f);

    // 根据input_request_ids内存在的request_id，找到kv-cache内每个样本在对应layer上的__half*初始位置，以void**形式传入cache_start_ptrs。
    kv_cache_ref->read_batch_kv_ref(input_request_ids, layer_id, cache_start_ptrs, cache_start_offsets, max_dynamic_bsz, 0, cache_ptr_loaded);

    int query_heads = scores.shape[1];
    int block_num = query_heads * bl_bound;
    const int grid_bound = 12288;
    const int min_block_size = 32;

    float* score_data = (float*)scores.cudaData;
    // __half* bhld_value_data = (__half*)bhld_value.cudaData;
    __half* query_data = (__half*) query.cudaData;
    void** cache_ptr_data = (void**)cache_start_ptrs.cudaData;
    uint8_t* bl_batch_mapping_data = (uint8_t*)bl_batch_ids.cudaData;
    int* batch_start_data = (int*)start_positions.cudaData;


    if (block_num > grid_bound) {
        dim3 dimGrid(grid_bound);
        dim3 dimBlock(128);
        batch_gqa_decode_score<128, grid_bound><<<dimGrid, dimBlock>>>(block_num, score_data, query_data, cache_ptr_data, dynamic_bsz, bl_batch_mapping_data, batch_start_data, query_heads, kv_heads, batch_maxt, dim_sqrt);
    } else {
        dim3 dimGrid(block_num);
        dim3 dimBlock(128);
        batch_gqa_decode_score<128, grid_bound><<<dimGrid, dimBlock>>>(block_num, score_data, query_data, cache_ptr_data, dynamic_bsz, bl_batch_mapping_data, batch_start_data, query_heads, kv_heads, batch_maxt, dim_sqrt);     
    }
    // scores.const_print(std::string("scores"));

    const int block_size = min_block_size*16; // 512
    const int block_channel_ratio = block_size / 128;
    std::pair<int, int> reduce_info = get_reduce_partitions(batch_maxt);
    int powers = reduce_info.first;
    int folds = reduce_info.second;
    dim3 decodeBlock(block_size);
    dim3 decodeGrid(dynamic_bsz * query_heads);
    __half* out_data = (__half*)attended_out.cudaData;
    // printf("batch_maxt=%i -> (powers=%i, folds=%i). dynamic_bsz=%i\n", batch_maxt, powers, folds, dynamic_bsz);
    // 使用flash_attention的方式融合了softmax以及matmul(prob, values)
    if (powers <= min_block_size) { // 32
        SoftmaxFuseDecodeAttnKernel<min_block_size, 128, block_size, block_channel_ratio><<<decodeGrid, decodeBlock>>>(out_data, score_data, cache_ptr_data, batch_start_data, query_heads, kv_heads);
    } else if (powers <= min_block_size*2) { // 64
        SoftmaxFuseDecodeAttnKernel<min_block_size*2, 128, block_size, block_channel_ratio><<<decodeGrid, decodeBlock>>>(out_data, score_data, cache_ptr_data, batch_start_data, query_heads, kv_heads);
    } else if (powers <= min_block_size*4) { //128
        SoftmaxFuseDecodeAttnKernel<min_block_size*4, 128, block_size, block_channel_ratio><<<decodeGrid, decodeBlock>>>(out_data, score_data, cache_ptr_data, batch_start_data, query_heads, kv_heads);
    } else if (powers <= min_block_size*8) { // 256
        SoftmaxFuseDecodeAttnKernel<min_block_size*8, 128, block_size, block_channel_ratio><<<decodeGrid, decodeBlock>>>(out_data, score_data, cache_ptr_data, batch_start_data, query_heads, kv_heads);
    } else { //512
        SoftmaxFuseDecodeAttnKernel<min_block_size*16, 128, block_size, block_channel_ratio><<<decodeGrid, decodeBlock>>>(out_data, score_data, cache_ptr_data, batch_start_data, query_heads, kv_heads);
    }
}


void cpu_embedding_fwd(int gpu_id, const liteqwen::Data& out_tensor,  const liteqwen::Data& input_ids, const liteqwen::Data& embedding_weights, int input_offset, int lookup_len, int channel) {

    int* cpu_input_ids = new int[lookup_len];
    size_t full_size = lookup_len * channel;
    __half* cpu_embeddings = new __half[full_size];
    uint8_t* dst_data = reinterpret_cast<uint8_t*>(cpu_input_ids);
    uint8_t* src_data = reinterpret_cast<uint8_t*>(input_ids.cudaData);
    __half* gpu_write_data = reinterpret_cast<__half*>(out_tensor.cudaData);
    size_t uint8_src_offset = static_cast<size_t>(input_offset) * 4;
    size_t tmp_gpu_size = lookup_len * 4;
    cudaMemcpy(dst_data, (src_data+uint8_src_offset), tmp_gpu_size, cudaMemcpyDeviceToHost);
    DeviceSynchronize();

    liteqwen::cpu_embedding_copy((uint8_t*)cpu_embeddings, (uint8_t*)embedding_weights.cpuData, cpu_input_ids, lookup_len, channel);
    DeviceSynchronize();
    cudaMemcpy(gpu_write_data, cpu_embeddings, lookup_len*channel*sizeof(__half), cudaMemcpyHostToDevice);
    DeviceSynchronize();
    delete cpu_input_ids;
    delete cpu_embeddings;
}

void rotary_lookup(bool is_prefill, int gpu_id, const liteqwen::Data& out_cos, const liteqwen::Data& out_sin, const liteqwen::Data& batch_bids, const liteqwen::Data& batch_starts, const liteqwen::Data& cos_gpu, const liteqwen::Data& sin_gpu, int input_offset, int lookup_len, int channel, int dynamic_bsz) {
    dim3 dimBlock(128);
    dim3 dimGrid((int)(ceil((float)(channel) / 128)) * lookup_len);

    // __half* outs, __half* embedding_weights, uint8_t* batch_ids, int* batch_starts, bool is_prefill, int lookup_len, int channel, int bsz
    embeddingDynamicLookupKernel<<<dimGrid, dimBlock>>>((__half*)out_cos.cudaData, (__half*)cos_gpu.cudaData, (uint8_t*)batch_bids.cudaData, (int*)batch_starts.cudaData, is_prefill, lookup_len, channel, dynamic_bsz);
    embeddingDynamicLookupKernel<<<dimGrid, dimBlock>>>((__half*)out_sin.cudaData, (__half*)sin_gpu.cudaData, (uint8_t*)batch_bids.cudaData, (int*)batch_starts.cudaData, is_prefill, lookup_len, channel, dynamic_bsz);
}

void rms_norm(const liteqwen::Data& out_tensor, const liteqwen::Data& hidden_in, const liteqwen::Data& norm_w, int sequence_length, int channel) {
    std::pair<int, int> reduce_info = get_reduce_partitions(channel);
    int powers = reduce_info.first;
    int folds = reduce_info.second;
    int k = powers * folds;
    dim3 dimBlock(powers);
    dim3 dimGrid(1 * sequence_length);

    // const int max_block_size = 512;
    const int min_block_size = 32;
    float eps = 1e-06;

    if (powers <= min_block_size){
        RMSNormKernel<min_block_size><<<dimGrid, dimBlock>>>((__half*)out_tensor.cudaData, (__half*)hidden_in.cudaData, (__half*)norm_w.cudaData, 1 * sequence_length, k, channel, folds, eps);
    }
    else if (powers <= min_block_size*2) { // 64
        RMSNormKernel<min_block_size*2><<<dimGrid, dimBlock>>>((__half*)out_tensor.cudaData, (__half*)hidden_in.cudaData, (__half*)norm_w.cudaData, 1 * sequence_length, k, channel, folds, eps);
    } else if (powers <= min_block_size*4) { //128
        RMSNormKernel<min_block_size*4><<<dimGrid, dimBlock>>>((__half*)out_tensor.cudaData, (__half*)hidden_in.cudaData, (__half*)norm_w.cudaData, 1 * sequence_length, k, channel, folds, eps);
    } else if (powers <= min_block_size*8) { // 256
        RMSNormKernel<min_block_size*8><<<dimGrid, dimBlock>>>((__half*)out_tensor.cudaData, (__half*)hidden_in.cudaData, (__half*)norm_w.cudaData, 1 * sequence_length, k, channel, folds, eps);
    } else { //512
        RMSNormKernel<min_block_size*16><<<dimGrid, dimBlock>>>((__half*)out_tensor.cudaData, (__half*)hidden_in.cudaData, (__half*)norm_w.cudaData, 1 * sequence_length, k, channel, folds, eps);
    }
}


void quant4_linear_fwd(const liteqwen::Data& out_tensor, const liteqwen::Data& x, uintptr_t w_ref, const liteqwen::Data& bias, bool use_bias) {
    int dim_num = out_tensor.shape.size();
    int n;
    if (dim_num == 4) {
        n = out_tensor.shape[dim_num-2] * out_tensor.shape[dim_num-1];
    } else {
        n = out_tensor.shape[dim_num-1];
    }
    int m = static_cast<int>((out_tensor.numel() / n));
    cublasHandle_t handle = get_cublas_handler(x.gpu_id);
    q4_matmul(x, w_ref, out_tensor, handle);
    if (use_bias) {
        dim3 dimBlock(32);
        dim3 dimGrid((int)(ceil((float)(n) / 32)));
        __half* bias_data = (__half*)(bias.cudaData);
        __half* out_data = (__half*)(out_tensor.cudaData);
        tileAddKernel<32><<<dimGrid, dimBlock>>>(out_data, bias_data, m, n); //使用bias repeat进行初始化。
    }
}

void quant4_lora_linear_fwd(const liteqwen::Data& out_tensor, const liteqwen::Data& x, uintptr_t w_ref, const liteqwen::Data& bias, bool use_bias, const liteqwen::Data& fp32_x, const liteqwen::Data& loraA_out, const liteqwen::Data& loraB_out, const liteqwen::Data& loraA_W, const liteqwen::Data& loraB_W, int r, float lora_scaling) {
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

    q4_matmul(x, w_ref, out_tensor, handle);
    if (use_bias) {
        dim3 dimBlock(32);
        dim3 dimGrid((int)(ceil((float)(n) / 32)));
        __half* bias_data = (__half*)(bias.cudaData);
        tileAddKernel<32><<<dimGrid, dimBlock>>>(out_data, bias_data, m, n); //使用bias repeat进行初始化。
    }

    if (m > 0) {
        float* fp32_x_data = (float*)(fp32_x.cudaData);
        float* loraB_out_data = (float*)(loraB_out.cudaData);

        // ========= loraA =========
        int inp_numel = m*k;
        dim3 dimBlockTmp(256);
        dim3 dimGridTmp((int)ceil((float)(inp_numel)/256));
        copyFp16ToFp32Kernel<<<dimGridTmp, dimBlockTmp>>>(fp32_x_data, (__half*)x.cudaData, inp_numel);
        // out_tensor.const_print(std::string("linear_out"));
        float f_beta = 0.0f;
        float f_alpha = 1.0f;
        cudaDataType_t LA_AType, LA_BType, LA_CType, LA_ComputeType;
        LA_AType = LA_BType = LA_CType = LA_ComputeType = CUDA_R_32F;
        cublasStatus_t la_status = cublasGemmEx(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                r, m, k,
                                &f_alpha, 
                                loraA_W_data, LA_AType, k, //lda=k [k x n] -> Cub Transpose to [n x k]
                                fp32_x_data, LA_BType, k, //ldb=k, [k x m]
                                &f_beta,
                                loraA_out_data, LA_CType, r, //ldc=n, out shape = [n x m] = [n x k] x [k x m] 
                                LA_ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));  
        if (la_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        // ========= loraB =========
        // loraA_out.const_print(std::string("loraA_out"));
        cudaDataType_t LB_AType, LB_BType, LB_CType, LB_ComputeType;
        LB_AType = LB_BType = LB_CType = LB_ComputeType = CUDA_R_32F;
        cublasStatus_t lb_status = cublasGemmEx(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                n, m, r,
                                &f_alpha, 
                                loraB_W_data, LB_AType, r, //lda=k
                                loraA_out_data, LB_BType, r, //ldb=k
                                &f_beta,
                                loraB_out_data, LB_CType, n, //ldc=n
                                LB_ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));  
        if (lb_status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }
        
        // loraB_out.const_print(std::string("loraB_out"));
        // ====== adding =======
        int boundary = m*n;
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid((int)ceil((float)(boundary)/BLOCK_SIZE));
        addFp32ToFp16KernelWithScaling<<<dimGrid, dimBlock>>>(out_data, loraB_out_data, boundary, lora_scaling);
    } 
    // else {
    //     dim3 dimBlockLoraA(256);
    //     dim3 dimGridLoraA(r);
    //     // x.const_print(std::string("loraA_x_k"+std::to_string(k)));
    //     GemvFp16Fp32Kernel2NoBias<256> <<<dimGridLoraA, dimBlockLoraA>>>((__half*)x.cudaData, loraA_W_data, loraA_out_data, k);
    //     // loraA_out.const_print(std::string("loraA_out"));
    //     dim3 dimBlockLoraB(64);
    //     dim3 dimGridLoraB(n);
    //     GemvFp32Fp16AddingKernel2NoBiasWithScaling<64><<<dimGridLoraB, dimBlockLoraB>>>(loraA_out_data, loraB_W_data, out_data, r, lora_scaling);
    // }
}

void apply_rotary_embeddings(const liteqwen::Data& query, const liteqwen::Data& key, int dynamic_bsz, int dynamic_l, int hidden_size, int head_num, const liteqwen::Data& cos, const liteqwen::Data& sin) {
    int channel = hidden_size / head_num;
    int half_channel = channel / 2;
    if (channel != 128 || head_num != 40) {
        printf("wrong with qwen2-14b rotary shape, make sure attention head num=40 and per_head_hidden=128\n");
        throw("");
    }
    __half* query_data = (__half*)query.cudaData;
    __half* key_data = (__half*)key.cudaData;
    __half* cos_data = (__half*)cos.cudaData;
    __half* sin_data = (__half*)sin.cudaData;
    // NOTICE: dynamic_l = dynamic_bl if is_prefill, else dynamic_l=dynamic_bsz
    dim3 dimGrid(dynamic_l*head_num);
    // block_size = 64 = half_channel
    RotaryHeadedKernel<40, 128><<<dimGrid, 64>>>(query_data, key_data, cos_data, sin_data, dynamic_bsz, dynamic_l);
}

void inplace_add_half(const liteqwen::Data& a, const liteqwen::Data& b, size_t boundary) {
    __half* a_data = (__half*)(a.cudaData);
    __half* b_data = (__half*)(b.cudaData);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)ceil((float)(boundary)/BLOCK_SIZE));
    addKernel<<<dimGrid, dimBlock>>>(a_data, b_data, boundary);
}

void qwen_silu(const liteqwen::Data& out_tensor, const liteqwen::Data& x, const liteqwen::Data& up, int row_num, int ffn_hidden_size) {
    __half* out_data = (__half*)out_tensor.cudaData;
    __half* x_data = (__half*)x.cudaData;
    __half* up_data = (__half*)up.cudaData;
    int grid_per_row = (int)ceil((float)(ffn_hidden_size)/BLOCK_SIZE);
    int padded_hidden = grid_per_row * BLOCK_SIZE;
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(grid_per_row*row_num);
    if (row_num >= 1024) { // 防止int溢出
        siluMulKernel<size_t><<<dimGrid, dimBlock>>>(out_data, x_data, up_data, row_num, ffn_hidden_size, padded_hidden);
    } else {
        siluMulKernel<int><<<dimGrid, dimBlock>>>(out_data, x_data, up_data, row_num, ffn_hidden_size, padded_hidden);
    }
}

void dynamic_slice_last(const liteqwen::Data& final_step_hidden, const liteqwen::Data& hidden_state, const liteqwen::Data& local_qstarts, int dynamic_bsz, int hidden_size) {
    __half* hidden_data = (__half*)hidden_state.cudaData;
    __half* final_data = (__half*) final_step_hidden.cudaData;

    int tile_num = (hidden_size - 1) / BLOCK_SIZE + 1;
    dim3 dimGrid(dynamic_bsz*tile_num);
    dim3 dimBlock(BLOCK_SIZE);
    copyLastStepKernel<<<dimGrid, dimBlock>>>(final_data, hidden_data, (int*)local_qstarts.cudaData, hidden_size, tile_num);
}


void linear_fwd(const liteqwen::Data& out_tensor, const liteqwen::Data& x, const liteqwen::Data& W, const liteqwen::Data& b, int m, int n, int k, bool use_bias) {
    // x[B, H], W[O, H], W is transposed to give output out[B, O]
    __half* out_data = (__half*)(out_tensor.cudaData);
    __half* x_data = (__half*)(x.cudaData);
    __half* W_data = (__half*)(W.cudaData);

    if (m>1) {
        cublasHandle_t handle = get_cublas_handler(W.gpu_id);
        __half h_alpha = __float2half_rn(1.0);
        __half h_beta;

        if (use_bias) {
            dim3 dimBlock(32);
            dim3 dimGrid((int)(ceil((float)(n) / 32)));
            __half* bias_data = (__half*)(b.cudaData);
            tileCopyKernel<32><<<dimGrid, dimBlock>>>(out_data, bias_data, m, n); //使用bias repeat进行初始化。
            h_beta = __float2half_rn(1.0);
        } else {
            // dim3 dimBlock(32);
            // dim3 dimGrid((int)(ceil((float)(n) / 32)));
            // FillContiguousKernel<<<dimGrid, dimBlock>>>(out, __float2half(0.0f), 0, m*n); //初始化置0，可以不使用。
            h_beta = __float2half_rn(0.0);
        }

        // cublasHandle_t handle = getCublasHandle();
        // cudaDeviceSynchronize();

        cudaDataType_t AType, BType, CType, ComputeType;
        AType = BType = CType = ComputeType = CUDA_R_16F;

        // matmul操作计算out=xW时，不对W进行transpose的代码，备用。
        // cublasStatus_t status = cublasGemmEx(handle,
        //                         CUBLAS_OP_N, CUBLAS_OP_N,
        //                         n, m, k,
        //                         &h_alpha, 
        //                         W, AType, n, //lda = n, cublas将W[k,n]视作WT[n, k]=cubA来计算
        //                         x, BType, k, //ldb = k, cublas将x[m,k]视作xT[k,n]=cubB来计算
        //                         &h_beta,
        //                         out, CType, n, // ldc=n, cublas计算 cubA cubB = WT xT = (xW)T = cubC[n, m]，切换回行优先则变回了C=xW
        //                         ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        
        // linear层计算out=xW^T时使用。
        cublasStatus_t status = cublasGemmEx(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                n, m, k,
                                &h_alpha, 
                                W_data, AType, k, //lda=k
                                x_data, BType, k, //ldb=k
                                &h_beta,
                                out_data, CType, n, //ldc=n
                                ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }
        // cudaDeviceSynchronize();
    } else {
        // m=1时，常规的正方形划分m x n矩阵的并行度不够，所以需要尽量使用长条block=[1xmin(hidden,256)]，dimGrid=n来划分并行。
        if (use_bias) {
            dim3 dimBlock2(256);
            dim3 dimGrid2(n);
            __half* bias_data = (__half*)(b.cudaData);
            GemvFp16Fp16Kernel2<256> <<<dimGrid2,dimBlock2>>>(x_data, W_data, out_data, bias_data, k);
            // check_print(std::string("linearCust"), out, m*n, m, n, 2);
        } else {
            dim3 dimBlock2(256);
            dim3 dimGrid2(n);
            GemvFp16Fp16Kernel2NoBias<256> <<<dimGrid2,dimBlock2>>>(x_data, W_data, out_data, k);
        }
    }
}

void dynamic_gqa_tile(const liteqwen::Data& k_proj_layer_tiled, const liteqwen::Data& v_proj_layer_tiled, const liteqwen::Data& k_proj_layer, const liteqwen::Data& v_proj_layer, int dynamic_bl, int attention_heads, int kv_heads, int channels) {
    __half* out_k_data = (__half*)k_proj_layer_tiled.cudaData;
    __half* out_v_data = (__half*)v_proj_layer_tiled.cudaData;

    __half* in_k_data = (__half*) k_proj_layer.cudaData;
    __half* in_v_data = (__half*) v_proj_layer.cudaData;

    if (channels != 128) {
        printf("only support num_hidden_per_head=128 models, other kv channel not implemented.\n");
        throw("");
    }

    const int grid_bound = 12288;
    dim3 dimBlock2(128);
    size_t bound = static_cast<size_t>(dynamic_bl) * kv_heads;
    if (bound>grid_bound) {
        dim3 dimGrid(grid_bound);
        copy_kv_tile_kernel<128, 12288><<<dimGrid, dimBlock2>>>(out_k_data, out_v_data, in_k_data, in_v_data, attention_heads, kv_heads, bound);
    } else {
        dim3 dimGrid(bound);
        copy_kv_tile_kernel<128, 12288><<<dimGrid, dimBlock2>>>(out_k_data, out_v_data, in_k_data, in_v_data, attention_heads, kv_heads, bound);
    }
}
#include "core_cpu.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

#ifdef  __cplusplus
extern "C" {
#endif

void setup_cublas_handler(int gpu_id);
cublasHandle_t get_cublas_handler(int gpu_id);
void print_cuda_info(int gpu_id);
void SetDevice(int gpu_id);
void DeviceSynchronize();

void CopyGPUData(liteqwen::DataType dtype, void* dst, void* src, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length, bool use_kernel);
void CopyBetweenGPUs (liteqwen::DataType dtype, int dstId, void *dst, int srcId, void *src, size_t copy_length);
void UploadData(liteqwen::DataType dtype, void* dst, uint8_t* src, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length);
void UploadInt32(void* dst, uint8_t* src, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length, size_t dst_maxlen, bool right_pad_zero);
void UploadCastFp32ToFp16Data(void* gpu_data, float* cpu_values, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length);
void GpuCastFp16ToFp32(void* dst_data, void* src_data, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length);
void ConstantFill(void* dst_data, liteqwen::DataType data_type,  size_t numel, double value);
void FillArange(void* data, int limit);
void DownloadData(liteqwen::DataType dtype, uint8_t* dst, void* src, size_t dst_offset, size_t src_offset, size_t copy_length);
void PrintRow(std::string row_info, liteqwen::DataType dtype, void* data, size_t row_id, int cols, int print_width);
void CPUConvertFp16ToFp32(float* out, void* in, liteqwen::DataType dtype, size_t numel);

void QuickUploadData(liteqwen::DataType dtype, void* dst, uint8_t* src, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length);

void* GetCudaMalloc(int gpu_id, size_t size, bool managed); // malloc, specifing whether managed.
void CudaFree(void* gpu_arr, bool managed, int gpu_id); // free, specifying whether managed.
void* GetDtypeCudaMalloc(int gpu_id, liteqwen::DataType dtype, size_t numel, bool managed); // malloc according to dtype and numel.
void * ManagedCudaMalloc(int gpu_id, size_t size); // should be called via GetCudaMalloc
void ManagedCudaFree(void *ret, int gpu_id); 
void* CudaDirectMalloc(size_t size); // should be called via GetCudaMalloc
void CudaDirectFree(void *ret);
void ManagedCudaMallocBigBuffer(int gpu_id, size_t size); // allocate to BigBuffer
void ManagedCudaClearBigBuffer(int data_group_start_gpu, int data_group_size); // Clear all BigBuffer data from device=[data_group_start_gpu, ..., data_group_start_gpu+data_group_size]

void CheckGPUValues(liteqwen::DataType dtype, size_t numel, void* data); // slow check and print nan/inf, develop use only.

void WriteKVCacheFromBatch(bool is_prefill, void* block_cache_data, void* activation_data, int gpu_id, size_t layer_data_offset, const liteqwen::Data& pos_starts, int bi, int cache_channel, int example_len);

void MoveGPUKVPtrs(const liteqwen::Data& kv_ptrs, void* cache_data, const liteqwen::Data& gpu_offsets, int bsz);
void WriteGPUKV(bool is_prefill, void* cache_data, const liteqwen::Data& gpu_offsets, std::pair<liteqwen::Data*, liteqwen::Data*> kv_pair, const liteqwen::Data& kstarts, const liteqwen::Data& batch_ids, int dynamic_l, int kv_heads);
void PrintWithShift(void* data, size_t layer_shift, int end_step, int channels);

void MoveToLayerStarts(void** layerData, const liteqwen::Data& gpu_layer_pointer, void* cache_pool_data, size_t layer_stride, int num_layers);


void WriteKVCaches(bool is_prefill, int local_layer_id, const liteqwen::Data& batch_ptrs, std::pair<liteqwen::Data*, liteqwen::Data*> kv_pair, const liteqwen::Data& act_kstarts, const liteqwen::Data& batch_ids, int dynamic_l, int kv_heads, int max_B);
void ScatterLayerKVExamplePtrs(const liteqwen::Data& batch_layer_starts, const liteqwen::Data& example_numel_shifts_gpu, const liteqwen::Data& gpu_layer_start_ptrs, int dynamic_bsz);


#ifdef  __cplusplus
}
#endif


#ifndef CORE_GPU_H
#define CORE_GPU_H

// 1D grid parallel kernel. capable of dealing with layer norm, transpose, softmax, ... etc, but not matmul, attention.
template <
    // The datatype of data
    typename scalar_t_,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    typename index_t_,
    int kPerBlock_, // launching block_size calculated according to reduce op.
    int maxThreadsPerBlock_> // max block_size, throws error if kPerBlock_>maxThreadsPerBlock_
    // int kMaxK_ =  2147483647> // upperbound on channel=`max(value.shape[-1], query.shape[-1])`
struct DynamicKernel {
    using scalar_t = scalar_t_;
    using index_t = index_t_;
    static constexpr int kPerBlock = kPerBlock_;
    static_assert(kPerBlock % 32 == 0, "");
    // static constexpr int kMaxK = kMaxK_; // probably be equalt to channels or hidden_size.
    // static constexpr bool kSingleValueIteration = kMaxK <= kPerBlock; // if not kSingleValueIteration, folds would apply to threads to process all channels.
    // Launch bounds
    static constexpr int kNumWarpsPerBlock = kPerBlock / 32;
    // static constexpr int kWarpSize = 32;
    // static constexpr int kNumThreads = kWarpSize * kNumWarpsPerBlock; // for 1d grid, kNumThreads=kPerBlock. Would not be the case for 2d grid.
    static constexpr int maxThreadsPerBlock = kPerBlock; // may be more efficient than default? kPerBlock acting as maxThreadsPerBlock
    // static constexpr int kMinBlocksPerSm = getWarpsPerSmFw<scalar_t, ArchTag>() / kNumWarpsPerBlock; // getWarpsPerSmFw=12 for sm<80
    static constexpr int kMinBlocksPerSm = 12/kNumWarpsPerBlock;
    
    // static constexpr int maxThreadsPerBlock = maxThreadsPerBlock_; //default blocks
    // static constexpr int kMinBlocksPerSm = 1;

    struct Params {
        scalar_t* data_ptr = nullptr; // start of each example
        int8_t* data_bids_ptr;
        int* seqstart_ptr = nullptr;
        int step_id;
        int batch_id;

        int blks_per_step;
        int length_stride;
        int head_num;
        int channel;
        int seq_start;
        int next_seq_start;
        int seq_len;

#ifdef __CUDA_ARCH__
        __forceinline__ __device__  bool advance_to_example() {
            if (blks_per_step>0) {
                batch_id = static_cast<int>(data_bids_ptr[blockIdx.x/blks_per_step]);
            } else {
                batch_id = blockIdx.x / head_num;
            }

            // int step_offset_on_blks = (blockIdx.x % blks_per_step);
            seqstart_ptr += batch_id;
            seq_start = seqstart_ptr[0];
            next_seq_start = seqstart_ptr[1];
            seq_len = next_seq_start - seq_start;
            
            index_t example_offset;
            if (blks_per_step>0) {
                step_id = (blockIdx.x / blks_per_step) - seq_start;
                // for dimGrid(B,L): step_offset_on_blks=0, example_offset=H*D
                // for dimGrid(B, L*H): step_offset_on_blks=head_id, example_offset=H*D
                example_offset = static_cast<index_t>(seq_start) * length_stride;
            } else {
                step_id = -1; // different block does not distinguish steps, but heads.
                // for dimGrid(B, H): length_stride=channel.
                example_offset = static_cast<index_t>(seq_start) * length_stride * head_num;
            }
            
            data_ptr += example_offset;
            // printf("entering block advance at blockIdx.x=%i, threadIdx.x=%i\n", blockIdx.x, threadIdx.x);
            return true;
        }
        
#endif // __CUDA_ARCH__
        __host__ void set_dynamic_bl(int head_num_, int channel_) {
            // 默认dimGrid(BL)，每个step对应一个block，所以blks_per_step=1, length_stride=H*D。
            blks_per_step = 1;
            channel = channel_;
            head_num = head_num_;
            length_stride = head_num_ * channel_;
        }

        __host__ void set_dynamic_blh(int head_num_, int channel_) {
            // dimGrid(BL*H)，每个step对应H个block，所以blks_per_step=H, length_stride=H*D。
            blks_per_step = head_num_;
            channel = channel_;
            head_num = head_num_;
            length_stride = head_num_ * channel_;
        }

        __host__ void set_dynamic_bh(int head_num_, int channel_) {
            // dimGrid(B*H), L被threadIdx描述。
            blks_per_step = -1;
            channel = channel_;
            head_num = head_num_;
            length_stride = channel_;
        }

        __host__ dim3 get_dimGrid_bl(size_t dynamic_boundary) const {
            return dim3(static_cast<int>(dynamic_boundary));
        }

        __host__ dim3 get_dimGrid_blh(size_t dynamic_boundary, int head_num) const {
            return dim3(static_cast<int>(dynamic_boundary) * head_num);
        }

        __host__ dim3 get_dimGrid_bh(int dynamic_bsz, int head_num) const {
            return dim3(dynamic_bsz * head_num);
        }
    };
    
#ifdef __CUDA_ARCH__
    static void __forceinline__ __device__ check_block_start(Params& p) {
        // int step_offset_on_blks = (blockIdx.x % p.blks_per_step);
        index_t example_offset;
        if (p.blks_per_step>0) {
            example_offset = static_cast<index_t>(p.seq_start) * p.length_stride;
        } else {
            example_offset = static_cast<index_t>(p.seq_start) * p.length_stride * p.head_num;
        }
        if (threadIdx.x == 0) {
            printf("block(%i) printing, example_offset={%lu}, bid=%i, step_id=%i\n", blockIdx.x, static_cast<unsigned long>(example_offset), p.batch_id, p.step_id);
        }
    }
#endif // __CUDA_ARCH__
};


// __global__ void 
// #ifdef __CUDACC__
// __launch_bounds__(
//     DynamicKernel<DT, IT, block_size, max_block_size>::maxThreadsPerBlock,
//     DynamicKernel<DT, IT, block_size, max_block_size>::kMinBlocksPerSm)
// #endif // __CUDACC__
// launch_matched(typename DynamicKernel<DT, IT, block_size, max_block_size>::Params p);

// //dtype, index_type, T cb=function(kernel_struct, pre_launch_func(kernel_struct::Params)), where pre_launch_func defined in dummy_test_launch, assigning kernel_struct values.
// template <typename DT, typename IT, int block_size, int max_block_size, typename T>
// void dispatch_kernel(T cb) {
//     // ignoring matching DT and __half
//     // launch_matched(cb);
//     cb(DynamicKernel<DT, IT, block_size, max_block_size>(), launch_matched);
// }

void dynamic_check_launch(const liteqwen::Data&  x, const liteqwen::Data& batch_ids, const liteqwen::Data& start_positions, int dynamic_bsz, size_t dynamic_boundary, int head_num, int channel);
void check_cache_read(const liteqwen::Data& kv_pointers, int dynamic_bsz, const liteqwen::Data& seq_starts, int channel);


#endif // CORE_GPU_H
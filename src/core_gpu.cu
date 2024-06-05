#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "core_cpu.h"
#include "core_gpu.cuh"


#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)
void showError(cudaError_t result, char const* const message, const char* const file,
           int const line) {
    if (cudaSuccess != result) {
        printf("%s\n  CUDA error = %d, %s at %s:%d\n  '%s'\n",
            message, result, cudaGetErrorName(result), file, line, cudaGetErrorString(result));
    }  
}

__global__ void print_fp32_data(float* data, size_t offset, size_t stride, int prt_size, bool no_tail) {
    for (int i=0; i< prt_size; i++) {
        printf("%f|", data[offset + i]);
    }
    if (prt_size < stride && !no_tail) {
        printf("...");
        size_t end_offset = offset + stride - prt_size;
        for (int j=0; j< prt_size; j++) {
            printf("%f|", data[end_offset + j]);
        }
    }
    printf("\n");
}

__global__ void print_fp16_data(__half* data, size_t offset, size_t stride, int prt_size, bool no_tail) {
    for (int i=0; i< prt_size; i++) {
        printf("%f|", __half2float(data[offset + i]));
    }
    if (prt_size < stride && !no_tail) {
        printf("...");
        size_t end_offset = offset + stride - prt_size;
        for (int j=0; j< prt_size; j++) {
            printf("%f|", __half2float(data[end_offset + j]));
        }
    }
    printf("\n");
}

__global__ void print_int_data(int* data, size_t offset, size_t stride, int prt_size, bool no_tail) {
    for (int i=0; i< prt_size; i++) {
        printf("%i|", data[offset + i]);
    }
    if (prt_size < stride && !no_tail) {
        printf("...");
        size_t end_offset = offset + stride - prt_size;
        for (int j=0; j< prt_size; j++) {
            printf("%i|", data[end_offset + j]);
        }
    }
    printf("\n");
}

__global__ void print_invalid_float(float* data, size_t numel) {
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    int invalid_ct = 0;
    for (size_t pos=start; pos<numel; pos +=256) {
        if (pos < numel && invalid_ct==0) {
            float val = data[pos];
            if (isnan(val) || isinf(val)) {
                printf("[%i,%f]", pos, val);
                invalid_ct +=1;
            }
        }
    }
}

__global__ void print_invalid_fp16(__half* data, size_t numel) {
    size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    int invalid_ct = 0;
    for (size_t pos=start; pos<numel; pos += 256) {
        if (pos < numel && invalid_ct==0) {
            __half val = data[pos];
            if (__hisnan(val) || __hisinf(val)) {
                printf("[%i,%f]", pos, __half2float(val));
                invalid_ct +=1;
            }
        }
    }
}

__global__ void print_with_shift(__half* data, size_t shift, size_t numel, size_t stride, int prt_row_num, int prt_col_num) {
    printf("\n-----------------\n");
    __half* new_data = data + shift;
    int nrows = numel / stride;
    int ncols =stride;
    for (size_t i=0; i<prt_row_num&&i<nrows; i++) {
        printf("r=%lu|", i);
        for (int j=0; j<prt_col_num && j<stride; j++) {
            printf("%f|", __half2float(new_data[i*stride+j]));
        }
        printf("...");
        for (int j=stride-prt_col_num-1; j>=prt_col_num&&j<stride; j++) {
            printf("%f|", __half2float(new_data[i*stride+j]));
        }
        printf("\n");
    }

    for (size_t i=nrows-prt_row_num; i>=prt_row_num && i<nrows; i++) {
        printf("r=%lu|", i);
        for (int j=0; j<prt_col_num && j<stride; j++) {
            printf("%f|", __half2float(new_data[i*stride+j]));
        }
        printf("...");
        for (int j=stride-prt_col_num-1; j>=prt_col_num&&j<stride; j++) {
            printf("%f|", __half2float(new_data[i*stride+j]));
        }
        printf("\n");        
    }
    printf("-----------------\n");
}

template<typename index_t=int64_t>
__global__ void CopyContiguousKernel(uint8_t* dst, uint8_t* src, index_t dst_offset, index_t src_offset, index_t copy_length) {
    index_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < copy_length) {
        dst[dst_offset + pos] = src[src_offset + pos];
    }
}

__global__ void CopyContiguousInt32Kernel(int* dst, int* src, unsigned long dst_offset, unsigned long src_offset, unsigned long copy_length) {
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < copy_length) {
        dst[dst_offset + pos] = src[src_offset + pos];
    }
}

__global__ void CopyContiguousInt32KernelRightPadZero(int* dst, int* src, unsigned long dst_offset, unsigned long src_offset, unsigned long copy_length, unsigned long dst_maxlen) {
    unsigned long pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < copy_length) {
        dst[dst_offset + pos] = src[src_offset + pos];
    } else if (dst_offset + pos < dst_maxlen) {
        dst[dst_offset + pos] = 0;
    }
}

__global__ void FillArangeKernel(int* data, int limit) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < limit) {
        data[pos] = pos;
    }
}

template<typename dtype>
__global__ void constValueKernel(dtype* data, size_t limit, dtype value) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < limit) {
        data[pos] = value;
    }
}

__global__ void CopyCastFp32ToFp16ContiguousKernel(__half* dst, float* src, size_t dst_offset, size_t src_offset, size_t copy_length) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < copy_length) {
        dst[dst_offset + pos] = __float2half(src[src_offset + pos]);
    }
}

__global__ void CopyCastFp16ToFp32ContiguousKernel(float* dst, __half* src, size_t dst_offset, size_t src_offset, size_t copy_length) {
    size_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < copy_length) {
        dst[dst_offset + pos] = __half2float(src[src_offset + pos]);
    }
}


__global__ void move_ptrs_according_to_shifts(void** pointers, __half* cache_data, size_t* shifts, int ptr_boundary) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ptr_boundary) {
        __half* hptr = cache_data + shifts[tid];
        pointers[tid] = reinterpret_cast<void*>(hptr);
    }
}

__global__ void copyBatchDecodeCacheDataKernel(void* cache_data, size_t* write_shifts, __half* key_act, __half* value_act, int* key_starts, int dynamic_l, int kv_heads) {
    int batch_id = blockIdx.x / kv_heads;
    int head_id = blockIdx.x % kv_heads;
    int tid = threadIdx.x;
    const int channels = 128;
    __shared__ size_t sbatch_cache_shifts[2]; // cache example shift for key & value.
    __shared__ int step_id;

    __half* h_cache_data = reinterpret_cast<__half*>(cache_data);
    
    if (tid < 2) {
        sbatch_cache_shifts[tid] = write_shifts[2*batch_id+tid];
    }
    if (tid==0) {
        step_id = key_starts[batch_id+1] - key_starts[batch_id] - 1;
        // printf("batch_id=%i, head_id=%i, dynamic_l=%i, pos=[%i,%i), kshift=%lu, vshift=%lu\n", batch_id, head_id, dynamic_l, sbatch_kv_act_pos[0], sbatch_kv_act_pos[1], sbatch_cache_shifts[0], sbatch_cache_shifts[1]);
    }
    __syncthreads();

    // read according to dynamic batched pos.
    size_t read_pos_chn_shift = static_cast<size_t>(batch_id) * kv_heads * channels + head_id * channels + tid;
    // write according to step_id
    size_t write_pos_chn_shift = static_cast<size_t>(step_id) * kv_heads * channels + head_id * channels + tid;

    if (step_id >= 0) {
        h_cache_data[sbatch_cache_shifts[0] + write_pos_chn_shift] = key_act[read_pos_chn_shift];
        h_cache_data[sbatch_cache_shifts[1] + write_pos_chn_shift] = value_act[read_pos_chn_shift];
    }

}

template<int grid_bound>
__global__ void copyBatchPrefillCacheDataKernel(void* cache_data, size_t* write_shifts, __half* key_act, __half* value_act, int* key_starts, uint8_t* bids, int dynamic_l, int kv_heads, size_t block_boundary) {

    int tid = threadIdx.x;
    const int channels = 128;
    __shared__ int step_id;
    __shared__ int example_len;
    __shared__ size_t sbatch_cache_shifts[2]; // cache example shift for key & value.

    __half* h_cache_data = reinterpret_cast<__half*>(cache_data);

    for (int blk_id=blockIdx.x; blk_id < block_boundary; blk_id+= grid_bound) {
        int pos_id = blk_id / kv_heads;
        int head_id = blk_id % kv_heads;

        if (tid == 0) {
            int batch_id = static_cast<int>(bids[pos_id]);
            int example_act_start = key_starts[batch_id];
            int example_act_end = key_starts[batch_id+1];
            step_id = pos_id - example_act_start;
            example_len = example_act_end - example_act_start;
            sbatch_cache_shifts[0] = write_shifts[2*batch_id];
            sbatch_cache_shifts[1] = write_shifts[2*batch_id+1];
        }
        __syncthreads();
        if (step_id >= 0 && step_id < example_len) {
            size_t read_pos_chn_shift = static_cast<size_t>(pos_id) * kv_heads * channels + head_id * channels + tid;
            size_t write_pos_chn_shift = static_cast<size_t>(step_id) * kv_heads * channels + head_id * channels + tid;

            h_cache_data[sbatch_cache_shifts[0] + write_pos_chn_shift] = key_act[read_pos_chn_shift];
            h_cache_data[sbatch_cache_shifts[1] + write_pos_chn_shift] = value_act[read_pos_chn_shift];            
        }

    }
}

// ==================================================
// cpu codes 
// ==================================================

// static std::shared_ptr<std::map<int, cublasHandle_t>> CublasHandleMap = nullptr; //负责cublas
std::map<int, cublasHandle_t> CublasHandleMap;

void setup_cublas_handler(int gpu_id) {
    // if (CublasHandleMap == nullptr) {
    //     // CublasHandleMap = std::make_shared<std::map<int, cublasHandle_t>>();
    // }
    SetDevice(gpu_id);

    cublasHandle_t handler = nullptr;
    auto stat = cublasCreate(&handler);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed:%d\n", stat);
        exit(0);
    } else {
        // (*(CublasHandleMap.get()))[gpu_id] = handler;
        CublasHandleMap[gpu_id] = handler;
        printf("device %i initialized cublas handler\n", gpu_id);
    }
}

cublasHandle_t get_cublas_handler(int gpu_id) {
    int id;
    if (gpu_id < 0) {
        int id = -1;
        printf("initializing default cublas handler\n");
        setup_cublas_handler(id);
    } else {
        id = gpu_id;
    }
    // return (CublasHandleMap.get())->find(id)->second;
    return CublasHandleMap.find(id)->second;
}

void print_cuda_info(int gpu_id) {
    if (gpu_id >=0) {
        SetDevice(gpu_id);
    }
    size_t mem_free;
    size_t mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);
    printf("### device (%i): mem_free=%dMB, mem_total=%dMB\n", gpu_id, mem_free/(1024*1024), mem_total/(1024*1024));
}

void SetDevice(int gpu_id) {
    if (gpu_id >= -1) {
        cudaSetDevice(gpu_id);
    }
}

void CheckGPUValues(liteqwen::DataType dtype, size_t numel, void* data) {
    cudaDeviceSynchronize();
    // int num_grid = (numel-1) / 256 + 1;
    if (dtype == liteqwen::DataType::FLOAT32) {
        print_invalid_float<<<1, 256>>>((float*)data, numel);
        cudaDeviceSynchronize();
    } else if (dtype == liteqwen::DataType::FLOAT16) {
        print_invalid_fp16<<<1, 256>>>((__half*)data, numel);
        cudaDeviceSynchronize();
    } else {
        printf("print only supports fp16, fp32 and int\n");
        return;
    }
}

void DeviceSynchronize() {
    cudaDeviceSynchronize();
}

// ---------- memory --------
struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    CudaMemoryBuffer () {}

    CudaMemoryBuffer (void *data, size_t size, bool busy) :
            data(data), size(size), busy(busy) {}
};
std::map<int, std::vector <CudaMemoryBuffer>> cudaBuffersMap;
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector <CudaMemoryBuffer>> bigBuffersMap;

void * CudaDirectMalloc(size_t size) {
    // printf("direct malloc, slow\n");
    void * ret;
    cudaError_t state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %d kB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
        return nullptr;
    }
    return ret;
}

void CudaDirectFree(void *ret) {
    // printf("direct free, slow\n");
    cudaError_t state = cudaFree(ret);
    checkCudaErrors("Error: CUDA error when release memory!", state);
}

void * ManagedCudaMalloc(int gpu_id, size_t size) {
    // 在当前设备进行malloc，需要外层进行SetDevice。
    // int id = -1;
    // cudaError_t state = cudaSuccess;
    // state = cudaGetDevice(&id);
    // checkCudaErrors("Error: CUDA error when find device!", state);

    int id = gpu_id;
    if (size > 1024 * 1024) {
        // >1m的显存分配，需要bigBuffers中寻找空闲的，并且空闲块的size与申请的size差距<1m。如果没找到则新申请。
        auto &bigBuffers = bigBuffersMap[id];
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy
                && bigBuffers[i].size - size < 1 * 1024 * 1024) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
            // printf("return big buffered malloc at %i, device=%i\n", selId, id);
            return bigBuffers[selId].data;
        }

        void * ret;
        // printf("risky malloc of size=%i, should not be using during running\n", (int)size);
        cudaError_t state = cudaMalloc(&ret, size);
        if (cudaSuccess != state) {
            printf("Error: CUDA error when allocating %d MB memory! maybe there's no enough memory left on device.", size >> 20);
            checkCudaErrors("", state);
            return nullptr;
        }
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, true));
        // printf("managed big direct malloc device=%i, slow\n", id);
        return ret;
    }

    // 对于<1m的分配需求，不要求size接近，只要空闲即可。noBusyCnt[gpu_id]的存储值表示当前small buffer list内所有块的剩余总可用显存size。
    auto &cudaBuffers = cudaBuffersMap[id];
    for (int i = 0; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            noBusyCnt[id] -= cudaBuffers[i].size;
            // printf("return buffered malloc at %i, device=%i\n", i, id);
            return cudaBuffers[i].data;
        }
    }
    void * ret;
    // printf("risky malloc of size=%i, should not be using during running\n", (int)size);
    cudaError_t state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %d KB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
        return nullptr;
    }
    cudaBuffers.push_back(CudaMemoryBuffer(ret, size, true));
    // printf("managed direct malloc device=%i, size=%i, slow\n", id, (int)size);
    return ret;
}


void ManagedCudaFree(void *ret, int gpu_id) {
    // 如果gpu_id>=0，则只释放给定gpu上的内容。需要注意给定的data指针ret必须位于给定的gpu_id，否则会出错。
    if (ret == nullptr) {
        return;
    }
    if (cudaBuffersMap.empty())
        return;
    cudaError_t state = cudaSuccess;

    // *ret指向的显存都会被释放（物理或逻辑释放），但释放前有一些固定操作：
    // 1. 当小buffer块总显存过多时，循环当前gpu的小buffer块，清理不被占用的。并重置noBusyCnt（noBusyCnt[gpu_id]的存储值表示当前small buffer list内所有块的剩余总可用显存size）
    // 忘记被释放的小buffer可能会堆积，noBusyCnt重置后无法纳入监控，成为碎片。
    for (auto &it: cudaBuffersMap) {
        if (gpu_id >= 0 && gpu_id != it.first) {
            continue;
        }

        if (noBusyCnt[it.first] > 1024 * 1024 * 1024) {
            auto &cudaBuffers = it.second;
            std::vector <CudaMemoryBuffer> temp;
            for (int i = 0; i < cudaBuffers.size(); i++) {
                if (!cudaBuffers[i].busy) {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cudaBuffers[i].data);
                    if (cudaSuccess != state)
                        printf("Error: CUDA error when release memory on device %d!", it.first);
                    checkCudaErrors("", state);
                } else {
                    temp.push_back(cudaBuffers[i]);
                }
            }
            cudaBuffers.clear();
            it.second = temp;
            noBusyCnt[it.first] = 0;
        }
    }

    // 2. 小buffer块内剩余被占用的块，如果匹配到当前ret，则逻辑释放，noBusyCnt增加可用空间。
    // 3. big buffer内剩余被占用的块，如果匹配到当前ret，则逻辑释放。
    for (auto &it: cudaBuffersMap) {
        if (gpu_id >= 0 && gpu_id != it.first) {
            continue;
        }

        auto &cudaBuffers = it.second;
        for (int i = 0; i < cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                noBusyCnt[it.first] += cudaBuffers[i].size;
                cudaBuffers[i].busy = false;
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
                // printf("logical free\n");
                return;
            }
        }
    }

    // 4. 任何没被前面两个buff释放的ret，被当作direct free，直接释放。
    // printf("direct free, slow\n");
    if (gpu_id >= 0) {
        SetDevice(gpu_id);
    }
    state = cudaFree(ret);
    checkCudaErrors("CUDA error when release memory!", state);
}

void* GetCudaMalloc(int gpu_id, size_t size, bool managed) {
    int id = -1;
    if (gpu_id >= 0) {
        SetDevice(gpu_id);
        id = gpu_id;
    } else {
        cudaError_t state = cudaSuccess;
        state = cudaGetDevice(&id);
        checkCudaErrors("Error: CUDA error when find device!", state);
        SetDevice(id);
    }

    void* ret;
    if (managed) {
        // printf("trying managed malloc, size=%i\n", (int)size);
        ret = ManagedCudaMalloc(id, size);
    } else {
        // printf("direct unmanaged malloc, size=%i, slow\n", (int)size);
        ret = CudaDirectMalloc(size);
    }
    return ret;
}

void* GetDtypeCudaMalloc(int gpu_id, liteqwen::DataType dtype, size_t numel, bool managed) {
    auto uint_info = liteqwen::GetUintInfo(dtype);
    int unitSize = uint_info.first;
    int unitSizeDiv = uint_info.second;

    size_t uint8_len = numel * unitSize / unitSizeDiv;
    size_t size = sizeof(uint8_t) * uint8_len;

    int id = -1;
    if (gpu_id >= 0) {
        SetDevice(gpu_id);
        id = gpu_id;
    } else {
        cudaError_t state = cudaSuccess;
        state = cudaGetDevice(&id);
        checkCudaErrors("Error: CUDA error when find device!", state);
        SetDevice(id);
    }
    void* res;
    if (managed) {
        res = ManagedCudaMalloc(id, size);
    } else {
        res = CudaDirectMalloc(size);
    }
    return res;
}

void CudaFree(void* gpu_arr, bool managed, int gpu_id) {
    if (gpu_arr == nullptr) {
        return;
    }
    int id = -1;
    if (gpu_id >= 0) {
        SetDevice(gpu_id);
        id = gpu_id;
    } else {
        cudaError_t state = cudaSuccess;
        state = cudaGetDevice(&id);
        checkCudaErrors("Error: CUDA error when find device!", state);
        SetDevice(id);
    }
    // printf("freeing:%p, managed=%i\n", gpu_arr, (int)managed);
    if (managed) {
        ManagedCudaFree(gpu_arr, id);
    } else {
        CudaDirectFree(gpu_arr);
    }
    return;
}

void ManagedCudaMallocBigBuffer(int gpu_id, size_t size) {
    int id = -1;
    if (gpu_id >= 0) {
        SetDevice(gpu_id);
        id = gpu_id;
    } else {
        cudaError_t state = cudaSuccess;
        state = cudaGetDevice(&id);
        checkCudaErrors("Error: CUDA error when find device!", state);
        SetDevice(id);
    }
    void * ret;
    auto &bigBuffers = bigBuffersMap[id];
    // printf("registering new big buffer, slow\n");
    cudaMalloc(&ret, size);
    cudaError_t state = cudaMalloc(&ret, size);
    if (cudaSuccess != state)
        printf("Error: CUDA error when allocating %d MB memory! maybe there's no enough memory left on device.", size >> 20);
    checkCudaErrors("", state);
    bigBuffers.push_back(CudaMemoryBuffer(ret, size, false));
}

void ManagedCudaClearBigBuffer(int data_group_start_gpu, int data_group_size) {
    // 只针对data group以内的cache进行清理，不影响其他data_id下的内容。
    if (bigBuffersMap.empty())
        return;
    cudaError_t state = cudaSuccess;
    for (auto &it : bigBuffersMap) {
        if (it.first < data_group_start_gpu || it.first >=(data_group_start_gpu+data_group_size)) {
            continue;
        }
        int id = it.first;
        cudaSetDevice(id);
        auto &bigBuffers = it.second;
        std::vector <CudaMemoryBuffer> temp;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                state = cudaSetDevice(it.first);
                state = cudaFree(bigBuffers[i].data);
                if (cudaSuccess != state)
                    printf("Error: CUDA error when release memory on device %d!", it.first);
                checkCudaErrors("", state);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
}

// ------copy/move-------------

void CopyGPUData(liteqwen::DataType dtype, void* dst, void* src, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length, bool use_kernel) {
    if (gpu_id >= -1) {
        SetDevice(gpu_id);
    }
    auto uint_info = liteqwen::GetUintInfo(dtype);
    int unitSize = uint_info.first;
    int unitSizeDiv = uint_info.second;

    size_t uint8_len = copy_length * unitSize / unitSizeDiv;
    size_t uint8_dst_offset = dst_offset * unitSize / unitSizeDiv;
    size_t uint8_src_offset = src_offset * unitSize / unitSizeDiv;

    if (use_kernel) {
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid((unsigned long)(ceil((float)(uint8_len) / BLOCK_SIZE)));

        size_t limit = max(uint8_dst_offset, uint8_src_offset) + uint8_len;
        liteqwen::IndexType index_type = liteqwen::get_index_type(limit, false);

        if (index_type == liteqwen::IndexType::UINT32_IDX) {
            CopyContiguousKernel<unsigned int><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)src, static_cast<unsigned int>(uint8_dst_offset), static_cast<unsigned int>(uint8_src_offset), static_cast<unsigned int>(uint8_len));
        } else if (index_type == liteqwen::IndexType::INT32_IDX) {
            CopyContiguousKernel<int><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)src, static_cast<int>(uint8_dst_offset), static_cast<int>(uint8_src_offset), static_cast<int>(uint8_len));
        } else if (index_type == liteqwen::IndexType::ULONG_IDX) {
            CopyContiguousKernel<size_t><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)src, uint8_dst_offset, uint8_src_offset, uint8_len);
        } else {
            CopyContiguousKernel<long><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)src, static_cast<long>(uint8_dst_offset), static_cast<long>(uint8_src_offset), static_cast<long>(uint8_len));
        }
        //cudaDeviceSynchronize();        
    } else {
        void* src_shifted = ((void*)src) + uint8_src_offset;
        void* dst_shifted = ((void*)dst) + uint8_dst_offset;
        cudaMemcpy(dst_shifted, src_shifted, uint8_len, cudaMemcpyDeviceToDevice);
    }
}

void CopyBetweenGPUs (liteqwen::DataType dtype, int dstId, void *dst, int srcId, void *src, size_t copy_length) {
    auto uint_info = GetUintInfo(dtype);
    int unitSize = uint_info.first;
    int unitSizeDiv = uint_info.second;

    size_t uint8_len = copy_length * unitSize / unitSizeDiv;
    size_t size = uint8_len * sizeof(uint8_t);

    int canPeerAccess = 0;
    cudaError_t state = cudaDeviceCanAccessPeer(&canPeerAccess, srcId, dstId);
    if (canPeerAccess) {
        state = cudaMemcpyPeer(dst, dstId, src, srcId, size);
    } else {
        uint8_t *cpuData = new uint8_t[size];
        state = cudaSetDevice(srcId);
        state = cudaMemcpy(cpuData, src, size, cudaMemcpyDeviceToHost);

        state = cudaSetDevice(dstId);
        state = cudaMemcpy(dst, cpuData, size, cudaMemcpyHostToDevice);
        delete[] cpuData;
    }
    checkCudaErrors("Error: CUDA error when copy Between GPUs!", state);
    //cudaDeviceSynchronize();
}

void UploadData(liteqwen::DataType dtype, void* dst, uint8_t* src, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length) {

    auto uint_info = GetUintInfo(dtype);
    int unitSize = uint_info.first;
    int unitSizeDiv = uint_info.second;

    long uint8_len = copy_length * unitSize / unitSizeDiv;
    size_t tmp_gpu_size = sizeof(uint8_t) * uint8_len;
    // printf("getcudamalloc %i, int8len=%i\n", tmp_gpu_size, uint8_len);
    SetDevice(gpu_id);
    void* tmp_gpu;
    tmp_gpu = GetCudaMalloc(gpu_id, tmp_gpu_size, false);
    // printf("copying %i\n", tmp_gpu_size);
    cudaMemcpy((uint8_t*)tmp_gpu, (uint8_t*)src, tmp_gpu_size, cudaMemcpyHostToDevice);

    int uint8_dst_offset = dst_offset * unitSize / unitSizeDiv;
    int uint8_src_offset = src_offset * unitSize / unitSizeDiv;

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((unsigned long)(ceil((float)(uint8_len) / BLOCK_SIZE)));
    // printf("printing dst_offset=%i, src_offset=%i, uint8_len=%i, tmp_gpu_size=%i, sizeofint=%i\n", uint8_dst_offset, uint8_src_offset, uint8_len, tmp_gpu_size, sizeof(int));

    size_t limit = max(uint8_dst_offset, uint8_src_offset) + uint8_len;
    liteqwen::IndexType index_type = liteqwen::get_index_type(limit, false);
    if (index_type == liteqwen::IndexType::UINT32_IDX) {
        CopyContiguousKernel<unsigned int><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)tmp_gpu, static_cast<unsigned int>(uint8_dst_offset), static_cast<unsigned int>(uint8_src_offset), static_cast<unsigned int>(uint8_len));
    } else if (index_type == liteqwen::IndexType::INT32_IDX) {
        CopyContiguousKernel<int><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)tmp_gpu, static_cast<int>(uint8_dst_offset), static_cast<int>(uint8_src_offset), static_cast<int>(uint8_len));
    } else if (index_type == liteqwen::IndexType::ULONG_IDX) {
        CopyContiguousKernel<size_t><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)tmp_gpu, uint8_dst_offset, uint8_src_offset, uint8_len);
    } else {
        CopyContiguousKernel<long><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)tmp_gpu, static_cast<long>(uint8_dst_offset), static_cast<long>(uint8_src_offset), static_cast<long>(uint8_len));
    }
    // CopyContiguousKernel<unsigned long><<<dimGrid, dimBlock>>>((uint8_t*)dst, (uint8_t*)tmp_gpu, uint8_dst_offset, uint8_src_offset, uint8_len);
    cudaDeviceSynchronize();
    CudaFree(tmp_gpu, false, -10);
}

void QuickUploadData(liteqwen::DataType dtype, void* dst, uint8_t* src, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length) {
    // no offset, no buffering cuda data.
    auto uint_info = GetUintInfo(dtype);
    int unitSize = uint_info.first;
    int unitSizeDiv = uint_info.second;

    long uint8_len = copy_length * unitSize / unitSizeDiv;
    size_t tmp_gpu_size = sizeof(uint8_t) * uint8_len;
    SetDevice(gpu_id);
    cudaMemcpy((uint8_t*)dst, (uint8_t*)src, tmp_gpu_size, cudaMemcpyHostToDevice);
}



void UploadInt32(void* dst, uint8_t* src, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length, size_t dst_maxlen, bool right_pad_zero) {
    // 针对input_ids的预分配dst，上一次upload时的uint8_t强转，可能导致本轮残留一些异常巨大的int数值，需要zero right padding覆盖掉，才能确保embedding kernel执行时不越界。
    liteqwen::DataType dtype = liteqwen::DataType::INT32;

    if (dst_maxlen < copy_length+dst_offset) {
        printf("copy offset(%lu)+length(%lu) > max_length(%lu) of dest tensor, please check length.\n", dst_offset, copy_length, dst_maxlen);
        throw("data error");
    }

    size_t tmp_gpu_size = sizeof(int) * copy_length;
    // printf("getcudamalloc %i, int8len=%i\n", tmp_gpu_size, uint8_len);
    SetDevice(gpu_id);
    void* tmp_gpu;
    tmp_gpu = GetCudaMalloc(gpu_id, tmp_gpu_size, false);
    // printf("copying %i\n", tmp_gpu_size);
    cudaMemcpy((int*)tmp_gpu, (int*)src, tmp_gpu_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    // printf("printing dst_offset=%i, src_offset=%i, uint8_len=%i, tmp_gpu_size=%i, sizeofint=%i\n", uint8_dst_offset, uint8_src_offset, uint8_len, tmp_gpu_size, sizeof(int));
    if (! right_pad_zero) {
        dim3 dimGrid((unsigned long)(ceil((float)(copy_length) / BLOCK_SIZE)));
        CopyContiguousInt32Kernel<<<dimGrid, dimBlock>>>((int*)dst, (int*)tmp_gpu, dst_offset, src_offset, copy_length);
    } else {
        dim3 dimGrid((unsigned long)(ceil((float)(dst_maxlen-dst_offset) / BLOCK_SIZE)));
        CopyContiguousInt32KernelRightPadZero<<<dimGrid, dimBlock>>>((int*)dst, (int*)tmp_gpu, dst_offset, src_offset, copy_length, dst_maxlen);
    }
    cudaDeviceSynchronize();
    CudaFree(tmp_gpu, false, -10);
}

void UploadCastFp32ToFp16Data(void* gpu_data, float* cpu_values, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length) {
    size_t flt_size = sizeof(float)*copy_length;
    SetDevice(gpu_id);
    void* tmp_gpu;
    tmp_gpu = GetCudaMalloc(gpu_id, flt_size, false);
    cudaMemcpy(tmp_gpu, cpu_values, flt_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((unsigned long)(ceil((float)(copy_length) / BLOCK_SIZE)));
    __half* gpu_half = reinterpret_cast<__half*>(gpu_data);
    float* gpu_float = reinterpret_cast<float*>(tmp_gpu);
    CopyCastFp32ToFp16ContiguousKernel<<<dimGrid, dimBlock>>>(gpu_half, gpu_float, dst_offset, src_offset, copy_length);
    cudaDeviceSynchronize();
    CudaFree(tmp_gpu, false, -10);
}

void ConstantFill(void* dst_data, liteqwen::DataType data_type,  size_t numel, double value) {
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((unsigned long)(ceil((float)(numel) / BLOCK_SIZE)));
    if (data_type == liteqwen::DataType::FLOAT16) {
        constValueKernel<__half><<<dimGrid, dimBlock>>>((__half*)dst_data, numel, __float2half(static_cast<float>(value)));
    } else if (data_type == liteqwen::DataType::FLOAT32) {
        constValueKernel<float><<<dimGrid, dimBlock>>>((float*)dst_data, numel, static_cast<float>(value));
    } else if (data_type == liteqwen::DataType::INT32) {
        constValueKernel<int><<<dimGrid, dimBlock>>>((int*)dst_data, numel, static_cast<int>(value));
    } else {
        printf("not supported fill dtype.\n");
        throw("");
    }
}

void FillArange(void* data, int limit) {
    int* casted = reinterpret_cast<int*>(data);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)(ceil((float)(limit) / BLOCK_SIZE)));
    FillArangeKernel<<<dimGrid, dimBlock>>>(casted, limit);
}

void GpuCastFp16ToFp32(void* dst_data, void* src_data, int gpu_id, size_t dst_offset, size_t src_offset, size_t copy_length) {
    SetDevice(gpu_id);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((unsigned long)(ceil((float)(copy_length) / BLOCK_SIZE)));
    __half* gpu_half = reinterpret_cast<__half*>(src_data);
    float* gpu_float = reinterpret_cast<float*>(dst_data);
    CopyCastFp16ToFp32ContiguousKernel<<<dimGrid, dimBlock>>>(gpu_float, gpu_half, dst_offset, src_offset, copy_length);
}

void DownloadData(liteqwen::DataType dtype, uint8_t* dst, void* src, size_t dst_offset, size_t src_offset, size_t copy_length) {
    auto uint_info = GetUintInfo(dtype);
    int unitSize = uint_info.first;
    int unitSizeDiv = uint_info.second;

    size_t uint8_len = copy_length * unitSize / unitSizeDiv;
    size_t tmp_gpu_size = sizeof(uint8_t) * uint8_len;
    size_t uint8_dst_offset = dst_offset * unitSize / unitSizeDiv;
    size_t uint8_src_offset = src_offset * unitSize / unitSizeDiv;

    uint8_t* tmp_cpu = new uint8_t[uint8_len];
    // printf("download copying uint8 size=%i\n", (int)tmp_gpu_size);
    uint8_t* casted_src = reinterpret_cast<uint8_t*>(src);
    cudaMemcpy(tmp_cpu, (casted_src+uint8_src_offset), tmp_gpu_size, cudaMemcpyDeviceToHost);
    DeviceSynchronize();
    // printf("printing dst_offset=%i, src_offset=%i, uint8_len=%i, tmp_gpu_size=%i\n", uint8_dst_offset, uint8_src_offset, uint8_len, tmp_gpu_size);
    uint8_t* casted = reinterpret_cast<uint8_t*>(dst);
    for (int i=0; i<uint8_len; i++) {
        casted[i] = tmp_cpu[i];
    }
    delete[] tmp_cpu;
}


void PrintRow(std::string row_info, liteqwen::DataType dtype, void* data, size_t row_id, int cols, int print_width) {
    cudaDeviceSynchronize();
    // printf("row_tensor_addr=%p, offset=%i\n", data, row_id*cols);
    bool no_tail = (print_width == 0);
    if (no_tail) {
        print_width=1;
    }

    printf((row_info + std::string("|")).c_str());
    if (dtype == liteqwen::DataType::FLOAT32) {
        print_fp32_data<<<1, 1>>>((float*)data, row_id*cols, cols, print_width, no_tail);
        cudaDeviceSynchronize();
    } else if (dtype == liteqwen::DataType::FLOAT16) {
        print_fp16_data<<<1, 1>>>((__half*)data, row_id*cols, cols, print_width, no_tail);
        cudaDeviceSynchronize();
    } else if (dtype == liteqwen::DataType::INT32) {
        print_int_data<<<1, 1>>>((int*)data, row_id*cols, cols, print_width, no_tail);
        cudaDeviceSynchronize();        
    } else {
        printf("print only supports fp16, fp32 and int\n");
        return;
    }
}

void CPUConvertFp16ToFp32(float* out, void* in, liteqwen::DataType dtype, size_t numel) {
    __half* in_data = (__half*)in;
    for (size_t i=0; i<numel; i++) {
        out[i] = __half2float(in_data[i]);
    }
}

// ========================
// TESTING CODES
// ========================


template<typename DT, typename IT, int block_size, int max_block_size>
__global__ void __launch_bounds__(
    DynamicKernel<DT, IT, block_size, max_block_size>::maxThreadsPerBlock,
    DynamicKernel<DT, IT, block_size, max_block_size>::kMinBlocksPerSm)
launch_dynamic_check(typename DynamicKernel<DT, IT, block_size, max_block_size>::Params p) {
// #ifdef __CUDA_ARCH__
    if (!p.advance_to_example()) {
        return;
    }
    DynamicKernel<DT, IT, block_size, max_block_size>::check_block_start(p);
    return;
// #endif
}


void dynamic_check_launch(const liteqwen::Data& x, const liteqwen::Data& batch_ids, const liteqwen::Data& start_positions, int dynamic_bsz, size_t dynamic_boundary, int head_num, int channel) {
    // bool kernel_launched = false;

    // auto launchKernel = [&](auto _k, auto kernel_fn) {
    //     using Kernel = decltype(_k);
    //     using scalar_t = typename Kernel::scalar_t;
    //     (void)_k;

    //     if (kernel_launched) {
    //     	return;
    //     }
    //     kernel_launched = true;

    //     typename Kernel::Params p;
    //     p.data_ptr = (__half*)x.cudaData;
    //     p.data_bids_ptr = (int8_t*)batch_ids.cudaData;
    //     p.seqstart_ptr = (int*)start_positions.cudaData;
    //     p.set_dynamic_bh(head_num, channel);

    //     dim3 dimBlock(p.channel);
    //     // dim3 dimGrid = p.get_dimGrid_blh(dynamic_boundary, head_num);
    //     dim3 dimGrid = p.get_dimGrid_bh(dynamic_bsz, head_num);
    //     kernel_fn<<<dimGrid, dimBlock>>>(p);
    // };


    // size_t limit = dynamic_boundary*head_num*channel;
    // liteqwen::IndexType index_type = liteqwen::get_index_type(limit, false);

    // const int block_size = 128;
    
    // if (index_type == liteqwen::IndexType::UINT32_IDX) {
    //     dispatch_kernel<__half, unsigned int, block_size, block_size>(launchKernel);
    // } else if (index_type == liteqwen::IndexType::INT32_IDX) {
    //     dispatch_kernel<__half, int, block_size, block_size>(launchKernel);
    // } else if (index_type == liteqwen::IndexType::ULONG_IDX) {
    //     dispatch_kernel<__half, size_t, block_size, block_size>(launchKernel);
    // } else {
    //     dispatch_kernel<__half, long, block_size, block_size>(launchKernel);
    // }

    const int block_size = 128;
    const int max_block_size = 128;
    DynamicKernel<__half, size_t, block_size, max_block_size>::Params p;
    p.data_ptr = (__half*)x.cudaData;
    p.data_bids_ptr = (int8_t*)batch_ids.cudaData;
    p.seqstart_ptr = (int*)start_positions.cudaData;
    p.set_dynamic_bh(head_num, channel);
    dim3 dimBlock(p.channel);
    dim3 dimGrid = p.get_dimGrid_bh(dynamic_bsz, head_num);
    launch_dynamic_check<__half, size_t, block_size, max_block_size><<<dimGrid, dimBlock>>>(p);
}

__global__ void check_batch_cache_read_kernel(void** pointers, int channel, int* seq_starts) {
    int batch_id = blockIdx.x;
    __half* key_data_start = reinterpret_cast<__half*>(pointers[batch_id*2]); // batch_id的t=0的key-cache指针。time stride = H * D
    __half* value_data_start = reinterpret_cast<__half*>(pointers[batch_id*2+1]); // batch_id的t=0的val-cache指针。time stride = H * D
    int data_len = seq_starts[batch_id + 1] - seq_starts[batch_id];
    if (threadIdx.x < 3 || threadIdx.x > (channel-3)) {
        for (int t=0; t<3; t++) {
            int chn_offset = t * channel;
            printf("bid=%i, t=%i, chn_id=%i, data_len=%i, key=%f, val=%f\n", batch_id, t, threadIdx.x, data_len, __half2float(*(key_data_start+chn_offset+threadIdx.x)), __half2float(*(value_data_start+chn_offset+threadIdx.x)));
        }
        for (int t2=data_len-4; t2<data_len; t2++) {
            int chn_offset2 = t2 * channel;
            printf("bid=%i, t=%i, chn_id=%i, data_len=%i, key=%f, val=%f\n", batch_id, t2, threadIdx.x, data_len, __half2float(*(key_data_start+chn_offset2+threadIdx.x)), __half2float(*(value_data_start+chn_offset2+threadIdx.x)));
        }
    }
}

void check_cache_read(const liteqwen::Data& kv_pointers, int dynamic_bsz, const liteqwen::Data& seq_starts, int channel) {
    dim3 dimGrid(dynamic_bsz);
    dim3 dimBlock(channel);
    check_batch_cache_read_kernel<<<dimGrid, dimBlock>>>((void**)kv_pointers.cudaData, channel, (int*)seq_starts.cudaData);
}

template<size_t window>
__global__ void copyPrefillCacheDataKernel(__half* cache_data, __half* activation_data, size_t layer_offset, int* pos_starts, int batch_id, int cache_channels, size_t boundary) {
    __shared__ __half* layer_start;
    __shared__ __half* activation_start;
    size_t thread_id = threadIdx.x;
    if (thread_id == 0) {
        layer_start = cache_data + layer_offset;
        size_t read_offset = static_cast<size_t>(pos_starts[batch_id]) * cache_channels;
        activation_start = activation_data + read_offset;
    }
    __syncthreads();

    for (size_t base=0; base < boundary; base+= window) {
        size_t tid = blockIdx.x * blockDim.x + thread_id + base;
        if (tid < boundary) {
            layer_start[tid] = activation_start[tid];
        }
    }
}

__global__ void copyDecodeCacheDataKernel(__half* cache_data, __half* activation_data, size_t layer_offset, int* pos_starts, int batch_id, int cache_channels, size_t boundary) {
    __shared__ __half* example_last;
    __shared__ __half* activation_start;
    int thread_id = threadIdx.x;
    if (thread_id == 0) {
        size_t write_offset = static_cast<size_t>(pos_starts[batch_id+1]-1) * cache_channels; // last step position offset of example
        example_last = cache_data + layer_offset + write_offset;
        activation_start = activation_data + batch_id * cache_channels; // each example only has 1 step in batch activation.
    }
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + thread_id;
    if (tid < boundary) {
        example_last[tid] = activation_start[tid];
    }
}

void WriteKVCacheFromBatch(bool is_prefill, void* block_cache_data, void* activation_data, int gpu_id, size_t layer_data_offset, const liteqwen::Data& pos_starts, int bi, int cache_channel, int example_len) {
    // cache dtype is fp16
    SetDevice(gpu_id);
    const int grid_bound = 12288;
    if (is_prefill) {
        size_t boundary = static_cast<size_t>(example_len) * cache_channel;
        dim3 dimBlock(BLOCK_SIZE);
        size_t grid_size = (boundary - 1) / BLOCK_SIZE + 1;
        const size_t window = static_cast<size_t>(grid_bound) * BLOCK_SIZE;
        if (grid_size > grid_bound) {
            dim3 dimGrid(grid_bound);
            copyPrefillCacheDataKernel<window><<<dimGrid, dimBlock>>>((__half*)block_cache_data, (__half*)activation_data, layer_data_offset, (int*)pos_starts.cudaData, bi, cache_channel, boundary);
        } else {
            dim3 dimGrid(grid_size);
            copyPrefillCacheDataKernel<window><<<dimGrid, dimBlock>>>((__half*)block_cache_data, (__half*)activation_data, layer_data_offset, (int*)pos_starts.cudaData, bi, cache_channel, boundary);
        }
    } else {
        int boundary = example_len * cache_channel;
        dim3 dimGrid((int)(ceil((float)(boundary) / BLOCK_SIZE)));
        dim3 dimBlock(BLOCK_SIZE);
        copyDecodeCacheDataKernel<<<dimGrid, dimBlock>>>((__half*)block_cache_data, (__half*)activation_data, layer_data_offset, (int*)pos_starts.cudaData, bi, cache_channel, boundary);
    }
    
}

void MoveGPUKVPtrs(const liteqwen::Data& kv_ptrs, void* cache_data, const liteqwen::Data& gpu_offsets, int bsz) {
    int ptr_boundary = bsz*2;
    dim3 dimGrid((ptr_boundary - 1) / 32 + 1);
    dim3 dimBlock(32);
    size_t* shifts = (size_t*)gpu_offsets.cudaData;
    void** pointers = (void**)kv_ptrs.cudaData;
    move_ptrs_according_to_shifts<<<dimGrid, dimBlock>>>(pointers, (__half*)cache_data, shifts, ptr_boundary);
}

void WriteGPUKV(bool is_prefill, void* cache_data, const liteqwen::Data& gpu_offsets, std::pair<liteqwen::Data*, liteqwen::Data*> kv_pair, const liteqwen::Data& kstarts, const liteqwen::Data& batch_ids, int dynamic_l, int kv_heads) {
    // channel = 128 has been asserted.
    __half* key_act = (__half*)(kv_pair.first->cudaData);
    __half* value_act = (__half*)(kv_pair.second->cudaData);    
    if (is_prefill) {

        const int grid_bound = 12288;
        size_t block_boundary = static_cast<size_t>(dynamic_l)*kv_heads;
        dim3 dimBlock(128);
        if (block_boundary > grid_bound) {
            dim3 dimGrid(12288);
            copyBatchPrefillCacheDataKernel<grid_bound><<<dimGrid, dimBlock>>>(cache_data, (size_t*)gpu_offsets.cudaData, key_act, value_act, (int*)kstarts.cudaData, (uint8_t*)batch_ids.cudaData, dynamic_l, kv_heads, block_boundary);
        } else {
            dim3 dimGrid(dynamic_l*kv_heads);
            copyBatchPrefillCacheDataKernel<grid_bound><<<dimGrid, dimBlock>>>(cache_data, (size_t*)gpu_offsets.cudaData, key_act, value_act, (int*)kstarts.cudaData, (uint8_t*)batch_ids.cudaData, dynamic_l, kv_heads, block_boundary);
        } 
    } else {
        dim3 dimGrid(dynamic_l*kv_heads);
        dim3 dimBlock(128);
        copyBatchDecodeCacheDataKernel<<<dimGrid, dimBlock>>>(cache_data, (unsigned long*)gpu_offsets.cudaData, key_act, value_act, (int*)kstarts.cudaData, dynamic_l, kv_heads);
    }
}

void PrintWithShift(void* data, size_t layer_shift, int end_step, int channels) {
    DeviceSynchronize();
    size_t numel = static_cast<size_t>(end_step) * channels;
    print_with_shift<<<1, 1>>>((__half*)data, layer_shift, numel, static_cast<size_t>(channels), 3, 5);
}
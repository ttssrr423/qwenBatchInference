#include "sampling.cuh"
#include "core_gpu.cuh"

static std::shared_ptr<std::map<int, curandState*>> HandleRandStateMap = nullptr;

__global__ void setup_rand_kernel(curandState* state, int* seeds, int boundary)
{
    unsigned long long id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < boundary) {
        // sequence start = id * 32768, sequence_start + offset = sampling index from length 2^130. Every distinct seed produces such a sequence.
        unsigned long long offset = 0;
        curand_init(static_cast<unsigned long long>(seeds[id]), id*32768, offset, &state[id]);
    }
}

__global__ void invalidFilterWithTemperatureKerkenl(float* logits_out, __half* logits, float* temperatures, int vocab_size, int boundary, int eos_id) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_id = id / vocab_size;
    if (id<boundary) {
        __half original_val = logits[id];
        float temp = temperatures[batch_id];
        if (!(__hisnan(original_val) || __hisinf(original_val))) {
            float new_lgt = __half2float(original_val) / temp;
            // if (!(__hisnan(new_lgt) || __hisinf(new_lgt))) {
            //     if (eos_id>=0 && id==eos_id) {
            //         logits[id] = __float2half(-4096.0f);
            //     } else {
            //         logits[id] = new_lgt;
            //     }
            // } else {
            //     // printf("valid%f|", __half2float(new_lgt));
            //     logits[id] = __float2half(-4096.0f);
            // }
            logits_out[id] = new_lgt;
        } else {
            // printf("invalid%f|", __half2float(original_val));
            // 原始logits溢出，非eos置零，eos直接最大。
            if (id != eos_id) {
                logits_out[id] = -65536.0f;
            } else {
                logits_out[id] = 65536.0f;
            }
        }
    }
}

__device__  void insert_value(float* array, int* indices, int k, float data, int pos)
{
    for(int i=0; i<k; i++)
    {
        if(array[i] == data)
        {
            return;
        }
    }
    if(data < array[k-1])
    {
        return;
    }
    //19, 18, 17, 16,.........4, 3, 2, 1, 0
    for(int i = k-2; i>=0; i--)
    {
        if(data > array[i])
        {
            array[i + 1] = array[i];
            indices[i+1] = indices[i];
        }
        else
        {
            array[i+1] = data;
            indices[i+1] = pos;
            return;
        }
    }
    
    array[0] = data;
    indices[0] = pos;
}

template<int grid_size_per_example, int block_size, int topk>
__global__ void gpu_topk_first_pass(float* input, float* output, int* out_indices, int length)
{
    __shared__ float ken[block_size * topk];
    __shared__ int kid[block_size * topk];
    float top_array[topk];
    int top_indices[topk];

    int batch_id = blockIdx.x / grid_size_per_example;
    int blkid = blockIdx.x % grid_size_per_example;

    for(int i = 0; i<topk; i++)
    {
        top_array[i] = INT_MIN;
        top_indices[i] = -1;
    }

    for(int idx = threadIdx.x + blockDim.x * blkid; idx < length; idx += grid_size_per_example * blockDim.x)
    {
        insert_value(top_array, top_indices, topk, input[idx + batch_id * length], idx);
    }
    for(int i =0; i<topk; i++)
    {
        ken[topk * threadIdx.x + i] = top_array[i];
        kid[topk * threadIdx.x + i] = top_indices[i];
    }
    __syncthreads();

    for(int i = block_size/2; i>=1; i/=2)
    {
        if(threadIdx.x < i)
        {
            for(int m=0; m<topk; m++)
            {
                int k_num = topk *(threadIdx.x + i) + m;
                insert_value(top_array, top_indices, topk, ken[k_num], kid[k_num]);
            }
        }
        __syncthreads();
        if(threadIdx.x < i)
        {
            for(int m=0; m<topk; m++)
            {
                ken[topk * threadIdx.x + m] = top_array[m];
                kid[topk * threadIdx.x + m] = top_indices[m];
            }
        }
        __syncthreads();
    }
    // if(blockIdx.x * blockDim.x < length)
    if (blkid * blockDim.x < length)
    {
        if(threadIdx.x == 0 )
        {
            int batch_shift = batch_id * topk * grid_size_per_example;
            for(int i =0; i < topk; i++)
            {
                output[batch_shift + topk * blkid + i] = ken[i];
                out_indices[batch_shift + topk * blkid + i] = kid[i];
            }
        }
    }
}

template<int block_size, int topk>
__global__ void gpu_topk(float*input, int* input_idx, float*output, int* out_indices, int length, int k)
{
    // <<<bsz, block_size>>>
    // input.shape = [bsz, topk, block_size], length=topk*block_size
    __shared__ float ken[block_size * topk];
    __shared__ int kid[block_size * topk];
    float top_array[topk];
    int top_indices[topk];

    int batch_id = blockIdx.x;
    int blkid = 0;
    int batch_stride = batch_id * k * block_size;

    for(int i = 0; i<topk; i++)
    {
        top_array[i] = INT_MIN;
        top_indices[i] = -1;
    }

    for(int idx = threadIdx.x + blockDim.x * blkid; idx < length; idx += 1 * blockDim.x)
    {
        insert_value(top_array, top_indices, topk, input[batch_stride + idx], input_idx[batch_stride + idx]);
    }
    for(int i =0; i<topk; i++)
    {
        ken[topk * threadIdx.x + i] = top_array[i];
        kid[topk * threadIdx.x + i] = top_indices[i];
    }
    __syncthreads();

    for(int i = block_size/2; i>=1; i/=2)
    {
        if(threadIdx.x < i)
        {
            for(int m=0; m<topk; m++)
            {
                int k_num = topk *(threadIdx.x + i) + m;
                insert_value(top_array, top_indices, topk, ken[k_num], kid[k_num]);
            }
        }
        __syncthreads();
        if(threadIdx.x < i)
        {
            for(int m=0; m<topk; m++)
            {
                ken[topk * threadIdx.x + m] = top_array[m];
                kid[topk * threadIdx.x + m] = top_indices[m];
            }
        }
        __syncthreads();
    }
    if(blkid * blockDim.x < length)
    {
        if(threadIdx.x == 0 )
        {
            int example_shift = topk * batch_id;
            for(int i =0; i < topk; i++)
            {
                if (i < k) {
                    output[example_shift + topk * blkid + i] = ken[i];
                } else {
                    output[example_shift + topk * blkid + i] = -65536.0f; // __float2half(-4096.0f); // mask for probability
                }
                out_indices[example_shift + topk * blkid + i] = kid[i];
            }
        }
    }
}

__global__ void SampleKernel(int* sampled_id, __half* probs, int* vocab_indices, int cols, curandState* states, float* top_p, int topk) {
    int batch_id = blockIdx.x;
    curandState localState = states[batch_id];
    float x = curand_uniform(&localState);
    __half hx = __float2half(x * top_p[batch_id]);
    states[batch_id] = localState;

    __half sample_acc = 0.0;
    __half threshold = __float2half(1.0f);
    for (int _k=0; _k<cols; _k++) {
        sample_acc += probs[batch_id * topk + _k];
        if (sample_acc > hx && sample_acc <= threshold) {
            // printf("sampled item: %i, %i\n", _k, vocab_indices[_k]);
            sampled_id[batch_id] = vocab_indices[batch_id * topk + _k];
            sample_acc += __float2half(10.0f);
        }
    }
}

__global__ void SampleKernelFp32(int* sampled_id, float* probs, int* vocab_indices, int cols, curandState* states, float* top_p, int topk) {
    int batch_id = blockIdx.x;
    curandState localState = states[batch_id];
    float x = curand_uniform(&localState);
    float hx = x * top_p[batch_id];
    states[batch_id] = localState;

    float sample_acc = 0.0;
    float threshold = 1.0f;
    for (int _k=0; _k<cols; _k++) {
        sample_acc += probs[batch_id * topk + _k];
        if (sample_acc > hx && sample_acc <= threshold) {
            // printf("sampled item: %i, %i\n", _k, vocab_indices[_k]);
            sampled_id[batch_id] = vocab_indices[batch_id * topk + _k];
            sample_acc += 10.0f;
        }
    }
}

template <int powers>
__global__ void SoftmaxKernelAct16(__half* out, __half* in, int rows, int padded_cols, int cols, int folds)
{
    __shared__ __half sdata[powers];
    __shared__ __half maxV;

    int tid = threadIdx.x;
    int row_id = blockIdx.x;

    __half max_val = __float2half(-4096.0);

    for (int fi=folds-1; fi>=0; fi-=1) {
        int block_pos = tid + (row_id * folds + fi) * blockDim.x;
        int block_col = block_pos % padded_cols;
        int source_pos;
        if (block_col < cols)
        {
            source_pos = (block_pos / padded_cols) * cols + block_col;
        }
        else
        {
            source_pos = -1;
        }
        // printf("row_i=%d, fold_i=%d, abs_pos=%d(%d,%d)<-%d(%d,%d)\n", row_id, fi, block_pos, threadIdx.x, blockIdx.x, source_pos, inp_col, inp_row); //rm
        if (source_pos >= 0) {
            // max_val = __hmax(max_val, in[source_pos]);
            __half in_source_val = in[source_pos];
            if (max_val < in_source_val) {
                max_val = in_source_val;
            }
        }
    }
    
    sdata[tid] = max_val;
    __syncthreads();

    for (int pow=powers/2; pow>0; pow>>=1){
        if (tid < pow){
            // sdata[tid] = __hmax(sdata[tid], sdata[tid+pow]);
            if (sdata[tid] < sdata[tid+pow]) {
                sdata[tid] = sdata[tid+pow];
            }
        }
        __syncthreads();
    }

    if (tid==0){
        maxV = sdata[0];
        // printf("max sdata0=%f\n", __half2float(sdata[0]));
    }
    __syncthreads();

    __half sum = __float2half(0.0f);
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
            __half shifted = hexp(in[source_pos]-maxV);
            out[source_pos] = shifted;
            sum = sum + shifted;
        }
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int pow=powers/2; pow>0; pow>>=1){
        if (tid < pow){
            sdata[tid] = sdata[tid] + sdata[tid+pow];
        }
        __syncthreads();
    }

    // div by 0 processing
    bool using_epsilon = false;
    if (tid == 0) {
        __half epsilon = 1e-4;
        if (__habs(sdata[0]) < epsilon){
            // printf("replacing sum as epsilon ");
            sdata[0] = __float2half((float)cols) * epsilon;
            using_epsilon = true;
        }
        // printf("sum sdata0=%f\n", __half2float(sdata[0]));
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
            if (using_epsilon) {
                __half epsilon = 1e-4;
                out[source_pos] = (out[source_pos] + epsilon) / sdata[0];
            } else {
                out[source_pos] /= sdata[0];
            }
        }
    }

    __syncthreads();
}


template <int powers>
__global__ void SoftmaxKernelAct32(float* out, float* in, int rows, int padded_cols, int cols, int folds)
{
    __shared__ float sdata[powers];
    __shared__ float maxV;

    int tid = threadIdx.x;
    int row_id = blockIdx.x;

    float max_val = -65536.0f;

    for (int fi=folds-1; fi>=0; fi-=1) {
        int block_pos = tid + (row_id * folds + fi) * blockDim.x;
        int block_col = block_pos % padded_cols;
        int source_pos;
        if (block_col < cols)
        {
            source_pos = (block_pos / padded_cols) * cols + block_col;
        }
        else
        {
            source_pos = -1;
        }
        // printf("row_i=%d, fold_i=%d, abs_pos=%d(%d,%d)<-%d(%d,%d)\n", row_id, fi, block_pos, threadIdx.x, blockIdx.x, source_pos, inp_col, inp_row); //rm
        if (source_pos >= 0) {
            // max_val = __hmax(max_val, in[source_pos]);
            float in_source_val = in[source_pos];
            if (max_val < in_source_val) {
                max_val = in_source_val;
            }
        }
    }
    
    sdata[tid] = max_val;
    __syncthreads();

    for (int pow=powers/2; pow>0; pow>>=1){
        if (tid < pow){
            // sdata[tid] = __hmax(sdata[tid], sdata[tid+pow]);
            if (sdata[tid] < sdata[tid+pow]) {
                sdata[tid] = sdata[tid+pow];
            }
        }
        __syncthreads();
    }

    if (tid==0){
        maxV = sdata[0];
        // printf("max sdata0=%f\n", __half2float(sdata[0]));
    }
    __syncthreads();

    float sum = 0.0f;
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
            float shifted = exp(in[source_pos]-maxV);
            out[source_pos] = shifted;
            sum = sum + shifted;
        }
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int pow=powers/2; pow>0; pow>>=1){
        if (tid < pow){
            sdata[tid] = sdata[tid] + sdata[tid+pow];
        }
        __syncthreads();
    }

    // div by 0 processing
    bool using_epsilon = false;
    if (tid == 0) {
        float epsilon = 1e-8;
        if (abs(sdata[0]) < epsilon){
            // printf("replacing sum as epsilon ");
            sdata[0] = (float)cols * epsilon;
            using_epsilon = true;
        }
        // printf("sum sdata0=%f\n", __half2float(sdata[0]));
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
            if (using_epsilon) {
                float epsilon = 1e-8;
                out[source_pos] = (out[source_pos] + epsilon) / sdata[0];
            } else {
                out[source_pos] /= sdata[0];
            }
        }
    }

    __syncthreads();
}

curandState* get_gpu_curand_state(int gpu_id, int world_size, int handle_id) {
    if (gpu_id >= -1) {
        SetDevice(gpu_id);
    }
    curandState* state;
    int rand_key = handle_id * world_size + gpu_id;
    auto rand_state_kv = HandleRandStateMap->find(rand_key);
    if (rand_state_kv != HandleRandStateMap->end()) {
        state = rand_state_kv->second;
    } else {
        printf("error: cuda random not initialized\n");
        throw("cuda_error");
    }
    return state;
}

void gpu_curand_init(int gpu_id, int world_size, int data_size, int handle_id, const liteqwen::Data& seeds, int* cpu_seeds) {
    if (gpu_id >= -1) {
        SetDevice(gpu_id);
    }

    if (HandleRandStateMap == nullptr) {
        HandleRandStateMap = std::make_shared<std::map<int, curandState*>>();
    }

    curandState* state;
    int rand_key = handle_id * world_size + gpu_id;
    int grid_num = (int)(ceil((float)(data_size) / BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(grid_num);
    int* seed_gpu = (int*)(seeds.cudaData);

    size_t seed_size = data_size * sizeof(int) / sizeof(uint8_t);
    cudaMemcpy((uint8_t*)seeds.cudaData, (uint8_t*)cpu_seeds, seed_size, cudaMemcpyHostToDevice);

    auto rand_state_kv = HandleRandStateMap->find(rand_key);
    if (rand_state_kv != HandleRandStateMap->end()) {
        state = rand_state_kv->second;
        // printf("using existing curand state with key=%i\n", rand_key);
        setup_rand_kernel<<<dimGrid, dimBlock>>>(state, seed_gpu, data_size);
    } else {
        if (data_size > 1) {
            printf("new curand initializing with key=%i, curandSize=%ix%i\n", rand_key, grid_num * BLOCK_SIZE, sizeof(curandState));
            size_t size = grid_num * BLOCK_SIZE * sizeof(curandState);
            cudaMalloc(&state, size);
            setup_rand_kernel<<<dimGrid, dimBlock>>>(state, seed_gpu, data_size);
        } else {
            printf("new curand initializing with key=%i, curandSize=%i\n", rand_key, sizeof(curandState));
            size_t size = sizeof(curandState);
            cudaMalloc(&state, size);
            setup_rand_kernel<<<1, 1>>>(state, seed_gpu, data_size);
        }
        auto& state_map = *(HandleRandStateMap.get());
        state_map[rand_key] = state;
    }
    // cudaDeviceSynchronize();
}

void gpu_curand_init(int gpu_id, int world_size, int data_size, int handle_id, const liteqwen::Data& seeds) {
    curandState* state;
    int rand_key = handle_id * world_size + gpu_id;
    int grid_num = (int)(ceil((float)(data_size) / BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(grid_num);
    int* seed_gpu = (int*)(seeds.cudaData);

    auto rand_state_kv = HandleRandStateMap->find(rand_key);
    if (rand_state_kv != HandleRandStateMap->end()) {
        state = rand_state_kv->second;
        // printf("using existing curand state with key=%i\n", rand_key);
        setup_rand_kernel<<<dimGrid, dimBlock>>>(state, seed_gpu, data_size);
    } else {
        if (data_size > 1) {
            printf("new curand initializing with key=%i, curandSize=%ix%i\n", rand_key, grid_num * BLOCK_SIZE, sizeof(curandState));
            size_t size = grid_num * BLOCK_SIZE * sizeof(curandState);
            cudaMalloc(&state, size);
            setup_rand_kernel<<<dimGrid, dimBlock>>>(state, seed_gpu, data_size);
        } else {
            printf("new curand initializing with key=%i, curandSize=%i\n", rand_key, sizeof(curandState));
            size_t size = sizeof(curandState);
            cudaMalloc(&state, size);
            setup_rand_kernel<<<1, 1>>>(state, seed_gpu, data_size);
        }
        auto& state_map = *(HandleRandStateMap.get());
        state_map[rand_key] = state;
    }
}


void filterInvalidApplyTemperature(const liteqwen::Data& logitsFp32, const liteqwen::Data& logits, const liteqwen::Data& temperature_tensor, int vocab_size, int dynamic_bsz, int masking_eos_id) {

    __half* logits_data = (__half*)(logits.cudaData);
    float* logits_dataFp32 = (float*)(logitsFp32.cudaData);
    float* temp_data = (float*)(temperature_tensor.cudaData);
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((int)(ceil((float)(vocab_size * dynamic_bsz) / BLOCK_SIZE)));
    invalidFilterWithTemperatureKerkenl<<<dimGrid, dimBlock>>>(logits_dataFp32, logits_data, temp_data, vocab_size, dynamic_bsz*vocab_size, masking_eos_id);
}

void topk_sampling(const liteqwen::Data& out_id, int gpu_id, int world_size, int handle_id, const liteqwen::Data& logits, int channel, int top_k, const liteqwen::Data& top_p, const liteqwen::Data& _1_pass_result, const liteqwen::Data& _1_psss_indices, const liteqwen::Data& gpu_top_logits, const liteqwen::Data& gpu_top_indices, const liteqwen::Data& sample_softmax_out, int dynamic_bsz) {
    if (gpu_id >= -1) {
        cudaSetDevice(gpu_id);
    }
    if (top_k != 32) {
        printf("top k != 32 which is not default setup.\n");
    }

    float* logits_data = (float*)(logits.cudaData);
    int* sampled_id = (int*)(out_id.cudaData);

    const int grid_size_per_example = 32;
    int grid_size = grid_size_per_example * dynamic_bsz;
    const int block_size = 32;

    float* _1_pass_result_data = (float*)(_1_pass_result.cudaData);
    int* _1_pass_indices_data = (int*)(_1_psss_indices.cudaData);
    float* gpu_result_data = (float*)(gpu_top_logits.cudaData);
    int* gpu_indices_data = (int*)(gpu_top_indices.cudaData);
    float* softmax_out_data = (float*)(sample_softmax_out.cudaData);

    gpu_topk_first_pass<grid_size_per_example, block_size, 32><<<grid_size, block_size>>>(logits_data, _1_pass_result_data, _1_pass_indices_data, channel);

    gpu_topk<block_size, 32><<<dynamic_bsz, block_size>>>(_1_pass_result_data, _1_pass_indices_data, gpu_result_data, gpu_indices_data, top_k * grid_size_per_example, top_k);

    // rows=dynamic_bsz, padded_cols=32, cols=32, folds=1
    SoftmaxKernelAct32<32><<<dynamic_bsz, 32>>>(softmax_out_data, gpu_result_data, dynamic_bsz, 32, top_k, 1);

    curandState* sampling_state = get_gpu_curand_state(gpu_id, world_size, handle_id);
    float* top_p_data = (float*)top_p.cudaData;
    SampleKernelFp32<<<dynamic_bsz, 1>>>(sampled_id, softmax_out_data, gpu_indices_data, top_k, sampling_state, top_p_data, top_k);
}

liteqwen::BatchGeneratedRes download_sampled(const liteqwen::Data& sampled_id, int* cpu_sampled_id, int* eos_ids, int eos_num, int batch_size) {
    int* generated_id_data = (int*)(sampled_id.cudaData);
    cudaMemcpy(cpu_sampled_id, generated_id_data, sizeof(int)*batch_size, cudaMemcpyDeviceToHost);
    DeviceSynchronize();

    std::vector<int> ids;
    std::vector<bool> eoses;
    for (int bi=0; bi<batch_size; bi++) {
        int sampled_id_cpu = cpu_sampled_id[bi];
        bool is_eos = false;
        for (int i=0; i<eos_num; i++) {
            if (sampled_id_cpu==eos_ids[i]) {
                is_eos = true;
                break;
            }
        }
        if (is_eos) {
            ids.push_back(eos_ids[0]);
        } else {
            ids.push_back(sampled_id_cpu);
        }
        eoses.push_back(is_eos);
    }
    return liteqwen::BatchGeneratedRes{eoses, ids};
}

void batch_download_logits(int* top_batch_idx, float* top_batch_lgts, const liteqwen::Data& gpu_top_logits, const liteqwen::Data& gpu_top_indices, int dynamic_bsz, int top_k) {
    int* top_idx_data = (int*)gpu_top_indices.cudaData;
    float* top_lgts_data = (float*)gpu_top_logits.cudaData;
    cudaMemcpy(top_batch_idx, top_idx_data, sizeof(int)*dynamic_bsz*top_k, cudaMemcpyDeviceToHost);
    cudaMemcpy(top_batch_lgts, top_lgts_data, sizeof(int)*dynamic_bsz*top_k, cudaMemcpyDeviceToHost);
    DeviceSynchronize();
}
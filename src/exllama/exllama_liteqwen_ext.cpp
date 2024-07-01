#include "exllama_liteqwen_ext.h"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Device.h>
// Create Q4Matrix, return handle

// matmul_recons_thd = 8
// matmul_fused_remap = False
// matmul_no_half2 = False
ExLlamaTuning tuningParams = ExLlamaTuning{8, false, false}; // parameters fixed by auto-gptq.
// ExLlamaTuning tuningParams;
// void set_tuning_params
// (
//     int matmul_recons_thd,
//     bool matmul_fused_remap,
//     bool matmul_no_half2
// )
// {
//     tuningParams.matmul_recons_thd = matmul_recons_thd;
//     tuningParams.matmul_fused_remap = matmul_fused_remap;
//     tuningParams.matmul_no_half2 = matmul_no_half2;
// }




void prepare_buffers
(
    int device_index,
    const liteqwen::Data& temp_state,
    const liteqwen::Data& temp_dq
)
{
    
    // const at::cuda::OptionalCUDAGuard device_guard(device);
    const size_t max_int = std::numeric_limits<int>::max();

    prepare_buffers_cuda
    (
        device_index,
        // buffer size used for sanity checks
        std::clamp((size_t)temp_state.numel(), (size_t)0, max_int),
        (half*) temp_state.cudaData,
        (half*) temp_dq.cudaData
    );
}

__half* get_temp_dq(int gpu_id) {
    return get_temp_dq_ptr(gpu_id);
}

uintptr_t make_q4
(
    const liteqwen::Data& qweight,
    const liteqwen::Data& qzeros,
    const liteqwen::Data& scales,
    const liteqwen::Data& g_idx,
    int device
)
{
    // define pack_num := 32 / bits = 8
    // qweight: [in_feature / pack_num, out_feature], int32
    // qzeros: [ceil(in_feature/group_size), out_feature/pack_num], 每个zero值覆盖col上连续pack_num个权重，以及row上连续group_size个channel。相当于二维patch化。
    // scales: [ceil(in_feature/group_size), out_feature]，fp16, 每个out channel与input_group的对应位置W的元素w_ij的缩放scale。
    // TORCH_CHECK_SHAPES(qweight, 1, qzeros, 1, 8); 
    // TORCH_CHECK_SHAPES(scales, 1, qweight, 1, 1);
    // TORCH_CHECK_SHAPES(qzeros, 0, scales, 0, 1);

    int width = qweight.shape[1];
    int height = qweight.shape[0] * 8;
    int groups = qzeros.shape[0];

    bool is_null = (g_idx.cpuData == nullptr);

    // g_idx.const_print(std::string("g_idx"));
    Q4Matrix* m = new Q4Matrix
    (
        height,
        width,
        groups,

        (uint32_t*) qweight.cudaData,
        (uint32_t*) qzeros.cudaData,
        (half*) scales.cudaData,
        // g_idx.device().is_meta() ? NULL : (uint32_t*) g_idx.data_ptr(),
        g_idx.cpuData==nullptr ? NULL : (uint32_t*) g_idx.cpuData,

        device
    );

    g_q4_keep_matrix(m);
    
    return reinterpret_cast<uintptr_t> (m);
}

// Matmul half @ quant -> half

void q4_matmul
(
    const liteqwen::Data& x,
    uintptr_t w,
    const liteqwen::Data& out,
    cublasHandle_t handle
)
{
    SetDevice(x.gpu_id);
    Q4Matrix* wm = reinterpret_cast<Q4Matrix*> (w);


    // TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    // TORCH_CHECK(wm->height == x.size(-1), "x and w have incompatible shapes")

    // const at::cuda::OptionalCUDAGuard device_guard(at::device_of(x));
    // const at::cuda::OptionalCUDAGuard device_guard(static_cast<c10::DeviceIndex>(x.gpu_id));

    int dim_num = static_cast<int>(x.shape.size());
    int x_height;
    if (dim_num == 4) {
        x_height = x.numel() / (x.shape[dim_num-2] * x.shape[dim_num-1]);
    } else {
        x_height = x.numel() / x.shape[dim_num-1];
    }

    if (tuningParams.matmul_recons_thd == 0 || x_height < tuningParams.matmul_recons_thd)
    {
        q4_matmul_cuda
        (
            &tuningParams,
            (half*) x.cudaData,
            x_height,
            wm,
            (half*) out.cudaData
        );
    }
    else
    {
        q4_matmul_recons_cuda
        (
            &tuningParams,
            (half*) x.cudaData,
            x_height,
            wm,
            (half*) out.cudaData,
            // at::cuda::getCurrentCUDABlasHandle()
            handle
        );
    }
}
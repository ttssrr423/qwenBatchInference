#include "xformer_attention.h"
#include <cmath>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "autogen/cutlassF.h"
#include "kernel_forward.h"
// #include "pytorch_utils.h"

#include <ATen/cuda/CUDAConfig.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <mutex>
#include <deque>
#include <vector>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/macros/Macros.h>

#include <limits>

namespace at { namespace cuda {

DeviceIndex num_gpus = -1;
std::once_flag init_flag;
std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initCUDAContextVectors(int world_size) {
  num_gpus = world_size;
  if (device_flags.size() < world_size) {
    device_flags.resize(num_gpus);
    device_properties.resize(num_gpus);
  }
}

void initDeviceProperty(DeviceIndex device_index) {
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_index);
  device_properties[device_index] = device_prop;
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  // std::call_once(init_flag, initCUDAContextVectors, world_size);
  // std::call_once(device_flags[device], initDeviceProperty, device);
  // std::call_once(device_flags[0], initDeviceProperty, device);
  return &device_properties[device];
}
}}


void init_device_property_for_device(int world_size, int64_t device_start, int64_t device_end) {
	// std::call_once(at::cuda::init_flag, at::cuda::initCUDAContextVectors, world_size);
    at::cuda::initCUDAContextVectors(world_size);
	for (int64_t i = device_start; i<device_end; i++) {
        cudaSetDevice(i);
		// std::call_once(at::cuda::device_flags[i], at::cuda::initDeviceProperty, i);
        at::cuda::initDeviceProperty(i);
	}
    cudaSetDevice(device_start);
}


cudaStream_t defaultStream = nullptr;

void xformer_self_attention_fwd(const liteqwen::Data& attended_tmp_out, const liteqwen::Data& query, const liteqwen::Data& key, const liteqwen::Data& value, const liteqwen::Data& seqstart_q, const liteqwen::Data& seqstart_k, const liteqwen::Data& seqlen_k, const liteqwen::Data& flash_attn_workspace, int dynamic_bsz, int dynamic_length, int query_heads,  int max_q_len, int max_k_len, int channel) { // std::vector<int> view_shape_q, std::vector<int> view_shape_k
    // xformer forward self attention modified from xformer source code (path= ./attention_forward_generic.cu), the original code is not compiled but using as a comparison backup.
	// q, k, v shape = [b, seqlen, num_heads, K]
    //bias shape = [b, num_heads, seqlen, seqlen]; bias.has_value() should be false, unless using relative position bias such as in bert.
    
    // int64_t max_seqlen_q, max_seqlen_k; == T_max if static self attention, or max(input_lengths) if dynamic self_attention. may be different for decoding.
	int64_t max_seqlen_q = max_q_len;
	int64_t max_seqlen_k = 0; // set inside kernel during dynamic batch

    if (defaultStream == nullptr) {
        cudaError_t create_result = cudaStreamCreate(&defaultStream);
    }

    // prefill shape
    std::vector<int> query_shape{1, dynamic_length, query_heads, channel};
    int64_t chn_ = static_cast<int64_t>(channel);
    std::vector<int64_t> query_strides{dynamic_length*query_heads*chn_, query_heads*chn_, chn_, 1};
    std::vector<int> key_shape = query_shape;
    std::vector<int> value_shape = query_shape;
    std::vector<int64_t> key_strides = query_strides;
    std::vector<int64_t> value_strides = query_strides;

    int64_t B = 1;
    int64_t M = query_shape[1]; //query.shape[1];
    int64_t N = key_shape[1]; //key.shape[1];
    int64_t num_heads = query_heads;
    int64_t K = channel;
    int64_t Kv = channel;

    const bool use_dropout = false; // false for inference
	const bool compute_logsumexp = false; // true when needs_gradient
	const int64_t custom_mask_type = 1; // no-mask = 0; CausalFromTopLeft=1; CausalFromBottomRight=2;

    cudaDeviceProp* dp = at::cuda::getDeviceProperties(query.gpu_id);
    const int computeCapability = dp->major * 10 + dp->minor;
    // printf("compute compatibility = %i\n", computeCapability);

    bool kernel_launched = false;
    const auto maxShmem = dp->sharedMemPerBlockOptin;

    auto launchKernel = [&](auto _k, auto kernel_fn) {
        using Kernel = decltype(_k);
        using scalar_t = typename Kernel::scalar_t;
        (void)_k;

        if (kernel_launched) {
        	return;
        }
        // Check if this kernel is compatible
        if (!Kernel::kSupportsDropout && use_dropout) {
        	return;
        }
		
        if (value_shape[3] > Kernel::kMaxK || key_shape[3] > Kernel::kMaxK) {
        	return;
        }

        // // Uses too much shmem
        size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
        if (smem_bytes > maxShmem) {
        	return;
        }
        kernel_launched = true;

        // // NOTE: Should be aligned (by padding) in case M is
        // // not a good number for loading during backward
        constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;

        typename Kernel::Params p;
        p.query_ptr = (scalar_t*)query.cudaData;
        p.key_ptr = (scalar_t*)key.cudaData;
        p.value_ptr = (scalar_t*)value.cudaData;

		p.logsumexp_ptr  = nullptr;

		p.output_accum_ptr = nullptr;
        p.output_ptr = (typename Kernel::output_t*)attended_tmp_out.cudaData;

        if (seqstart_q.cudaData != nullptr) {
            p.seqstart_q_ptr = (int32_t*)seqstart_q.cudaData;
            p.seqstart_k_ptr = (int32_t*)seqstart_k.cudaData;
        }

        p.num_heads = num_heads;
        p.head_dim =  query_shape[3]; //query.shape[3];
        p.head_dim_value = value_shape[3]; //value.shape[3];
        p.num_queries = max_seqlen_q;
        p.num_keys = max_seqlen_k;
        p.num_batches =  seqstart_q.cudaData != nullptr ? dynamic_bsz : B;
        p.custom_mask_type = custom_mask_type;
		// printf("cutlass shape: [B=%i,M=%i,HN=%i,K=%i]\n", p.num_batches, p.num_queries, p.num_heads, p.head_dim);

        p.seqlen_k_ptr = nullptr;

        p.scale = float(1.0 / std::sqrt(float(p.head_dim)));

		p.q_strideB = query_strides[0]; // query.get_stride(0);
		p.k_strideB = key_strides[0]; //key.get_stride(0);
		p.v_strideB = value_strides[0]; // value.get_stride(0);
		p.q_strideM = query_strides[1]; //query.get_stride(1);
		p.k_strideM = key_strides[1]; // key.get_stride(1);
		p.v_strideM = value_strides[1]; // value.get_stride(1);
		p.q_strideH = query_strides[2]; // query.get_stride(2);
		p.k_strideH = key_strides[2]; // key.get_stride(2);
		p.v_strideH = value_strides[2]; //value.get_stride(2);
		p.o_strideM = query_strides[1]; // attended_tmp_out.get_stride(1);
        
        // printf("cutlass strides: [B=%i,M=%i,HN=%i,K=1]\n", p.q_strideB, p.q_strideM, p.q_strideH);

        // if (bias.has_value()) {
        // CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
        // TORCH_CHECK(
        //     bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
        //     "invalid dtype for bias - should match query's dtype");

        // p.attn_bias_ptr = (scalar_t*)bias->data_ptr();
		p.attn_bias_ptr = nullptr;

        p.use_dropout = use_dropout;

        if (smem_bytes > 0xc000) {
        	auto err = cudaFuncSetAttribute(
            	kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
		
			if (err == cudaErrorInvalidValue) {
				printf("ERROR: This GPU does not have enough shared-memory (kernel requires %i kb)", smem_bytes / 1024);
			}
        }
        auto blocks = p.getBlocksGrid();
        
        // if (blocks.x * blocks.y * blocks.z == 0 || key.size(1) == 0) {
		// 	res.zero_();
		// 	return;
        // }
        // Kernel::check_supported(p);
        kernel_fn<<<blocks, p.getThreadsGrid(), smem_bytes>>>(p); // not using defaultStream, seems not compatible with multi-thread ddp streams.
        // cudaDeviceSynchronize();
        // printf("kernel exit\n");
        // attended_tmp_out.const_print(std::string("kernel out print"));
    };
	
    if (query.dtype == liteqwen::DataType::FLOAT32) {                        
    	using scalar_t = float;                                              
		dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
    } else if (query.dtype ==  liteqwen::DataType::FLOAT16) {
    	using scalar_t = cutlass::half_t;
    	dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
    } else if (query.dtype == liteqwen::DataType::BFLOAT16) {
    	using scalar_t = cutlass::bfloat16_t;
    	dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
    } else {          
		printf("invalid dtype in query, key value for cutlass input. Should be using float32, float16 or bfloat16.");
    	// XFORMERS_CHECK(false, "Only fp32, half & bf16 supported at the moment");
    }    	

	// TORCH_CHECK(kernel_launched, "cutlassF: no kernel found to launch!");
	// AT_CUDA_CHECK(cudaGetLastError());
	if (! kernel_launched) {
		printf("ERROR: Kernel not launched, cutlass flash attention not returning correct result.\n");
	}
}

cudaStream_t defaultStream2 = nullptr;

void xformer_self_attention_fwd_old(const liteqwen::Data& attended_tmp_out, const liteqwen::Data& query, const liteqwen::Data& key, const liteqwen::Data& value, const liteqwen::Data& flash_attn_workspace, int bsz, int query_heads, int kv_heads, int inp_len, int channel) {
    // old static bsz=1 version of flash attention. not used.
    // xformer forward self attention modified from xformer source code (path= ./attention_forward_generic.cu), the original code is not compiled but using as a comparison backup.
	// q, k, v shape = [b, seqlen, num_heads, K]
    //bias shape = [b, num_heads, seqlen, seqlen]; bias.has_value() should be false, unless using relative position bias such as in bert.
    // int64_t max_seqlen_q, max_seqlen_k; == inp_len

	int64_t max_seqlen_q = inp_len;
	int64_t max_seqlen_k = inp_len;
    if (defaultStream2 == nullptr) {
        cudaError_t create_result = cudaStreamCreate(&defaultStream2);
    }
    int64_t B = bsz;
    int64_t M = inp_len;
    int64_t N = inp_len;
    int64_t num_heads = query_heads;
    int64_t K = channel;
    int64_t Kv = channel;

    const bool use_dropout = false; // false for inference
	const bool compute_logsumexp = false; // true when needs_gradient
	const int64_t custom_mask_type = 1; // no-mask = 0; CausalFromTopLeft=1; CausalFromBottomRight=2;

    cudaDeviceProp* dp = at::cuda::getDeviceProperties(query.gpu_id);
    const int computeCapability = dp->major * 10 + dp->minor;
    // printf("compute compatibility = %i\n", computeCapability);

    bool kernel_launched = false;
    const auto maxShmem = dp->sharedMemPerBlockOptin;

    auto launchKernel = [&](auto _k, auto kernel_fn) {
        using Kernel = decltype(_k);
        using scalar_t = typename Kernel::scalar_t;
        (void)_k;

        if (kernel_launched) {
        	return;
        }
        // Check if this kernel is compatible
        if (!Kernel::kSupportsDropout && use_dropout) {
        	return;
        }
		
        if (value.shape[3] > Kernel::kMaxK || key.shape[3] > Kernel::kMaxK) {
        	return;
        }
        // Alignment
        if ((query.get_stride(2) % Kernel::kAlignmentQ) ||
            (key.get_stride(2) % Kernel::kAlignmentK) ||
            (value.get_stride(2) % Kernel::kAlignmentV)) {
			printf("strides mismatched, not executing cutlass kernel...\n");
        	return;
        }
        // // Uses too much shmem
        size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
        if (smem_bytes > maxShmem) {
        	return;
        }
        kernel_launched = true;

        // res = at::empty(
        //     {B, M, num_heads, Kv},
        //     query.options().dtype(
        //         CutlassToAtenDtype<typename Kernel::output_t>::atScalarType()));
        // // res is same as attended_tmp_out

        // // NOTE: Should be aligned (by padding) in case M is
        // // not a good number for loading during backward
        constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;
        // logsumexp = at::empty(
        //     {seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B,
        //     num_heads,
        //     compute_logsumexp ? ceil_div(max_seqlen_q, kAlignLSE) * kAlignLSE : 0},
        //     query.options().dtype(at::ScalarType::Float));

        typename Kernel::Params p;
        p.query_ptr = (scalar_t*)query.cudaData;
        p.key_ptr = (scalar_t*)key.cudaData;
        p.value_ptr = (scalar_t*)value.cudaData;
        // p.logsumexp_ptr = compute_logsumexp
        //     ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
        //     : nullptr;
		p.logsumexp_ptr  = nullptr;

		// printf("kernel kNeedsOutputAccumulatorBuffer = %i\n", static_cast<int>(Kernel::kNeedsOutputAccumulatorBuffer));
        // at::Tensor output_accum;
        // if (Kernel::kNeedsOutputAccumulatorBuffer) {
        // output_accum = at::empty(
        //     {B, M, num_heads, Kv},
        //     query.options().dtype(
        //         CutlassToAtenDtype<
        //             typename Kernel::output_accum_t>::atScalarType()));
        // p.output_accum_ptr =
        //     (typename Kernel::output_accum_t*)output_accum.data_ptr();
        // } else {
        // p.output_accum_ptr = nullptr;
        // }
		p.output_accum_ptr = nullptr;
        p.output_ptr = (typename Kernel::output_t*)attended_tmp_out.cudaData;

        // if (seqstart_q.has_value()) { // disabled, not dynamic batch inference.
        // p.seqstart_q_ptr = (int32_t*)seqstart_q->data_ptr();
        // p.seqstart_k_ptr = (int32_t*)seqstart_k->data_ptr();
        // }

        p.num_heads = num_heads;
        p.head_dim = query.shape[3];
        p.head_dim_value = value.shape[3];
        p.num_queries = max_seqlen_q;
        p.num_keys = max_seqlen_k;
        // p.num_batches = seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B;
		p.num_batches = B;
        p.custom_mask_type = custom_mask_type;
		// printf("[B=%i,M=%i,HN=%i,K=%i]\n", p.num_batches, p.num_queries, p.num_heads, p.head_dim);


        p.seqlen_k_ptr = nullptr;
        // if (seqlen_k.has_value()) {
        // CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(seqlen_k.value());
        // TORCH_CHECK(seqlen_k->scalar_type() == at::ScalarType::Int);
        // p.seqlen_k_ptr = (int32_t*)seqlen_k->data_ptr();
        // }

        // if (window_size.has_value()) {
        // p.window_size = *window_size;
        // }

        // if (scale.has_value()) {
        // p.scale = float(*scale);
        // } else {
        p.scale = float(1.0 / std::sqrt(float(p.head_dim)));
        // }

        // ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
        // ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
        // ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
        // ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
        // ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
        // ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
        // ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
        // ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
        // ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));
        // ASSIGN_CHECK_OVERFLOW(p.o_strideM, res.stride(1));

		p.q_strideB = query.get_stride(0);
		p.k_strideB = key.get_stride(0);
		p.v_strideB = value.get_stride(0);
		p.q_strideM = query.get_stride(1);
		p.k_strideM = key.get_stride(1);
		p.v_strideM = value.get_stride(1);
		p.q_strideH = query.get_stride(2);
		p.k_strideH = key.get_stride(2);
		p.v_strideH = value.get_stride(2);
		p.o_strideM = attended_tmp_out.get_stride(1);

        // if (bias.has_value()) {
        // CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
        // TORCH_CHECK(
        //     bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
        //     "invalid dtype for bias - should match query's dtype");

        // p.attn_bias_ptr = (scalar_t*)bias->data_ptr();
		p.attn_bias_ptr = nullptr;

        // TORCH_CHECK(bias->dim() == 4, "Bias expected in BMHK format");
        // TORCH_CHECK(
        //     bias->size(0) == query.size(0),
        //     "attn_bias: wrong shape (batch dimension)");
        // TORCH_CHECK(
        //     bias->size(1) == query.size(2),
        //     "attn_bias: wrong shape (head dimension)");
        // TORCH_CHECK(
        //     bias->size(2) == query.size(1),
        //     "attn_bias: wrong shape (seqlenQ dimension)");
        // TORCH_CHECK(
        //     bias->size(3) == key.size(1),
        //     "attn_bias: wrong shape (seqlenKV dimension)");
        // ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias->stride(0));
        // ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias->stride(1));
        // ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias->stride(2));
        // TORCH_CHECK(
        //     bias->stride(3) == 1,
        //     "attn_bias: wrong alignment (last dimension must be contiguous)");
        // }

        p.use_dropout = use_dropout;
        // if (p.use_dropout) {
        // p.rng_engine_inputs = rng_engine_inputs;
        // p.dropout_prob = dropout_p;
        // }

        if (smem_bytes > 0xc000) {
        	auto err = cudaFuncSetAttribute(
            	kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
		
			if (err == cudaErrorInvalidValue) {
				printf("ERROR: This GPU does not have enough shared-memory (kernel requires %i kb)", smem_bytes / 1024);
			}
			// XFORMERS_CHECK(
			// 	err != cudaErrorInvalidValue,
			// 	"This GPU does not have enough shared-memory (kernel requires ",
			// 	smem_bytes / 1024,
			// 	" kb)");
			// AT_CUDA_CHECK(err);
        }
        auto blocks = p.getBlocksGrid();

        // if (blocks.x * blocks.y * blocks.z == 0 || key.size(1) == 0) {
		// 	res.zero_();
		// 	return;
        // }
        // Kernel::check_supported(p);
        kernel_fn<<<blocks, p.getThreadsGrid(), smem_bytes>>>(p); // not using defaultStream2, seems not compatible with multi-thread ddp streams.
    };

    // Dispatch to the right kernel
    // DISPATCH_liteqwen_TYPES(query, ([&]() {
    //                dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
    //              }));
	
    if (query.dtype == liteqwen::DataType::FLOAT32) {                        
    	using scalar_t = float;                                              
		dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
    } else if (query.dtype ==  liteqwen::DataType::FLOAT16) {
    	using scalar_t = cutlass::half_t;
    	dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
    } else if (query.dtype == liteqwen::DataType::BFLOAT16) {
    	using scalar_t = cutlass::bfloat16_t;
    	dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
    } else {          
		printf("invalid dtype in query, key value for cutlass input. Should be using float32, float16 or bfloat16.");
    	// XFORMERS_CHECK(false, "Only fp32, half & bf16 supported at the moment");
    }    	

	// TORCH_CHECK(kernel_launched, "cutlassF: no kernel found to launch!");
	// AT_CUDA_CHECK(cudaGetLastError());
	if (! kernel_launched) {
		printf("ERROR: Kernel not launched, cutlass flash attention not returning correct result.\n");
	}

	// uint64_t -> int64_t bitwise casting as PyTorch don't support uint64_t
	// so just fake it as a int64_t
	// int64_t seed, offset;
	// if (use_dropout) {
    // 	std::memcpy(&seed, &rng_engine_inputs.seed_, sizeof(seed));
    // 	std::memcpy(&offset, &rng_engine_inputs.offset_.val, sizeof(offset));
	// }
	// return std::make_tuple(res, logsumexp, seed, offset);
}
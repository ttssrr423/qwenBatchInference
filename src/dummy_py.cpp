
#ifdef WIN32
#define DLL_EXPORT _declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#include "core_cpu.h"
#include "core_gpu.cuh"
// #include "pool.h"
// #include "kv_cache.h"
#include "entities.h"
#include "xformer_attention.h"
// #include "forward_gpu.cuh"
void dummy_test() {
    printf("dummy testing memory-efficient attention with dynamic batching\n");
    int BL = 512;
    int H = 32;
    int D = 128;
    int max_B = 32;
    size_t numel = static_cast<size_t>(BL) * H * D;

    init_device_property_for_device(1, 0, 1);
    DeviceSynchronize();

    auto tens_a = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{1, BL, H, D}, 0, true);
    tens_a.Allocate();
    auto tens_o = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{1, BL, H, D}, 0, true);
    tens_o.Allocate();

    auto flash_attn_workspace = liteqwen::Data(liteqwen::DataType::FLOAT32, std::vector<int>{1, BL, H, D}, 0, true); // 只在flash attention的output非fp32时被使用
    flash_attn_workspace.Allocate();

    auto tens_starts = liteqwen::Data(liteqwen::DataType::INT32, std::vector<int>{max_B}, 0, true);
    tens_starts.Allocate();
    auto tens_k_starts = liteqwen::Data(liteqwen::DataType::INT32, std::vector<int>{max_B}, 0, true); // 需要q和k的starts不同指针。
    tens_k_starts.Allocate();
    auto seqlen_k = liteqwen::Data(liteqwen::DataType::INT32); // null data

    float* cpu_data = new float[numel];
    int* cpu_starts = new int[max_B];
    int max_q_len = 0;
    std::vector<int> start_poses{0, 10, 30, 80, 250, 400, 480};
    for (int bi = 0; bi < start_poses.size(); bi++) {
        int end_t = (bi == (int)(start_poses.size() - 1)) ? BL : start_poses[bi+1];
        printf("assigning for bi=%i, t=[%i, %i) with values bi+1\n", bi, start_poses[bi], end_t);
        for (size_t t=start_poses[bi]; t<end_t; t++) {
            for (size_t hd=0; hd<H*D; hd++) {
                cpu_data[t*H*D+hd] = static_cast<float>(bi+1);
            }
        }
        cpu_starts[bi] = start_poses[bi];
        if (bi > 0) {
            int b_len = (start_poses[bi] - start_poses[bi-1]);
            if (b_len > max_q_len)  max_q_len = b_len;
        }
    }
    int dynamic_bsz = static_cast<int>(start_poses.size()-1);
    // cpu_starts[start_poses.size()-1] = start_poses.back();
    tens_a.Fp32CpuToFp16Upload(0, cpu_data);
    tens_a.print(std::string("query"));

    tens_starts.UploadIntValues(start_poses.size(), 0, cpu_starts);
    tens_starts.print(std::string("starts_q"));
    tens_k_starts.UploadIntValues(start_poses.size(), 0, cpu_starts);
    tens_k_starts.print(std::string("starts_k"));

    // tens_o.print(std::string("attended_out"));
    DeviceSynchronize();
    // xformer_self_attention_fwd(tens_o, tens_a, tens_a, tens_a, tens_starts, tens_k_starts, seqlen_k, flash_attn_workspace, dynamic_bsz, H, H, max_q_len, max_q_len, D);
    xformer_self_attention_fwd(tens_o, tens_a, tens_a, tens_a, tens_starts, tens_k_starts, seqlen_k, flash_attn_workspace, dynamic_bsz, H, H, max_q_len, max_q_len, D);
    // xformer_self_attention_fwd_old(tens_o, tens_a, tens_a, tens_a, flash_attn_workspace, 1, H, H, BL, D);
    // tens_o.print(std::string("attended_out"));
    DeviceSynchronize();
    
    for (int bi = 0; bi < start_poses.size()-1; bi++) {
        int start = start_poses[bi];
        int end = start_poses[bi+1];
        size_t offset_bi = static_cast<size_t>(start) * H * D;
        auto slice_bi = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{(end-start),H,D}, 0, tens_o.cudaData, offset_bi);
        DeviceSynchronize();
        slice_bi.print(std::string("attended_out_b=")+std::to_string(bi));
    }
    // tens_o.print(std::string("attended_out"));

    tens_starts.print(std::string("starts_q"));
    tens_k_starts.print(std::string("starts_k"));

    tens_a.Free();
    tens_starts.Free();
    tens_k_starts.Free();
    tens_o.Free();
    flash_attn_workspace.Free();
    DeviceSynchronize();
}

extern "C" {
    DLL_EXPORT void initialize_empty_qwen2(int world_size, int data_parallel_size, int pipeline_parallel, char* json_config_path, int layer_num, int* block2device_list, int max_dynamic_bsz, int max_sequence_length, int max_queue_size, int timeout) {
        dummy_test();

        // std::string config_path = std::string(json_config_path);
        // std::vector<int> block2device_list_vec; // = std::vector<int>(layer_num);
        // for (int li=0; li < layer_num; li++){
        //     block2device_list_vec.push_back(block2device_list[li]);
        // }
        // init_empty_qwen2(world_size, data_parallel_size, config_path, block2device_list_vec, max_dynamic_bsz, max_sequence_length, max_queue_size, timeout);
        return;
    }
}
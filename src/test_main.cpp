#include <iostream>
#include <chrono>
#include "core_cpu.h"
#include "core_gpu.cuh"
#include "pool.h"
#include "kv_cache.h"
#include "entities.h"
#include "xformer_attention.h"
#include "forward_gpu.cuh"

void test_tensors() {
    //std::cout << "Hello World!" << std::endl;
    int B = 1;
    int L = 320000;
    int D = 8192;
    size_t numel = static_cast<size_t>(B) * L * D;
    printf("numel=%lu\n", numel);
    // auto tens_a = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{B, L, D}, 0, true);
    auto tens_b = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{B, L, D}, 0, true);
    auto tens_c = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{B, L+2, D}, 0, true);
    // tens_a.Allocate();
    tens_b.Allocate();
    tens_c.Allocate();

    ConstantFill(tens_b.cudaData, tens_b.dtype, tens_b.numel(), 3.0);
    tens_b.print(std::string("tensb"));
    CopyGPUData(tens_b.dtype, tens_c.cudaData, tens_b.cudaData, 0, static_cast<size_t>(D), (size_t)0, numel, false);
    tens_c.print(std::string("tensc"));

    // tens_a.Free();
    tens_b.Free();
    tens_c.Free();
}

void test_dynamic_batch_kernel() {
    int head_num = 40;
    int channel = 128;
    int BL = 1024;
    int max_batch = 10;

    auto pos_mapping = liteqwen::Data(liteqwen::DataType::INT8, std::vector<int>{BL}, 0, true);
    auto start_positions = liteqwen::Data(liteqwen::DataType::INT32, std::vector<int>{max_batch+1}, 0, true); 
    
    int8_t* block_positions_cpu = new int8_t[BL];  // 紧密排布下pos到batch_id的映射
    int* position_offsets_cpu = new int[max_batch+1]; // 每个batch_id的位置偏移
    int pos_acc = 0;
    int prev_acc = 0;
    for (int i=0; i<max_batch; i++) {
        int len = 10 + i;
        pos_acc += len;
        for (int j = prev_acc; j<pos_acc; j++) {
            block_positions_cpu[j] = static_cast<int8_t>(i);
        }
        position_offsets_cpu[i] = prev_acc;
        prev_acc = pos_acc;
    }
    position_offsets_cpu[max_batch] = pos_acc;
    size_t boundary = pos_acc;

    pos_mapping.Allocate();
    start_positions.Allocate();
    printf("uploading int32 positions\n");
    // UploadInt32(pos_mapping.cudaData, (uint8_t*)block_positions_cpu, 0, 0, 0, BL, BL, false);
    UploadData(liteqwen::DataType::INT8, pos_mapping.cudaData, (uint8_t*)block_positions_cpu, 0, 0, 0, BL);
    UploadInt32(start_positions.cudaData, (uint8_t*)position_offsets_cpu, 0, 0, 0, max_batch+1, max_batch+1, false);

    auto tens_a = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{1, BL, head_num*channel}, 0, true); //最大placeholder
    tens_a.Allocate();
    DeviceSynchronize();
    printf("start test launching dynamic kernel\n");

    // typedef DynamicBatchParam<__half, int> ParamFp16IntIdx;
    // ParamFp16IntIdx para;
    // para.data_ptr = tens_a.cudaData;
    // para.seqstart_ptr = start_positions.cudaData;
    // para.set_dynamic_bl(head_num, channel);
    
    dynamic_check_launch(tens_a, pos_mapping, start_positions, 10, boundary, head_num, channel);
    DeviceSynchronize();
    tens_a.Free();
    pos_mapping.Free();
    start_positions.Free();
}

void test_cache() {
    int kv_heads = 2;
    int channels = 128;
    int kv_size = kv_heads * channels;
    int dummy_len = 70;
    std::vector<int> layer_id_map;
    for (int i =0; i<40; i++) {
        layer_id_map.push_back(0);
    }
    int max_BL = 2048;
    int max_B = 16;
    int layer_id = 4;
    auto kv_cache_ref = new liteqwen::PipelineKVPool(max_B, max_BL, kv_size, layer_id_map);

    liteqwen::StringArray req_ids;
    req_ids.Init(512, 16);
    req_ids.push_back(std::string("req001"));
    req_ids.push_back(std::string("req002"));
    req_ids.push_back(std::string("req003"));
    std::vector<int> example_maxlens{128, 512, 1024};

    std::vector<liteqwen::AllocateParam> alloc_params;
    for (int bi=0; bi<req_ids.size(); bi++) {
        liteqwen::AllocateParam new_par = kv_cache_ref->pipeline_caches[0]->search_block_sequence(req_ids[bi], example_maxlens[bi], &alloc_params);
        if (new_par.successful) {
            alloc_params.push_back(new_par);
        }
    }
    kv_cache_ref->sequence_allocate_cache(alloc_params);
    DeviceSynchronize();

    kv_cache_ref->free(std::string("req002"));
    bool suc4 = kv_cache_ref->allocate_cache(std::string("req004"), 256); // 剩余 001, 004, 003

    auto dummy_k = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{dummy_len, kv_size}, 0, true);
    auto dummy_v = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{dummy_len, kv_size}, 0, true);
    dummy_k.Allocate();
    dummy_v.Allocate();
    float* dummy_values_cpu = new float[dummy_len * kv_size];
    for (int i=0; i< dummy_len*kv_size; i++) {
        dummy_values_cpu[i] = static_cast<float>(i % 100);
    }
    dummy_k.Fp32CpuToFp16Upload(0, dummy_values_cpu);
    dummy_v.Fp32CpuToFp16Upload(0, dummy_values_cpu);
    dummy_k.print(std::string("dummy_kv_value"));
    // 最终生效cache长度 = (写入长度dummy_len=70) +（write_cache_shift=2), 按照最终生效长度增量写入prefill kv
    kv_cache_ref->write_layer_kv(std::string("req004"), layer_id, std::pair<liteqwen::Data*, liteqwen::Data*>(&dummy_k, &dummy_v), 2, 0, dummy_len); // read_tensor_shift=0, write_cache_shift=2
    int prefill_len = dummy_len + 2;
    DeviceSynchronize();

    // 准备 prefill_len+1步的增量数据（request_ids=[req004], dynamic_bsz=1, dynamic_bl=prefill_len+1）。
    req_ids.clear();
    req_ids.push_back(std::string("req004"));
    int dynamic_bsz = 1;
    auto dummy_k_delta = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{1, kv_size}, 0, true);
    auto dummy_v_delta = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{1, kv_size}, 0, true);
    dummy_k_delta.Allocate();
    dummy_v_delta.Allocate();
    float* dummy_delta_cpu = new float[1 * kv_size];
    for (int i=0; i< 1*kv_size; i++) {
        dummy_delta_cpu[i] = static_cast<float>(i % 100) + 100.0f;
    }
    dummy_k_delta.Fp32CpuToFp16Upload(0, dummy_delta_cpu);
    dummy_v_delta.Fp32CpuToFp16Upload(0, dummy_delta_cpu);

    auto cpu_starts = new int[max_B+1];
    cpu_starts[0] = 0;
    cpu_starts[1] = prefill_len + 1;
    auto cpu_bids = new uint8_t[max_BL];
    for (int i=0; i<(prefill_len+1); i++) {
        int batch_id = 0;
        cpu_bids[i] = static_cast<uint8_t>(batch_id);
    }

    auto pos_starts = liteqwen::Data(liteqwen::DataType::INT32, std::vector<int>{max_B+1}, 0, true);
    auto pos_offsets = liteqwen::Data(liteqwen::DataType::INT64, std::vector<int>{2 * max_B}, 0, true); // 中间结果
    int ptr_data_len = sizeof(void*) * 2 * max_B / sizeof(uint8_t); // ptrs = [&k1, &v1, &k2, &v2, ...]
    auto batch_kv_ptrs = liteqwen::Data(liteqwen::DataType::INT8, std::vector<int>{ptr_data_len}, 0, true); // 只在decode attention时使用。
    auto seq_bids = liteqwen::Data(liteqwen::DataType::INT8, std::vector<int>{max_BL}, 0, true);
    pos_starts.Allocate();
    seq_bids.Allocate();
    pos_offsets.Allocate();
    batch_kv_ptrs.Allocate();

    UploadInt32(pos_starts.cudaData, (uint8_t*)cpu_starts, 0, 0, 0, static_cast<size_t>(dynamic_bsz+1), static_cast<size_t>(max_B+1), true);
    pos_starts.print(std::string("pos_starts"));
    UploadData(liteqwen::DataType::INT8, seq_bids.cudaData, cpu_bids, 0, 0, 0, max_BL);

    // 写入decode kv
    kv_cache_ref->write_batch_layer_kv(false, req_ids, layer_id, std::pair<liteqwen::Data*, liteqwen::Data*>(&dummy_k_delta, &dummy_v_delta), pos_offsets, pos_starts, seq_bids, max_B, (prefill_len+1), kv_heads, channels);
    DeviceSynchronize();

    // 检验增量写入后的全量kv内容(req004)
    printf("key prefill+delta is:\n");
    DeviceSynchronize();
    kv_cache_ref->print_cache(req_ids[0], layer_id, false, prefill_len+1);
    DeviceSynchronize();
    printf("value prefill+delta is:\n");
    DeviceSynchronize();
    kv_cache_ref->print_cache(req_ids[0], layer_id, true, prefill_len+1);
    DeviceSynchronize();
    dummy_k.Free();
    dummy_v.Free();
    dummy_k_delta.Free();
    dummy_v_delta.Free();
}

liteqwen::ResponseContext get_mock_request(std::string req_id, int ids_arange_len, int ids_arange_base, int maxlen, std::string adapter_name) {
    liteqwen::ResponseContext resp1;
    liteqwen::GenerationConfig gen_cfg1;
    std::vector<int> inp_ids;
    for (int i=0; i<ids_arange_len;i++) {
        inp_ids.push_back(i+ids_arange_base);
    }
    gen_cfg1.max_new_tokens = maxlen-ids_arange_len;
    gen_cfg1.adapter_name = adapter_name;
    resp1.Init(req_id, inp_ids, gen_cfg1, 0.0, 0.0, std::vector<int>(), false);
    return resp1;
}

std::string verifying_batch_inputs(std::vector<int> int_li, std::vector<int> starts, bool short_print=true) {
    int list_len = (int)(int_li.size());
    int bsz = (int)(starts.size()) - 1;
    std::string sb = std::string("len=") + std::to_string(list_len) + std::string(", bsz=") + std::to_string(bsz)  + std::string(", vals=[");

    bool dot_appended = false;
    for (int i=0; i<list_len; i++) {
        if (short_print && list_len > 10 && i > 4 && i<list_len-5) {
            if (!dot_appended) {
                sb += std::string(" ... ");
                dot_appended = true;
            }

            if (starts.size() > 2) {
                for (int j=1; j<starts.size()-1; j++) {
                    if (i == starts[j]) {
                        int pos_new_start = starts[j];
                        int prev_tk_id = int_li[pos_new_start-1];
                        int cur_tk_id = int_li[pos_new_start];
                        int nxt_tk_id = int_li[pos_new_start+1];
                        sb += (std::to_string(prev_tk_id) + std::string(",")+ std::to_string(cur_tk_id) + std::string(",") + std::to_string(nxt_tk_id) + std::string(" ..."));
                        break;
                    }
                }
            }
            continue;
        }
        if (i==0) {
            sb += std::to_string(int_li[i]);
        } else {
            sb += (","+std::to_string(int_li[i]));
        }
    }
    return sb + std::string("]");
}

void test_batching() {
    int kv_size = 256;
    
    std::vector<int> layer_id_map;
    for (int i =0; i<40; i++) {
        layer_id_map.push_back(0);
    }
    auto kv_cache_ref = new liteqwen::PipelineKVPool(16, 1024, kv_size, layer_id_map);

    int max_queue_size = 10;
    int timeout = 120 * 1000;
    auto context_pool = (std::shared_ptr<liteqwen::ContextPool>)(new liteqwen::ContextPool(max_queue_size, timeout));

    auto batch_inp_preparer = new liteqwen::BatchInputPreparer(16, 1024);
    batch_inp_preparer->Empty();

    std::vector<int> req_lengths{100, 5, 140, 200, 10, 1020, 64};
    std::vector<int> req_maxlens{128, 64, 256, 256, 64, 1024, 128};
    std::vector<int> req_eos_lengths{110, 64, 200, 240, 30, 1022, 128};
    // std::vector<int> req_eos_lengths{100, 10, 140, 200, 10, 1020, 64};
    std::vector<std::string> req_loras{"default", "default", "default", "default", "default", "default", "default"};
    std::map<std::string, int> request_id_map;
    for (int bi=0; bi<req_lengths.size(); bi++) {
        std::string req_id = std::string("req00")+std::to_string(bi);
        liteqwen::ResponseContext ctx = get_mock_request(req_id, req_lengths[bi], 10000*bi, req_maxlens[bi], req_loras[bi]);
        context_pool->Add(req_id, ctx);
        request_id_map[req_id] = bi;
    }

    int generated_t = 0;
    int eos_iter_ct = 0;
    int total_prefill_ct = 0;
    int total_decode_ct = 0;

    liteqwen::StringArray decode_request_ids;
    std::string prev_forward_lora = "skip"; // warmup memory allocation
    decode_request_ids.Init(512, 16);
    while (true) {
        std::string preparer_lora_name = batch_inp_preparer->GetLoraName();
        std::vector<liteqwen::AllocateParam> allocate_params = context_pool->Reload(0, preparer_lora_name, batch_inp_preparer->all_eos, kv_cache_ref);
        if (allocate_params.size() > 0) { // prefill
            batch_inp_preparer->ClearPrefill();
            auto suc = kv_cache_ref->sequence_allocate_cache(allocate_params);
            // std::vector<std::string> prefill_req_ids;
            for (int prefill_i=0; prefill_i < allocate_params.size(); prefill_i++) {
                std::string prefill_req_id = allocate_params[prefill_i].request_id;
                batch_inp_preparer->AddPrefill(context_pool->GetRes(prefill_req_id, true));
                // prefill_req_ids.push_back(prefill_req_id);
            }

            if (allocate_params[0].lora_name != prev_forward_lora) {
                // 切换lora时，lora_r可能不同，所以清理BigBuffer, 重新预分配。
                printf("CUDA MEM: data_id %i clearing big buffer for lora switch: %s->%s\n", 0, prev_forward_lora.c_str(), allocate_params[0].lora_name.c_str());
                //ManagedCudaClearBigBuffer(...,...);
                DeviceSynchronize();
            }

            // printf("first frame forwarding with %i prefills...\n", (int)(allocate_params.size()));
            // TODO:: executing upload & forward...
            // std::string ids_string = verifying_batch_inputs(batch_inp_preparer->input_ids, batch_inp_preparer->input_starts);
            // printf((std::string("prefill_ids:") + ids_string + std::string("\n\n")).c_str());
            total_prefill_ct += 1;

            // mocking sampling results...
            std::vector<int> dummy_generated_ids;
            std::vector<bool> dummy_eos;
            for (int bi=0; bi<(batch_inp_preparer->prefill_req_ids).size(); bi++) {
                std::string req_id = (batch_inp_preparer->prefill_req_ids)[bi];
                int req_id_num = request_id_map.find(req_id)->second;
                int prev_id = 0;
                dummy_generated_ids.push_back(req_id_num * 10000 + (prev_id+1));
                int example_id = request_id_map.find(req_id)->second;
                bool is_eos = req_eos_lengths[example_id] - req_lengths[example_id] < ((prev_id+1) % 10000);
                dummy_eos.push_back(is_eos);
            }

            generated_t += 1;
            // printf("generated mock new steps=%i with %i examples, updating...\n", generated_t, (int)(prefill_req_ids.size()));
            batch_inp_preparer->PrefillUpdate(0, batch_inp_preparer->prefill_req_ids, dummy_eos, dummy_generated_ids, context_pool, kv_cache_ref);
            continue;
        }

        std::string latest_lora = batch_inp_preparer->GetLoraName();
        int decoding_example_ct = (int)(batch_inp_preparer->decoding_examples).size();
        decode_request_ids.clear();
        for (int bi=0; bi< decoding_example_ct; bi++) {
            // 将已经prepare好的样本的request_ids备份到decode_request_ids中，在运行ClearDecode()前。
            std::string req_id = (batch_inp_preparer->decode_req_ids)[bi];
            decode_request_ids.push_back(req_id);
        }
        // printf("decoding frame forwarding with %i examples...\n", decoding_example_ct);
        // TODO:: executing upload & forward...

        if (decoding_example_ct > 0)
            total_decode_ct += 1;
        batch_inp_preparer->ClearDecode();
        if (latest_lora != std::string("ANY")) {
            prev_forward_lora = latest_lora;
        }
        
        std::vector<int> decode_ids;
        std::vector<bool> decode_eos;
        for (int bi=0; bi< decoding_example_ct; bi++) {
            std::string req_id = decode_request_ids[bi];
            auto example_ctx_ref = context_pool->GetRes(req_id);
            int example_token_len = (int)(example_ctx_ref->tokens.size());
            int prev_token_id = (example_ctx_ref->tokens)[example_token_len-1];
            decode_ids.push_back(prev_token_id+1);
            int example_id = request_id_map.find(req_id)->second;
            bool is_eos = req_eos_lengths[example_id] - req_lengths[example_id] < ((prev_token_id+1) % 10000);
            decode_eos.push_back(is_eos);
        }
        generated_t += 1;

        batch_inp_preparer->DecodeUpdate(0, decode_request_ids, decode_eos, decode_ids, context_pool, kv_cache_ref);
        // batch_inp_preparer->check_print();

        if (batch_inp_preparer->all_eos) {
            eos_iter_ct += 1;
        } else {
            eos_iter_ct = 0;
        }
        if (generated_t > 1000 || (batch_inp_preparer->all_eos && eos_iter_ct > 10)) {
            printf("stop testing dynamic batching, exiting loop.\n");
            break;
        }
    }

    printf("\n\n===================\nfinal checking generated results:\n");
    for (int bi=0; bi<req_lengths.size(); bi++) {
        std::string req_id = std::string("req00")+std::to_string(bi);
        liteqwen::ResponseContext* ctx_ref = context_pool->GetRes(req_id);
        std::string sb = req_id + std::string("(") + std::to_string(ctx_ref->input_length) + std::string("|") + std::to_string(ctx_ref->tokens.size()) + std::string("): ");
        for (int tid=ctx_ref->input_length; tid<ctx_ref->tokens.size(); tid++) {
            int token_id = (ctx_ref->tokens)[tid];
            sb += (std::string(",")+std::to_string(token_id));
        }
        printf((sb+std::string("\n\n")).c_str());
    }

    printf("total prefills: %i, total decodes: %i\n", total_prefill_ct, total_decode_ct);
}

void test_fmha() {
    printf("testing memory-efficient attention with dynamic batching\n");
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
    xformer_self_attention_fwd(tens_o, tens_a, tens_a, tens_a, tens_starts, tens_k_starts, seqlen_k, flash_attn_workspace, dynamic_bsz, BL, H, max_q_len, max_q_len, D);
    tens_o.print(std::string("attended_out"));
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
    delete cpu_data;
}


void test_fmha2(bool do_init) {
    printf("testing memory-efficient attention with dynamic batching\n");
    int BL = 4096;
    int H = 40;
    int D = 128;
    int max_B = 17;
    size_t numel = static_cast<size_t>(BL) * H * D;

    if (do_init) {
        init_device_property_for_device(1, 0, 1);
        DeviceSynchronize();
    }
    
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
    std::vector<int> start_poses{0, 1874, 1900};
    for (int bi = 0; bi < start_poses.size(); bi++) {
        int end_t = (bi == (int)(start_poses.size() - 1)) ? BL : start_poses[bi+1];
        printf("assigning for bi=%i, t=[%i, %i) with values bi+1\n", bi, start_poses[bi], end_t);
        if (bi < start_poses.size()-1) {
            for (size_t t=start_poses[bi]; t<end_t; t++) {
                for (size_t hd=0; hd<H*D; hd++) {
                    cpu_data[t*H*D+hd] = static_cast<float>(bi+1);
                }
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
    xformer_self_attention_fwd(tens_o, tens_a, tens_a, tens_a, tens_starts, tens_k_starts, seqlen_k, flash_attn_workspace, dynamic_bsz, BL, H, max_q_len, max_q_len, D);
    tens_o.print(std::string("attended_out"));
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
    delete cpu_data;
}

void test_decode_attn() {
    int kv_heads = 2;
    int kv_size = 128 * kv_heads;
    int query_heads = 32;
    int max_batch = 7;
    int max_BL = 2048;
    int channel = 128;
    std::vector<int> layer_id_map;
    for (int i =0; i<40; i++) {
        layer_id_map.push_back(0);
    }
    auto kv_cache_ref = new liteqwen::PipelineKVPool(16, max_BL, kv_size, layer_id_map);
    int layer_id = 4;

    liteqwen::StringArray req_ids;
    req_ids.Init(512, max_batch);
    req_ids.push_back(std::string("req001"));
    req_ids.push_back(std::string("req002"));
    req_ids.push_back(std::string("req003"));
    int dynamic_bsz = req_ids.size();
    std::vector<int> request_lens{60, 140, 200};
    std::vector<int> max_lens{128, 256, 1024};
    std::vector<float> k_floats{1.0, 5.0, 10.0};

    // 申请kv-cache块
    std::vector<liteqwen::AllocateParam> alloc_params;
    for (int bi=0; bi<dynamic_bsz; bi++) {
        auto new_param = kv_cache_ref->pipeline_caches[0]->search_block_sequence(req_ids[bi], max_lens[bi], &alloc_params);
        if (new_param.successful) {
            alloc_params.push_back(new_param);
        }
    }
    kv_cache_ref->sequence_allocate_cache(alloc_params);

    // 准备key value数据
    int max_t = 1;
    auto cpu_starts = new int [max_batch+1];
    cpu_starts[0] = 0;
    int len_accu = 0;
    auto cpu_bids = new uint8_t[max_BL];

    auto cpu_keys = new float[max_BL * kv_size];
    auto cpu_values = new float[max_BL * kv_size];
    auto key_tensor = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{max_BL, kv_size}, 0, false);
    auto value_tensor = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{max_BL, kv_size}, 0, false);
    key_tensor.Allocate();
    value_tensor.Allocate();
    for (int bi = 0; bi < req_ids.size(); bi++) {
        printf("inserting example kv-cache %s\n", req_ids[bi].c_str());
        if (max_t < request_lens[bi]) max_t = request_lens[bi];
        cpu_starts[bi+1] = len_accu + request_lens[bi];
        len_accu += request_lens[bi];
        printf("bids for bi=%i, pos_range=[%i, %i)\n", bi, cpu_starts[bi],  cpu_starts[bi]+request_lens[bi]);
        for (int pos=0; pos< request_lens[bi]; pos++) {
            cpu_bids[cpu_starts[bi]+pos] = static_cast<uint8_t>(bi);
        }
        int inp_offset = cpu_starts[bi] * kv_size;
        for (int t=0; t<request_lens[bi]; t++) {
            for (int c=0; c<kv_size; c++) {
                cpu_keys[inp_offset + t * kv_size + c] = k_floats[bi];
                cpu_values[inp_offset + t * kv_size + c] = k_floats[bi]+1.0;
            }
        }
    }
    int bl_bound = cpu_starts[dynamic_bsz];
    DeviceSynchronize();
    UploadCastFp32ToFp16Data((void*)key_tensor.cudaData, cpu_keys, 0, 0, 0, static_cast<size_t>(bl_bound)*kv_size);
    UploadCastFp32ToFp16Data((void*)value_tensor.cudaData, cpu_values, 0, 0, 0, static_cast<size_t>(bl_bound)*kv_size);
    // prefill kv 写入cache。其实是前一步prefill(t步)+1步decode拼接后的数据一起写入。
    auto pos_starts = liteqwen::Data(liteqwen::DataType::INT32, std::vector<int>{max_batch+1}, 0, true);
    auto pos_offsets = liteqwen::Data(liteqwen::DataType::INT64, std::vector<int>{2 * max_batch}, 0, true);
    int ptr_data_len = sizeof(void*) * 2 * max_batch / sizeof(uint8_t); // ptrs = [&k1, &v1, &k2, &v2, ...]
    auto batch_kv_ptrs = liteqwen::Data(liteqwen::DataType::INT8, std::vector<int>{ptr_data_len}, 0, true); // 只在decode attention时使用。
    auto seq_bids = liteqwen::Data(liteqwen::DataType::INT8, std::vector<int>{max_BL}, 0, true);
    pos_starts.Allocate();
    seq_bids.Allocate();
    pos_offsets.Allocate();
    batch_kv_ptrs.Allocate();

    UploadInt32(pos_starts.cudaData, (uint8_t*)cpu_starts, 0, 0, 0, static_cast<size_t>(dynamic_bsz+1), static_cast<size_t>(max_batch+1), true);
    pos_starts.print(std::string("pos_starts"));
    UploadData(liteqwen::DataType::INT8, seq_bids.cudaData, cpu_bids, 0, 0, 0, max_BL);
    kv_cache_ref->write_batch_layer_kv(true, req_ids, layer_id, std::pair<liteqwen::Data*, liteqwen::Data*>(&key_tensor, &value_tensor), pos_offsets, pos_starts, seq_bids, max_batch, bl_bound, kv_heads, 128);

    // 准备单步query
    auto query_cpu = new float[max_batch * query_heads * channel];
    for (int i=0; i< dynamic_bsz; i++) {
        for (int j = 0; j< query_heads * channel; j++) {
            query_cpu[i * query_heads * channel + j] = static_cast<float>(i+1);
        }
    }
    auto query = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{max_batch, query_heads, channel}, 0, true);
    query.Allocate();
    query.Fp32CpuToFp16Upload(0, query_cpu);
    query.print(std::string("query"));
    DeviceSynchronize();

    // out put placeholders, over-numel allocation to ensure non-compact data can be stored without re-allocate.
    auto attended_out = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{max_batch, 1, query_heads*channel}, 0, true);
    attended_out.Allocate();
    auto scores = liteqwen::Data(liteqwen::DataType::FLOAT32, std::vector<int>{max_BL, query_heads}, 0, true);
    scores.Allocate();
    // auto bhld_value = liteqwen::Data(liteqwen::DataType::FLOAT16, std::vector<int>{max_batch, query_heads, max_t, channel}, 0, true);
    // bhld_value.Allocate(0, liteqwen::DataType::FLOAT16, static_cast<size_t>(max_batch)*query_heads*max_BL*channel);

    DeviceSynchronize();

    auto start_tm = std::chrono::system_clock::now();

    // method call
    for (int ii=0; ii<10000; ii++) {
        decode_attention(attended_out, scores, req_ids, bl_bound, max_t, layer_id, query, seq_bids, batch_kv_ptrs, pos_offsets, pos_starts, max_batch, kv_cache_ref, kv_heads, channel, true);
        // DeviceSynchronize();
        // scores.print(std::string("scores"));
        // attended_out.print(std::string("attended_out"));
    }
    DeviceSynchronize();
    auto end_tm = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_tm - start_tm);
    std::cout<< double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << " secs" << std::endl;

    scores.print(std::string("scores"));
    attended_out.print(std::string("attended_out"));
    DeviceSynchronize();
}

int main()
{
    // // 测试张量以及动态batch传参
    // test_tensors();
    // test_dynamic_batch_kernel();

    // // 测试cache搜索、全量以及增量写入
    // test_cache();

    // // 测试generate循环
    // test_batching();

    // // 测试prefill flash attention
    // test_fmha2(true);

    // 测试decode阶段attention
    test_decode_attn();
    return 0;
}
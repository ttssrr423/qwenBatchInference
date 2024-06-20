#include "generate.h"
#include "core_gpu.cuh"
#include "forward_gpu.cuh"
#include "kv_cache.h"
#include "cutlass_fmha/xformer_attention.h"
#include "exllama/exllama_liteqwen_ext.h"
#include "sampling.cuh"
#include "sgemm_lora.cuh"

std::mutex dp_locker;

namespace liteqwen{

std::string get_shape_str(std::vector<int> shape) {
    std::vector<std::string> tensor_shape;
        std::string delim = ",";
        for (int64_t di=0; di<(int)(shape.size()); di++){
            int size = shape.at(di);
            tensor_shape.push_back(std::to_string(size));
        }
    std::string shape_print = liteqwen::join(tensor_shape, delim);
    return shape_print;
}

void prepare_freqs_cis(int dim, int max_seq_len, float theta, float* cos_ten, float* sin_ten) {
    int half_dim = dim / 2;
    float* freqs = new float[half_dim];
    for (int i=0; i< dim; i+=2){
        float power_num = static_cast<float>(i) / static_cast<float>(dim);
        float val = (float)1.0 / (pow(theta, power_num)); //inv_freq
        freqs[i/2] = val;
    }
    //theta == freqs;
    for (int t=0; t< max_seq_len; t+=1){
        // cat the out_product(t, inv_freq) twice in the channel dim.
        for (int hd=0; hd<half_dim; hd+=1){
            cos_ten[t * dim + hd] = cos(freqs[hd] * static_cast<float>(t));
            sin_ten[t * dim + hd] = sin(freqs[hd] * static_cast<float>(t));
        }
        for (int hd=0; hd<half_dim; hd+=1){
            cos_ten[t * dim + half_dim + hd] = cos(freqs[hd] * static_cast<float>(t));
            sin_ten[t * dim + half_dim + hd] = sin(freqs[hd] * static_cast<float>(t));
        }
    }
}

void fill_cos_sin_on_gpu(int device_id, int max_seq_len, int channel, Data* cos_tensor, Data* sin_tensor) {
    printf("preparing gpu rotary cos and sin with [seq_length, channel]=[%i, %i] on device %i\n", max_seq_len, channel, device_id);

    float* fp32_cos = new float[max_seq_len*channel];
    float* fp32_sin = new float[max_seq_len*channel];
    float rope_theta = 1000000.0F; // defined in config
    prepare_freqs_cis(channel, max_seq_len, rope_theta, fp32_cos, fp32_sin);
    cos_tensor->Fp32CpuToFp16Upload(device_id, fp32_cos);
    sin_tensor->Fp32CpuToFp16Upload(device_id, fp32_sin);
}

void UploadInputs(int gpu_id, bool is_prefill, const Data& input_ids, const Data& inp_batch_ids, const Data& query_pos_starts, const Data& key_pos_starts, int* cpu_input_ids, uint8_t* cpu_inp_bids, int* cpu_query_starts, int prefill_len, int bsz) {
    int upload_len = is_prefill ? prefill_len : bsz;
    // QuickUploadData(DataType::INT32, (void*)input_ids.cudaData, (uint8_t*)cpu_input_ids, gpu_id, 0, 0, upload_len);
    QuickUploadData(DataType::INT8, (void*)inp_batch_ids.cudaData, (uint8_t*)cpu_inp_bids, gpu_id, 0, 0, prefill_len);
    QuickUploadData(DataType::INT32, (void*)query_pos_starts.cudaData, (uint8_t*)cpu_query_starts, gpu_id, 0, 0, bsz+1);
    QuickUploadData(DataType::INT32, (void*)key_pos_starts.cudaData, (uint8_t*)cpu_query_starts, gpu_id, 0, 0, bsz+1);
}

LoraConfig GetLora(std::string preparer_name, std::string one_req_lora, std::map<std::string, liteqwen::LoraConfig>* lora_meta) {
    LoraConfig lora_cfg;
    if (preparer_name == std::string("SKIP") || preparer_name == std::string("skip")) {
        LoraConfig skip_lora = LoraConfig{std::string("skip"), true, 0.0, 0, std::vector<std::string>()};
        lora_cfg = skip_lora;
    } else if (preparer_name == std::string("ANY")) {
        // preparer lora不限制时，选取任意batch内的样本的lora（同lora会被放入一个batch）
        auto lora_finded = lora_meta->find(one_req_lora);
        if (lora_finded == lora_meta->end()) {
            LoraConfig skip_lora = LoraConfig{std::string("skip"), true, 0.0, 0, std::vector<std::string>()};
            lora_cfg = skip_lora;
        } else {
            lora_cfg = lora_finded->second;
        }
    } else {
        auto lora_finded = lora_meta->find(one_req_lora);
        if (lora_finded == lora_meta->end()) {
            printf("warning: lora_name=%s could not be found, using 'skip' instead.\n", one_req_lora.c_str());
            LoraConfig skip_lora = LoraConfig{std::string("skip"), true, 0.0, 0, std::vector<std::string>()};
            lora_cfg = skip_lora;
        } else {
            lora_cfg = lora_finded->second;
        }                
    }
    return lora_cfg;
}

void forward(Data* logits_ptr, bool is_prefill, StringArray request_ids, int* cpu_inp_ids, int* cpu_query_starts, const Data& input_ids, const Data& inp_batch_ids, const Data& query_pos_starts, const Data& key_pos_starts, int batch_maxlen, int dynamic_bl, const LoraConfig& lora_cfg, Qwen2Params* config_ref, const Data& cos_embed, const Data& sin_embed, std::map<std::string, Data>* weights, std::map<std::string, std::pair<int, uintptr_t>>* quant4_meta, PipelineKVPool* kv_cache_ref, ExecuteTimer* timer) {
    int current_device = config_ref->input_deviceId;
    int hidden_size = config_ref->hidden_size;
    int num_layers = config_ref->num_layers;
    std::vector<int> layer_devices = config_ref->layer2deviceId;
    int attention_heads = config_ref->num_attention_heads;
    int kv_heads = config_ref->num_key_value_heads;
    int channels = config_ref->kv_channels;
    int ffn_hidden_size = config_ref->ffn_hidden_size;
    int vocab_size = config_ref->padded_vocab_size;
    int input_device = layer_devices.front();
    int output_device = layer_devices.back();
    int dynamic_bsz = (int)(request_ids.size());
    int max_B = config_ref->max_dynamic_bsz; // >= dynamic_bsz
    int max_BL = config_ref->max_sequence_length; // >= dynamic_bl
    size_t static_L = is_prefill ? static_cast<size_t>(max_BL) : static_cast<size_t>(max_B); // fixed max lens used in mem allocation to reduce allocation miss rates.
    int dynamic_L = is_prefill ? dynamic_bl : dynamic_bsz; // dynamic lens to size boundaries for kernels and prints.

    // 按照最大可能的numel进行activation的预分配，避免碎片化。dynamic长度必须在dim0。
    Data hidden_state = Data(DataType::FLOAT16, {dynamic_L, hidden_size}, input_device, true);
    hidden_state.Allocate(static_L*hidden_size);
    Data local_cos = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, channels}, input_device, true);
    local_cos.Allocate(static_L*channels);
    Data local_sin = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, channels}, input_device, true);
    local_sin.Allocate(static_L*channels);
    // 为了避免stage切换ToDevice时自动Free，利用浅克隆跳过Free。本体tensor在外层方法(Generate)中Free。
    Data local_bids = Data(inp_batch_ids, true);
    Data local_qstarts = Data(query_pos_starts, true);
    Data local_kstarts = Data(key_pos_starts, true);

    // activation需要用的tensors，先声明，需要显存分配时再Allocate或Reallocate。Reallocate在pipeline parallel切换设备时相当于旧gpu上Free同时新gpu上Allocate
    // 所有dynamic的dim都必须限定在dim0上，才能保证与static_L的数据的strides保持一致。
    Data null_bias;
    Data normalized_hidden = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, hidden_size}, -1, true);
    Data q_proj_layer = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, hidden_size}, -1, true);
    Data k_proj_layer = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, channels*kv_heads}, -1, true);
    Data v_proj_layer = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, channels*kv_heads}, -1, true);
    Data k_proj_layer_tiled = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, channels*attention_heads}, -1, true); // 只在gqa prefill时使用，给cutlass flash attention
    Data v_proj_layer_tiled = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, channels*attention_heads}, -1, true); // 只在gqa prefill时使用，给cutlass flash attention
    Data flash_attn_workspace = Data(DataType::FLOAT32, std::vector<int>{1, static_cast<int>(static_L), attention_heads, channels}, -1, true); // 只在flash attention的output非fp32时被使用
    Data scores = Data(DataType::FLOAT32, std::vector<int>{dynamic_bl, attention_heads}, -1, true); // 只在decode attention时使用，注意并非dynamic_L，而是dynamic_bl。
    // Data scores = Data(DataType::FLOAT16, std::vector<int>{dynamic_bl, attention_heads}, -1, true); // 只在decode attention时使用，注意并非dynamic_L，而是dynamic_bl。
    Data attended_out = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, hidden_size}, -1, true);
    Data dense_out =  Data(DataType::FLOAT16, std::vector<int>{dynamic_L, hidden_size}, -1, true);
    Data post_normalized_hidden = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, hidden_size}, -1, true);
    Data mlp_intermediate = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, ffn_hidden_size}, -1, true);
    Data mlp_gate =  Data(DataType::FLOAT16, std::vector<int>{dynamic_L, ffn_hidden_size}, -1, true);
    Data mlp_act_hidden = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, ffn_hidden_size}, -1, true);
    Data mlp_out = Data(DataType::FLOAT16, std::vector<int>{dynamic_L, hidden_size}, -1, true);
    Data final_step_hidden = Data(DataType::FLOAT16, std::vector<int>{dynamic_bsz, hidden_size}, -1, true);
    Data final_step_normalized = Data(DataType::FLOAT16, std::vector<int>{dynamic_bsz, hidden_size}, -1, true);

    // lora 相关activation，shape根据不同adapter有不同，所以先判定adapter再Init对应的shape
    Data qproj_fp32_inp = Data();
    Data qproj_loraA_hidden = Data();
    Data qproj_loraB_hidden = Data();
    Data kproj_fp32_inp = Data();
    Data kproj_loraA_hidden = Data();
    Data kproj_loraB_hidden = Data();
    Data vproj_fp32_inp = Data();
    Data vproj_loraA_hidden = Data();
    Data vproj_loraB_hidden = Data();
    Data oproj_fp32_inp = Data();
    Data oproj_loraA_hidden = Data();
    Data oproj_loraB_hidden = Data();

    Data down_proj_fp32_inp = Data();
    Data down_proj_A_hidden = Data();
    Data down_proj_B_hidden = Data();
    Data gate_proj_fp32_inp = Data();
    Data gate_proj_A_hidden = Data();
    Data gate_proj_B_hidden = Data();
    Data up_proj_fp32_inp = Data();
    Data up_proj_A_hidden = Data();
    Data up_proj_B_hidden = Data();
    const bool use_fused_lora = true;

    std::string adapter_name = std::string("skip");
    int lora_r = 0;
    float lora_scaling = 0.0f;
    bool* lora_enabled = new bool[7];
    for (int ii=0; ii<7; ii++){
        lora_enabled[ii] = false;
    }
    // printf("forward lora is %s, rank=%i\n", lora_cfg.model_name.c_str(), (int)(lora_cfg.target_modules.size()));
    if (lora_cfg.model_name != std::string("skip") && (int)(lora_cfg.target_modules.size()) > 0) {
        // lora模块限制在以下7层：q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj
        // printf("using lora with %i layers\n", lora_cfg.target_modules.size());
        adapter_name = lora_cfg.model_name;
        lora_r = lora_cfg.r;
        lora_scaling = lora_cfg.lora_alpha / static_cast<float>(lora_r);
        if (lora_r > 0) {
            for (int module_i=0; module_i<lora_cfg.target_modules.size(); module_i++) {
                std::string module_nm = lora_cfg.target_modules[module_i];
                // printf("init lora activation %s\n", module_nm.c_str());
                if (module_nm == std::string("q_proj")) {
                    qproj_fp32_inp.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    qproj_loraA_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, lora_r}, -1, true);
                    qproj_loraB_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    lora_enabled[0] = true;
                } else if (module_nm == std::string("k_proj")) {
                    kproj_fp32_inp.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    kproj_loraA_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, lora_r}, -1, true);
                    kproj_loraB_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    lora_enabled[1] = true;
                } else if (module_nm == std::string("v_proj")) {
                    vproj_fp32_inp.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    vproj_loraA_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, lora_r}, -1, true);
                    vproj_loraB_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    lora_enabled[2] = true;
                } else if (module_nm == std::string("o_proj")) {
                    oproj_fp32_inp.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    oproj_loraA_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, lora_r}, -1, true);
                    oproj_loraB_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    lora_enabled[3] = true;
                } else if (module_nm == std::string("up_proj")) {
                    up_proj_fp32_inp.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    up_proj_A_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, lora_r}, -1, true);
                    up_proj_B_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, ffn_hidden_size}, -1, true);
                    lora_enabled[4] = true;
                } else if (module_nm == std::string("gate_proj")) {
                    gate_proj_fp32_inp.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    gate_proj_A_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, lora_r}, -1, true);
                    gate_proj_B_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, ffn_hidden_size}, -1, true);
                    lora_enabled[5] = true;
                } else if (module_nm == std::string("down_proj")) {
                    down_proj_fp32_inp.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, ffn_hidden_size}, -1, true);
                    down_proj_A_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, lora_r}, -1, true);
                    down_proj_B_hidden.Init(liteqwen::DataType::FLOAT32, std::vector<int>{dynamic_L, hidden_size}, -1, true);
                    lora_enabled[6] = true;
                } else {
                    printf("unexpected lora module, ignoring: %s\n", module_nm);
                    adapter_name = std::string("skip");
                }
            }
        }
    }

    // <=========== 开始forward主流程============
    if (is_prefill)
        timer->regist(std::string("embed_lookup"));
    cpu_embedding_fwd(current_device, hidden_state, cpu_inp_ids, (*weights)[std::string("model.embed_tokens.weight")], 0, dynamic_L, hidden_size);
    rotary_lookup(is_prefill, current_device, local_cos, local_sin, inp_batch_ids, key_pos_starts, cos_embed, sin_embed, 0, dynamic_L, channels, dynamic_bsz);

    bool device_switched = false; // 当前层是否刚切换完gpu
    bool is_last_layer_of_device = true; // 下一层是否会切换到新gpu
    for (int layer_id=0; layer_id<num_layers; layer_id++) {
        int layer_device = layer_devices[layer_id];
        if (layer_id < num_layers-1) {
            is_last_layer_of_device = (layer_devices[layer_id+1] != layer_device);
        } else {
            is_last_layer_of_device = true;
        }
        if (layer_device != current_device) {
            // 多卡pipeline并行推理时，移动至新gpu: [hidden_state, local_cos, local_sin]
            // printf("moving device%i->%i\n", current_device, layer_device);
            hidden_state.ToDevice(layer_device);
            local_cos.ToDevice(layer_device);
            local_sin.ToDevice(layer_device);
            local_bids.ToDevice(layer_device);
            local_qstarts.ToDevice(layer_device);
            local_kstarts.ToDevice(layer_device);
            current_device = layer_device;
            device_switched = true;
        } else {
            device_switched = false;
        }

        // 尽量分配大小是固定的显存块，只在每个device上第一次使用时（即layer_id=0或device_switeched对应的layer_id时），申请显存。layer+1时可以复用显存。
        // 在一个gpu上执行完stage的最后一层layer之后，ToDevice到下一个device上。即将脱离loop（layer_id=num_layers-1）时，也需要释放显存。
        // 这样避免了layer之间不停地malloc以及free，节省运行时间。
        bool should_reallocate = device_switched || layer_id==0;
        bool should_free = is_last_layer_of_device;

        if (should_reallocate) {
            // 一次性计算stage内所有layer下每个样本的kv_cache的start指针。
            kv_cache_ref->scatter_example_ptrs(request_ids, layer_id);
        }

        SetDevice(current_device);
        std::string layer_prefix = std::string("model.layers.")+std::to_string(layer_id)+".";

        if (is_prefill)
            timer->regist(std::string("prenorm")+std::to_string(layer_id));
        
        normalized_hidden.Reallocate(current_device, should_reallocate, static_L*hidden_size);
        rms_norm(normalized_hidden, hidden_state, (*weights)[layer_prefix+std::string("input_layernorm.weight")], dynamic_L, hidden_size);

        if (is_prefill)
            timer->regist(std::string("qkv_linear")+std::to_string(layer_id));

        q_proj_layer.Reallocate(current_device, should_reallocate, static_L*hidden_size);
        if (lora_r > 0 && lora_enabled[0]) {
            // lora名称样例：'base_model.model.model.layers.9.self_attn.v_proj.lora_A.weight'
            std::string qproj_loraA_key = std::string("base_model.model.") + layer_prefix + std::string("self_attn.q_proj.lora_A.") + adapter_name + std::string(".weight");
            std::string qproj_loraB_key = std::string("base_model.model.") + layer_prefix + std::string("self_attn.q_proj.lora_B.") + adapter_name + std::string(".weight");

            if (!use_fused_lora) {
                // 旧cublas lora计算，显存占用过多，弃用。
                qproj_loraA_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                qproj_loraB_hidden.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                qproj_fp32_inp.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                quant4_lora_linear_fwd(q_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.q_proj")].second, (*weights)[layer_prefix+std::string("self_attn.q_proj.bias")], true, qproj_fp32_inp, qproj_loraA_hidden, qproj_loraB_hidden, (*weights)[qproj_loraA_key], (*weights)[qproj_loraB_key], lora_r, lora_scaling);
                qproj_loraA_hidden.Free(should_free);
                qproj_loraB_hidden.Free(should_free);
                qproj_fp32_inp.Free(should_free);
            } else {
                qproj_loraA_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                quant4_lora_linear_fused(q_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.q_proj")].second, (*weights)[layer_prefix+std::string("self_attn.q_proj.bias")], true, qproj_loraA_hidden, (*weights)[qproj_loraA_key], (*weights)[qproj_loraB_key], lora_r, lora_scaling);
                qproj_loraA_hidden.Free(should_free);
            }
        } else {
            quant4_linear_fwd(q_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.q_proj")].second, (*weights)[layer_prefix+std::string("self_attn.q_proj.bias")], true);
        }

        k_proj_layer.Reallocate(current_device, should_reallocate, static_L*channels*kv_heads);
        if (lora_r > 0 && lora_enabled[1]) {
            std::string kproj_loraA_key = std::string("base_model.model.") + layer_prefix + std::string("self_attn.k_proj.lora_A.") + adapter_name + std::string(".weight");
            std::string kproj_loraB_key = std::string("base_model.model.") + layer_prefix + std::string("self_attn.k_proj.lora_B.") + adapter_name + std::string(".weight");

            if (!use_fused_lora) {
                kproj_loraA_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                kproj_loraB_hidden.Reallocate(current_device, should_reallocate, static_L*channels*kv_heads);
                kproj_fp32_inp.Reallocate(current_device, should_reallocate, static_L*channels*kv_heads);
                quant4_lora_linear_fwd(k_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.k_proj")].second, (*weights)[layer_prefix+std::string("self_attn.k_proj.bias")], true, kproj_fp32_inp, kproj_loraA_hidden, kproj_loraB_hidden, (*weights)[kproj_loraA_key], (*weights)[kproj_loraB_key], lora_r, lora_scaling);
                kproj_loraA_hidden.Free(should_free);
                kproj_loraB_hidden.Free(should_free);
                kproj_fp32_inp.Free(should_free);
            } else {
                kproj_loraA_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                quant4_lora_linear_fused(k_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.k_proj")].second, (*weights)[layer_prefix+std::string("self_attn.k_proj.bias")], true, kproj_loraA_hidden, (*weights)[kproj_loraA_key], (*weights)[kproj_loraB_key], lora_r, lora_scaling);
                kproj_loraA_hidden.Free(should_free);
            }
        } else {
            quant4_linear_fwd(k_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.k_proj")].second, (*weights)[layer_prefix+std::string("self_attn.k_proj.bias")], true);
        }

        v_proj_layer.Reallocate(current_device, should_reallocate, static_L*channels*kv_heads);
        if (lora_r > 0 && lora_enabled[2]) {
            std::string vproj_loraA_key = std::string("base_model.model.") + layer_prefix + std::string("self_attn.v_proj.lora_A.") + adapter_name + std::string(".weight");
            std::string vproj_loraB_key = std::string("base_model.model.") + layer_prefix + std::string("self_attn.v_proj.lora_B.") + adapter_name + std::string(".weight");

            if (!use_fused_lora) {
                vproj_loraA_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                vproj_loraB_hidden.Reallocate(current_device, should_reallocate, static_L*channels*kv_heads);
                vproj_fp32_inp.Reallocate(current_device, should_reallocate, static_L*channels*kv_heads);
                quant4_lora_linear_fwd(v_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.v_proj")].second, (*weights)[layer_prefix+std::string("self_attn.v_proj.bias")], true, vproj_fp32_inp, vproj_loraA_hidden, vproj_loraB_hidden, (*weights)[vproj_loraA_key], (*weights)[vproj_loraB_key], lora_r, lora_scaling);
                vproj_loraA_hidden.Free(should_free);
                vproj_loraB_hidden.Free(should_free);
                vproj_fp32_inp.Free(should_free);
            } else {
                vproj_loraA_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                quant4_lora_linear_fused(v_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.v_proj")].second, (*weights)[layer_prefix+std::string("self_attn.v_proj.bias")], true, vproj_loraA_hidden, (*weights)[vproj_loraA_key], (*weights)[vproj_loraB_key], lora_r, lora_scaling);
                vproj_loraA_hidden.Free(should_free);
            }
        } else {
            quant4_linear_fwd(v_proj_layer, normalized_hidden, (*quant4_meta)[layer_prefix+std::string("self_attn.v_proj")].second, (*weights)[layer_prefix+std::string("self_attn.v_proj.bias")], true);
        }

        normalized_hidden.Free(should_free);

        if (is_prefill)
            timer->regist(std::string("apply_rotary")+std::to_string(layer_id));
        apply_rotary_embeddings(q_proj_layer, k_proj_layer, dynamic_bsz, dynamic_L, hidden_size, attention_heads, local_cos, local_sin);

        // 全量或增量写入预分配的KV缓存地址中。
        if (is_prefill) {
            timer->regist(std::string("cat_pasts")+std::to_string(layer_id));
            kv_cache_ref->write_example_kvs_to_cache(true, request_ids.size(), layer_id, std::pair<Data*, Data*>(&k_proj_layer, &v_proj_layer), local_kstarts, local_bids, max_B, dynamic_bl, kv_heads, channels);
        } else {
            // decode step, append the dynamic_L=dynamic_bsz examples' incremental KV.
            kv_cache_ref->write_example_kvs_to_cache(false, request_ids.size(), layer_id, std::pair<Data*, Data*>(&k_proj_layer, &v_proj_layer), local_kstarts, local_bids, max_B, dynamic_bl, kv_heads, channels);
        }

        if (is_prefill) {
            timer->regist(std::string("core_attn")+std::to_string(layer_id));
            // prefill flash attention
            if (attention_heads != kv_heads) {
                // printf("TODO: should add a tile copy kernel to tile/repeat kv_caches to make flash attention work if using GQA (i.e. attention_heads > kv_heads in qwen 32B models)\n");
                // TODO: 未经测试代码
                k_proj_layer_tiled.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                v_proj_layer_tiled.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                dynamic_gqa_tile(k_proj_layer_tiled, v_proj_layer_tiled, k_proj_layer, v_proj_layer, dynamic_bl, attention_heads, kv_heads, channels);

                k_proj_layer.Free(should_free);
                v_proj_layer.Free(should_free);
            } // else for qwen14b, k_proj_layer already in shape [BL, H, D], data can be directly used in flash attention.

            flash_attn_workspace.Reallocate(current_device, should_reallocate, static_L*hidden_size);
            attended_out.Reallocate(current_device, should_reallocate, static_L*hidden_size);
            Data null_seqlen = Data(DataType::INT32);

            if (attention_heads != kv_heads) {
                xformer_self_attention_fwd(attended_out, q_proj_layer, k_proj_layer_tiled, v_proj_layer_tiled, local_qstarts, local_kstarts, null_seqlen, flash_attn_workspace, dynamic_bsz, dynamic_bl, attention_heads, batch_maxlen, batch_maxlen, channels);
            } else {
                xformer_self_attention_fwd(attended_out, q_proj_layer, k_proj_layer, v_proj_layer, local_qstarts, local_kstarts, null_seqlen, flash_attn_workspace, dynamic_bsz, dynamic_bl, attention_heads, batch_maxlen, batch_maxlen, channels);
            }
            q_proj_layer.Free(should_free);
            if (attention_heads != kv_heads) {
                k_proj_layer_tiled.Free(should_free);
                v_proj_layer_tiled.Free(should_free);
            } else {
                k_proj_layer.Free(should_free);
                v_proj_layer.Free(should_free);
            }
            flash_attn_workspace.Free(should_free);
        } else {
            k_proj_layer.Free(should_free);
            v_proj_layer.Free(should_free);

            scores.Reallocate(current_device, should_reallocate, max_BL*hidden_size);
            attended_out.Reallocate(current_device, should_reallocate, static_L*hidden_size);

            // if (layer_id == 0) {
            //     for (int _bi=0; _bi<request_ids.size(); _bi++) {
            //         std::string req_id = request_ids[_bi];
            //         // kv_cache_ref->print_cache(req_id, layer_id, false, example_inp_len);
            //         kv_cache_ref->print_cache(req_id, layer_id, true, example_inp_len);
            //     }
            // }

            decode_attention(attended_out, scores, request_ids, dynamic_bl, batch_maxlen, layer_id, q_proj_layer, local_bids, local_qstarts, max_B, kv_cache_ref, kv_heads, channels);
            // if (layer_id == 0 || layer_id == num_layers-1) {
            //     attended_out.print(std::string("decode_attended")+std::to_string(layer_id));
            // }
            q_proj_layer.Free(should_free);
            scores.Free(should_free);
        }

        if (is_prefill)
            timer->regist(std::string("dense_linear")+std::to_string(layer_id));
        
        dense_out.Reallocate(current_device, should_reallocate, static_L*hidden_size);
        if (lora_r > 0 && lora_enabled[3]) {
            std::string oproj_loraA_key = std::string("base_model.model.") + layer_prefix + std::string("self_attn.o_proj.lora_A.") + adapter_name + std::string(".weight");
            std::string oproj_loraB_key = std::string("base_model.model.") + layer_prefix + std::string("self_attn.o_proj.lora_B.") + adapter_name + std::string(".weight");

            if (!use_fused_lora) {
                oproj_loraA_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                oproj_loraB_hidden.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                oproj_fp32_inp.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                quant4_lora_linear_fwd(dense_out, attended_out, (*quant4_meta)[layer_prefix+std::string("self_attn.o_proj")].second, (*weights)[layer_prefix+std::string("self_attn.o_proj.bias")], true, oproj_fp32_inp, oproj_loraA_hidden, oproj_loraB_hidden, (*weights)[oproj_loraA_key], (*weights)[oproj_loraB_key], lora_r, lora_scaling);
                oproj_loraA_hidden.Free(should_free);
                oproj_loraB_hidden.Free(should_free);
                oproj_fp32_inp.Free(should_free);
            } else {
                oproj_loraA_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                quant4_lora_linear_fused(dense_out, attended_out, (*quant4_meta)[layer_prefix+std::string("self_attn.o_proj")].second, (*weights)[layer_prefix+std::string("self_attn.o_proj.bias")], true, oproj_loraA_hidden, (*weights)[oproj_loraA_key], (*weights)[oproj_loraB_key], lora_r, lora_scaling);
                oproj_loraA_hidden.Free(should_free);
            }
        } else {
            quant4_linear_fwd(dense_out, attended_out, (*quant4_meta)[layer_prefix+std::string("self_attn.o_proj")].second, (*weights)[layer_prefix+std::string("self_attn.o_proj.bias")], true);
        }

        attended_out.Free(should_free);
        if (is_prefill)
            timer->regist(std::string("residual_one")+std::to_string(layer_id));
        inplace_add_half(hidden_state, dense_out, static_cast<size_t>(dynamic_L)*hidden_size);
        dense_out.Free(should_free);

        if (is_prefill)
            timer->regist(std::string("post_ln")+std::to_string(layer_id));
        post_normalized_hidden.Reallocate(current_device, should_reallocate, static_L*hidden_size);
        rms_norm(post_normalized_hidden, hidden_state, (*weights)[layer_prefix+std::string("post_attention_layernorm.weight")], dynamic_L, hidden_size);

        if (is_prefill)
            timer->regist(std::string("mlp_first")+std::to_string(layer_id));
        mlp_intermediate.Reallocate(current_device, should_reallocate, static_L*ffn_hidden_size);
        if (lora_r > 0 && lora_enabled[4]) {
            std::string up_proj_A_key = std::string("base_model.model.") + layer_prefix + std::string("mlp.up_proj.lora_A.") + adapter_name + std::string(".weight");
            std::string up_proj_B_key = std::string("base_model.model.") + layer_prefix + std::string("mlp.up_proj.lora_B.") + adapter_name + std::string(".weight");

            if (!use_fused_lora) {
                up_proj_A_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                up_proj_B_hidden.Reallocate(current_device, should_reallocate, static_L*ffn_hidden_size);
                up_proj_fp32_inp.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                quant4_lora_linear_fwd(mlp_intermediate, post_normalized_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.up_proj")].second, (*weights)[layer_prefix+std::string("mlp.up_proj.bias")], true, up_proj_fp32_inp, up_proj_A_hidden, up_proj_B_hidden, (*weights)[up_proj_A_key], (*weights)[up_proj_B_key], lora_r, lora_scaling);
                up_proj_A_hidden.Free(should_free);
                up_proj_B_hidden.Free(should_free);
                up_proj_fp32_inp.Free(should_free);
            } else {
                up_proj_A_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                quant4_lora_linear_fused(mlp_intermediate, post_normalized_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.up_proj")].second, (*weights)[layer_prefix+std::string("mlp.up_proj.bias")], true, up_proj_A_hidden, (*weights)[up_proj_A_key], (*weights)[up_proj_B_key], lora_r, lora_scaling);
                up_proj_A_hidden.Free(should_free);
            }
        } else {
            quant4_linear_fwd(mlp_intermediate, post_normalized_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.up_proj")].second, (*weights)[layer_prefix+std::string("mlp.up_proj.bias")], true);
        }

        mlp_gate.Reallocate(current_device, should_reallocate, static_L*ffn_hidden_size);
        if (lora_r > 0 && lora_enabled[5]) {
            std::string gate_proj_A_key = std::string("base_model.model.") + layer_prefix + std::string("mlp.gate_proj.lora_A.") + adapter_name + std::string(".weight");
            std::string gate_proj_B_key = std::string("base_model.model.") + layer_prefix + std::string("mlp.gate_proj.lora_B.") + adapter_name + std::string(".weight");

            if (!use_fused_lora) {
                gate_proj_A_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                gate_proj_B_hidden.Reallocate(current_device, should_reallocate, static_L*ffn_hidden_size);
                gate_proj_fp32_inp.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                quant4_lora_linear_fwd(mlp_gate, post_normalized_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.gate_proj")].second, (*weights)[layer_prefix+std::string("mlp.gate_proj.bias")], true, gate_proj_fp32_inp, gate_proj_A_hidden, gate_proj_B_hidden, (*weights)[gate_proj_A_key], (*weights)[gate_proj_B_key], lora_r, lora_scaling);
                gate_proj_A_hidden.Free(should_free);
                gate_proj_B_hidden.Free(should_free);
                gate_proj_fp32_inp.Free(should_free);
            } else {
                gate_proj_A_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                quant4_lora_linear_fused(mlp_gate, post_normalized_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.gate_proj")].second, (*weights)[layer_prefix+std::string("mlp.gate_proj.bias")], true, gate_proj_A_hidden, (*weights)[gate_proj_A_key], (*weights)[gate_proj_B_key], lora_r, lora_scaling);
                gate_proj_A_hidden.Free(should_free);
            }
        } else {
            quant4_linear_fwd(mlp_gate, post_normalized_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.gate_proj")].second, (*weights)[layer_prefix+std::string("mlp.gate_proj.bias")], true);
        }

        if (is_prefill)
            timer->regist(std::string("mlp_gated_silu")+std::to_string(layer_id));
        mlp_act_hidden.Reallocate(current_device, should_reallocate, static_L*ffn_hidden_size);
        qwen_silu(mlp_act_hidden, mlp_gate, mlp_intermediate, dynamic_L, ffn_hidden_size);
        post_normalized_hidden.Free(should_free);
        mlp_intermediate.Free(should_free);
        mlp_gate.Free(should_free);

        if (is_prefill)
            timer->regist(std::string("mlp_second")+std::to_string(layer_id));
        mlp_out.Reallocate(current_device, should_reallocate, static_L*hidden_size);
        if (lora_r > 0 && lora_enabled[6]) {
            std::string down_proj_A_key = std::string("base_model.model.") + layer_prefix + std::string("mlp.down_proj.lora_A.") + adapter_name + std::string(".weight");
            std::string down_proj_B_key = std::string("base_model.model.") + layer_prefix + std::string("mlp.down_proj.lora_B.") + adapter_name + std::string(".weight");

            if (!use_fused_lora) {
                down_proj_A_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                down_proj_B_hidden.Reallocate(current_device, should_reallocate, static_L*hidden_size);
                down_proj_fp32_inp.Reallocate(current_device, should_reallocate, static_L*ffn_hidden_size);
                quant4_lora_linear_fwd(mlp_out, mlp_act_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.down_proj")].second, (*weights)[layer_prefix+std::string("mlp.down_proj.bias")], true, down_proj_fp32_inp, down_proj_A_hidden, down_proj_B_hidden, (*weights)[down_proj_A_key], (*weights)[down_proj_B_key], lora_r, lora_scaling);
                down_proj_A_hidden.Free(should_free);
                down_proj_B_hidden.Free(should_free);
                down_proj_fp32_inp.Free(should_free);
            } else {
                down_proj_A_hidden.Reallocate(current_device, should_reallocate, static_L*lora_r);
                quant4_lora_linear_fused(mlp_out, mlp_act_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.down_proj")].second, (*weights)[layer_prefix+std::string("mlp.down_proj.bias")], true, down_proj_A_hidden, (*weights)[down_proj_A_key], (*weights)[down_proj_B_key], lora_r, lora_scaling);
                down_proj_A_hidden.Free(should_free);
            }
        } else {
            quant4_linear_fwd(mlp_out, mlp_act_hidden, (*quant4_meta)[layer_prefix+std::string("mlp.down_proj")].second, (*weights)[layer_prefix+std::string("mlp.down_proj.bias")], true);
        }

        mlp_act_hidden.Free(should_free);

        if (is_prefill)
            timer->regist(std::string("residual_two")+std::to_string(layer_id));
        inplace_add_half(hidden_state, mlp_out, dynamic_L*hidden_size);
        mlp_out.Free(should_free);

    }

    if (is_prefill) {
        timer->regist(std::string("final_ln"));
        final_step_hidden.Reallocate(current_device, true, static_cast<size_t>(max_B)*hidden_size);
        dynamic_slice_last(final_step_hidden, hidden_state, local_qstarts, dynamic_bsz, hidden_size);
        hidden_state.Free();
        final_step_normalized.Reallocate(current_device, true, static_cast<size_t>(max_B)*hidden_size);
        rms_norm(final_step_normalized, final_step_hidden, (*weights)[std::string("model.norm.weight")], dynamic_bsz, hidden_size);
        final_step_hidden.Free();
    } else {
        final_step_normalized.Reallocate(current_device, true, static_cast<size_t>(max_B)*hidden_size);
        rms_norm(final_step_normalized, hidden_state, (*weights)[std::string("model.norm.weight")], dynamic_bsz, hidden_size);
        hidden_state.Free();
    }
    local_cos.Free();
    local_sin.Free();
    local_bids.Free();
    local_qstarts.Free();
    local_kstarts.Free();
    
    if (current_device != output_device) {
        final_step_normalized.ToDevice(output_device);
        current_device = output_device;
    }

    if (is_prefill)
        timer->regist(std::string("logit_linear"));
    
    linear_fwd(*logits_ptr, final_step_normalized, (*weights)[std::string("lm_head.weight")], null_bias, dynamic_bsz, vocab_size, hidden_size, false);
    final_step_normalized.Free();

    // logits_ptr->print(std::string("logits"));
    if (is_prefill)
        timer->regist(std::string("end"));
}

size_t expanded_numel(size_t numel) {
    // 让数据间隔出至少4个单位，且numel设置为32的整数倍。
    return (((numel + 4) - 1) / 32 + 1) * 32;
}


void Generate(int data_id, std::shared_ptr<ContextPool> pool, Qwen2Params model_param, std::map<std::string, Data*>* cpu_weights_ptr, std::map<std::string, liteqwen::LoraConfig>* lora_meta, std::map<std::string, liteqwen::Q4LinearMeta>* quant_meta) {

    Qwen2Params qwen2_param = model_param;
    qwen2_param.update_data_id(data_id);
    int input_device = qwen2_param.input_deviceId;
    int output_device = qwen2_param.output_deviceId;
    int vocab_size = qwen2_param.padded_vocab_size;
    int usual_eos_id = qwen2_param.eos_ids[0]; // 强制eos时使用。
    dp_locker.lock();
    DeviceSynchronize();
    printf("setting up inference on data_id=%i, using devices=[%i, %i), num_layers=%i, max_dynamic_bsz=%i, max_BL=%i\n", data_id, input_device, output_device+1, (int)(qwen2_param.layer2deviceId.size()), qwen2_param.max_dynamic_bsz, qwen2_param.max_sequence_length);

    // 依次初始化每块gpu上的tensors和states
    for (int dev_id=qwen2_param.input_deviceId; dev_id<qwen2_param.output_deviceId+1; dev_id++) {
        setup_gpu_cublas_handler(dev_id);
    }
    DeviceSynchronize();
    // 初始化cutlass所需的device_property
    init_device_property_for_device(qwen2_param.world_size, static_cast<int64_t>(qwen2_param.input_deviceId), static_cast<int64_t>(qwen2_param.output_deviceId+1));
    DeviceSynchronize();

    // 初始化kv_cache
    printf("allocating kv-cache pools for data_id=%i, size=[BL(%i), Hkv(%i)*D(%i)] * layer_num(%i in %i stages) * 2(KV)\n", data_id, qwen2_param.max_sequence_length, qwen2_param.num_key_value_heads, qwen2_param.kv_channels, qwen2_param.num_layers, qwen2_param.pp_size);
    PipelineKVPool* kv_cache_ref = new PipelineKVPool(qwen2_param.max_dynamic_bsz, qwen2_param.max_sequence_length, qwen2_param.num_key_value_heads * qwen2_param.kv_channels, qwen2_param.layer2deviceId);
    BatchInputPreparer* batch_inp_preparer = new BatchInputPreparer(qwen2_param.max_dynamic_bsz, qwen2_param.max_sequence_length);
    DeviceSynchronize();
    printf("input preparer initialized...allocating cos & sin, max_len=%i, kv_channels=%i, gpu_id=%i\n", qwen2_param.max_sequence_length, qwen2_param.kv_channels, input_device);

    // 初始化rotary
    Data full_cos = Data(DataType::FLOAT16, std::vector<int>{qwen2_param.max_sequence_length, qwen2_param.kv_channels}, input_device, false);
    full_cos.Allocate();
    Data full_sin = Data(DataType::FLOAT16, std::vector<int>{qwen2_param.max_sequence_length, qwen2_param.kv_channels}, input_device, false);
    full_sin.Allocate();
    fill_cos_sin_on_gpu(input_device, qwen2_param.max_sequence_length, qwen2_param.kv_channels, &full_cos, &full_sin);
    DeviceSynchronize();
    printf("data_id=%i loading all cpu weights to gpus according to device map.\n", data_id);

    std::map<std::string, Data> weights;
    std::map<std::string, int> quant_device_map;
    std::map<int, int> max_dq_buffer_size_map; // gptq缓存空间统计
    std::map<int, int> max_inner_outer_dim_map; // gptq缓存空间统计
    for (int dev_id=input_device; dev_id<output_device+1; dev_id++) {
        max_dq_buffer_size_map[dev_id] = 1;
        max_inner_outer_dim_map[dev_id] = 1;
    }

    for (auto w_iter=cpu_weights_ptr->begin(); w_iter != cpu_weights_ptr->end(); ++w_iter) {
        std::string w_key = w_iter->first;
        int weight_device = qwen2_param.get_name2device(w_key);
        Data* w_cpu_p = w_iter->second;

        bool is_gptq_param = false;
        ParamLocation location;
        for (auto quant_iter=quant_meta->begin(); quant_iter != quant_meta->end(); ++quant_iter) {
            // 匹配到任意GPTQ层参数，则按照规则copy到GPU或CPU. 其中QUANT_BUFFER需要统计最大所需的buffer值。
            (quant_iter->second).get_store_location(&location, w_key, qwen2_param.gptq_desc);
            if (location == ParamLocation::MISS) {
                continue;
            }
            is_gptq_param = true;
            if (location == ParamLocation::QUANT_BUFFER) {
                std::string prefix = quant_iter->first;
                quant_device_map[prefix] = weight_device;
                int max_dq_record = max_dq_buffer_size_map.find(weight_device)->second;
                max_dq_buffer_size_map[weight_device] = std::max(max_dq_record, (int)(w_cpu_p->numel()) * 8);
                if (qwen2_param.gptq_desc) {
                    int max_dim = std::max(quant_iter->second.in_features, quant_iter->second.out_features);
                    int max_record = max_inner_outer_dim_map.find(weight_device)->second;
                    max_inner_outer_dim_map[weight_device] = std::max(max_dim, max_record);
                }
                break;
            } else if (location == ParamLocation::CPU || location == ParamLocation::GPU || location == ParamLocation::IGNORE || location==ParamLocation::EMPTY) {
                break;
            } // else MISS
        }

        if (!is_gptq_param) {
            if (w_key == std::string("model.embed_tokens.weight")) {
                location = ParamLocation::CPU;
            } else {
                location = ParamLocation::GPU;
            }
        }  
        // else {// else gptq param location has already been calculated
        //     printf("q4 param: %s, location=%i on device %i\n", w_key.c_str(), (int)(location), weight_device);
        // }

        if (location==ParamLocation::CPU) {
            weights[w_key] = Data(w_cpu_p->dtype, w_cpu_p->shape, -1, false);
            weights[w_key].Allocate();
            weights[w_key].CopyFrom(*w_cpu_p, false); // deep copy
            printf("offloaded param %s to cpu\n", w_key.c_str());
        } else if (location == ParamLocation::GPU || location == ParamLocation::QUANT_BUFFER) {
            weights[w_key] = Data(w_cpu_p->dtype, w_cpu_p->shape, weight_device, false);
            weights[w_key].Allocate();
            if (w_cpu_p->dtype == DataType::FLOAT32) {
                weights[w_key].UploadValues(w_cpu_p->numel(), 0, w_cpu_p->cpuData, DataType::FLOAT32);
            } else if (w_cpu_p->dtype == DataType::FLOAT16) {
                weights[w_key].UploadValues(w_cpu_p->numel(), 0, w_cpu_p->cpuData, DataType::FLOAT16);
            } else if (w_cpu_p->dtype == DataType::INT32) {
                weights[w_key].UploadValues(w_cpu_p->numel(), 0, w_cpu_p->cpuData, DataType::INT32);
            } else {
                printf("TODO: not implemented upload of other dtypes");
                throw("not implemented error");
            }
        }
        else if (location == ParamLocation::EMPTY) {
            weights[w_key] = Data(w_cpu_p->dtype);
        } 
        else { // else IGNORE
            printf("skipping upload param: %s\n", w_key.c_str());
        }
        // printf("added weight %s to gpu %i, shape=[%s]\n", w_key.c_str(), weight_device, get_shape_str(w_cpu_p->shape).c_str());
        DeviceSynchronize();
    }
    SetEmbeddingBuffer(qwen2_param.max_sequence_length, qwen2_param.hidden_size);
    DeviceSynchronize();

    // 初始化gptq buffer
    std::map<int, Data> gptq_temp_state_map;
    std::map<int, Data> gptq_temp_dq_map;
    for (int layer_device=input_device; layer_device<output_device+1; layer_device++) {
        SetDevice(layer_device);
        int local_max_inner_outer_dim = max_inner_outer_dim_map.find(layer_device)->second;
        gptq_temp_state_map[layer_device] = Data(DataType::FLOAT16, std::vector<int>{qwen2_param.max_sequence_length, local_max_inner_outer_dim}, layer_device, false);
        gptq_temp_state_map[layer_device].Allocate();
        int local_max_dq_buffer_size =  max_dq_buffer_size_map.find(layer_device)->second;
        gptq_temp_dq_map[layer_device] = Data(DataType::FLOAT16, std::vector<int>{1, local_max_dq_buffer_size}, layer_device, false);
        gptq_temp_dq_map[layer_device].Allocate();
        DeviceSynchronize();
        printf("before buffer preparing\n");
        prepare_buffers(layer_device, gptq_temp_state_map[layer_device], gptq_temp_dq_map[layer_device]);
        printf("allocated gptq buffers on device %i: state[%i, %i], dq=[1, %i]. starting make_q4\n", layer_device, qwen2_param.max_sequence_length, local_max_inner_outer_dim, local_max_dq_buffer_size);
        DeviceSynchronize();
    }

    // 初始化4bit quant层
    std::map<std::string, std::pair<int, uintptr_t>> quant_ref_dict;
    for (auto quant_iter=quant_meta->begin(); quant_iter != quant_meta->end(); ++quant_iter) {
        std::string prefix = quant_iter->first;
        int quant_device = quant_device_map[prefix];
        SetDevice(quant_device);
        uintptr_t q4_ref = make_q4(weights[prefix+std::string(".qweight")], weights[prefix+std::string(".qzeros")], weights[prefix+std::string(".scales")], weights[prefix+std::string(".g_idx")], quant_device);
        quant_ref_dict[prefix] = std::make_pair(quant_device, q4_ref);
    }
    printf("quant weights prepared for data_id=%i\n", data_id);
    DeviceSynchronize();
    

    // 初始化Inputs，为了防止cudaMemcpy上传数据时可能发生的覆盖问题，稍微扩大数据间隔。这里声明的cpu数据不一定是上传源，正常上传源是preparer里的vector内的数据。
    SetDevice(input_device);
    Data input_ids = Data(DataType::INT32, std::vector<int>{qwen2_param.max_sequence_length}, input_device, false);
    input_ids.Allocate(expanded_numel(input_ids.numel())); // 
    Data inp_batch_ids = liteqwen::Data(liteqwen::DataType::INT8, std::vector<int>{qwen2_param.max_sequence_length}, input_device, false); // 每个input_id对应的batch_id
    inp_batch_ids.Allocate(expanded_numel(inp_batch_ids.numel())); //
    // flash_attention的pos_starts需要分开key与val才能避免干扰，所以需要两个内容一样的tensor。
    Data key_pos_starts = liteqwen::Data(liteqwen::DataType::INT32, std::vector<int>{qwen2_param.max_dynamic_bsz+1}, input_device, false);
    key_pos_starts.Allocate(expanded_numel(key_pos_starts.numel())); // 
    Data query_pos_starts = liteqwen::Data(liteqwen::DataType::INT32, std::vector<int>{qwen2_param.max_dynamic_bsz+1}, input_device, false);
    query_pos_starts.Allocate(expanded_numel(query_pos_starts.numel())); // 
    int* cpu_input_ids = new int[qwen2_param.max_sequence_length];
    uint8_t* cpu_inp_bids = new uint8_t[qwen2_param.max_sequence_length];
    int* cpu_query_starts = new int[qwen2_param.max_dynamic_bsz+1];
    SetDevice(output_device);
    Data seeds_tensor = liteqwen::Data(liteqwen::DataType::INT32, std::vector<int>{qwen2_param.max_dynamic_bsz}, output_device, false);
    seeds_tensor.Allocate(expanded_numel(seeds_tensor.numel())); // 
    Data temperature = liteqwen::Data(liteqwen::DataType::FLOAT32, std::vector<int>{qwen2_param.max_dynamic_bsz}, output_device, false);
    temperature.Allocate(expanded_numel(temperature.numel())); // 
    Data top_ps = Data(DataType::FLOAT32, std::vector<int>{qwen2_param.max_dynamic_bsz}, output_device, false);
    top_ps.Allocate(expanded_numel(top_ps.numel())); //
    int* cpu_seeds = new int[qwen2_param.max_dynamic_bsz];
    float* cpu_temperatures = new float[qwen2_param.max_dynamic_bsz];
    float* cpu_top_ps = new float[qwen2_param.max_dynamic_bsz];

    const int sample_grid_size = 32;
    const int sample_block_size = 32;
    int top_k = qwen2_param.top_k; // = 32. in case different requests has different top_ks, we use a global top_k to make more convenient inference.
    // 初始化其他activation
    SetDevice(output_device);
    Data logits_last = Data(DataType::FLOAT16, std::vector<int>{qwen2_param.max_dynamic_bsz, vocab_size}, output_device, false);
    logits_last.Allocate();
    Data sampled_id = Data(DataType::INT32, std::vector<int>{qwen2_param.max_dynamic_bsz}, output_device, false);
    sampled_id.Allocate();
    Data logitsFp32 = Data(DataType::FLOAT32, std::vector<int>{qwen2_param.max_dynamic_bsz, vocab_size}, output_device, false);
    logitsFp32.Allocate();

    Data _1_pass_result = Data(DataType::FLOAT32, std::vector<int>{qwen2_param.max_dynamic_bsz, top_k, sample_grid_size}, output_device, false);
    _1_pass_result.Allocate();
    Data _1_psss_indices = Data(DataType::INT32, std::vector<int>{qwen2_param.max_dynamic_bsz, top_k, sample_grid_size}, output_device, false);
    _1_psss_indices.Allocate();
    Data gpu_top_logits = Data(DataType::FLOAT32, std::vector<int>{qwen2_param.max_dynamic_bsz, top_k}, output_device, false);
    gpu_top_logits.Allocate();
    Data gpu_top_indices = Data(DataType::INT32, std::vector<int>{qwen2_param.max_dynamic_bsz, top_k}, output_device, false);
    gpu_top_indices.Allocate();
    Data sample_softmax_out = Data(DataType::FLOAT32, std::vector<int>{qwen2_param.max_dynamic_bsz, top_k}, output_device, false);
    sample_softmax_out.Allocate();
    // Data gpu_top_logits_fp32 = Data(DataType::FLOAT32, std::vector<int>{qwen2_param.max_dynamic_bsz, top_k}, output_device, false);
    // gpu_top_logits_fp32.Allocate();
    int* top_batch_idx = new int[qwen2_param.max_dynamic_bsz * top_k];
    float* top_batch_lgts = new float[qwen2_param.max_dynamic_bsz * top_k];
    bool* cpu_return_logits = new bool[qwen2_param.max_dynamic_bsz];

    int* eos_ids = new int[qwen2_param.eos_token_num];
    for (int eos_pos=0; eos_pos<qwen2_param.eos_token_num; eos_pos++) {
        eos_ids[eos_pos] = qwen2_param.eos_ids[eos_pos]; 
    }
    int* cpu_sampled_id = new int[qwen2_param.max_dynamic_bsz];

    // warmup之前需要先init curand，里面有cudaMalloc，先声明，防止显存踩踏。
    gpu_curand_init(output_device, qwen2_param.world_size, qwen2_param.max_dynamic_bsz, 0, seeds_tensor, cpu_seeds);
    DeviceSynchronize();

    for (int dev_id=input_device; dev_id<=output_device; dev_id++) {
        print_cuda_info(input_device);
    }
    DeviceSynchronize();
    dp_locker.unlock();

    LoraConfig skip_lora = LoraConfig{std::string("skip"), true, 0.0, 0, std::vector<std::string>()};
    // ======= warmup运行，预分配会被统计，加速后续推理。============
    printf("warmup prefilling...\n");
    SetDevice(input_device);
    ExecuteTimer dummy_timer;
    dummy_timer.disable();
    int warmup_prefill_len = (int)warmup_ids.size();
    int warmup_maxlen = warmup_prefill_len + 20;
    int warmup_prefill_len2 = (int)warmup_ids2.size();
    StringArray warmup_reqs;
    warmup_reqs.Init(512, qwen2_param.max_dynamic_bsz);
    warmup_reqs.push_back(std::string("warmup0001"));
    warmup_reqs.push_back(std::string("warmup0002"));
    LoraConfig warmup_lora = skip_lora;
    if ((int)(lora_meta->size()) > 0) {
        warmup_lora = (lora_meta->begin()++)->second;
    } else {
        warmup_lora = skip_lora;
    }
    printf("warming up for data_id=%i, lora detected=%s, prefill_len=%i ...\n", data_id, warmup_lora.model_name.c_str(), warmup_prefill_len);
    std::vector<AllocateParam> warmup_allocs;
    AllocateParam warmup_kv_param = ((kv_cache_ref->pipeline_caches.find(0))->second) ->search_block_sequence(std::string("warmup0001"), warmup_maxlen, &warmup_allocs);
    warmup_allocs.push_back(warmup_kv_param);
    AllocateParam warmup_kv_param2 = ((kv_cache_ref->pipeline_caches.find(0))->second) ->search_block_sequence(std::string("warmup0002"), warmup_prefill_len2+20, &warmup_allocs);
    warmup_allocs.push_back(warmup_kv_param2);
    kv_cache_ref->sequence_allocate_cache(warmup_allocs);
    int batch_maxlen_ = warmup_prefill_len;
    int dynamic_bsz_ = 2;
    int dynamic_bl_ = warmup_prefill_len + warmup_prefill_len2;
    for (int warmt=0; warmt < warmup_prefill_len; warmt++) {
        cpu_inp_bids[warmt] = static_cast<uint8_t>(0);
        cpu_input_ids[warmt] = warmup_ids[warmt];
    }
    for (int warmt=0; warmt <warmup_prefill_len2; warmt++) {
        cpu_inp_bids[warmt+warmup_prefill_len] = static_cast<uint8_t>(1);
        cpu_input_ids[warmt+warmup_prefill_len] = warmup_ids2[warmt];
    }
    cpu_query_starts[0] = 0;
    cpu_query_starts[1] = warmup_prefill_len;
    cpu_query_starts[2] = dynamic_bl_;
    UploadInputs(input_device, true, input_ids, inp_batch_ids, query_pos_starts, key_pos_starts, cpu_input_ids, cpu_inp_bids, cpu_query_starts, dynamic_bl_, dynamic_bsz_);
    forward(&logits_last, true, warmup_reqs, cpu_input_ids, cpu_query_starts, input_ids, inp_batch_ids, query_pos_starts, key_pos_starts, batch_maxlen_, dynamic_bl_, warmup_lora, &qwen2_param, full_cos, full_sin, &weights, &quant_ref_dict, kv_cache_ref, &dummy_timer);
    DeviceSynchronize();
    // --- warmup decode ---
    printf("warmup decoding...\n");
    int bids_accu = 0;
    std::vector<int> decode_lens_{warmup_prefill_len+1, warmup_prefill_len2+1};
    for (int bi_=0; bi_<dynamic_bsz_; bi_++) {
        cpu_input_ids[bi_] = 108386;
        for (int ti=0; ti<decode_lens_[bi_]; ti++) {
            cpu_inp_bids[bids_accu] = static_cast<uint8_t>(bi_);
            bids_accu++;
        }
        cpu_query_starts[bi_+1] = cpu_query_starts[bi_+1]+(bi_+1);
    }
    UploadInputs(input_device, false, input_ids, inp_batch_ids, query_pos_starts, key_pos_starts, cpu_input_ids, cpu_inp_bids, cpu_query_starts, dynamic_bl_, dynamic_bsz_);
    forward(&logits_last, false, warmup_reqs, cpu_input_ids, cpu_query_starts, input_ids, inp_batch_ids, query_pos_starts, key_pos_starts, batch_maxlen_, dynamic_bl_, warmup_lora, &qwen2_param, full_cos, full_sin, &weights, &quant_ref_dict, kv_cache_ref, &dummy_timer);
    DeviceSynchronize();
    kv_cache_ref->free(std::string("warmup0001"));
    kv_cache_ref->free(std::string("warmup0002"));
    DeviceSynchronize();
    printf("warmup finished for data_id=%i\n", data_id);
    // ============== warmup完成===============================================

    ExecuteTimer timer;
    timer.disable(); // 性能排查时注销，开启计时。要每次记录点都与cpu同步的话，需要开启timer.enable_device_sync()。会增加总耗时，但每层耗时的相对比例会更精确。
    // timer.enable_device_sync();
    set_loading_finished(1);
    printf("<=============data parallel id %i ready for inference.===========\n", data_id);

    bool prev_iteration_is_prefill = false;
    StringArray decode_request_ids;
    std::string prev_forward_lora = warmup_lora.model_name; // warmup memory allocation
    decode_request_ids.Init(512, qwen2_param.max_dynamic_bsz);
    while (true) {
        SetDevice(input_device);
        std::string current_lora = batch_inp_preparer->GetLoraName();
        // 按照时间顺序、lora name，以及kv剩余可用空间，可能允许插入新样本。
        std::vector<AllocateParam> allocate_params = pool->Reload(data_id, current_lora, batch_inp_preparer->all_eos, kv_cache_ref);
        if (allocate_params.size() > 0) {
            // 正式为新样本分配存储
            batch_inp_preparer->ClearPrefill();
            auto suc = kv_cache_ref->sequence_allocate_cache(allocate_params);
            if (!suc) {
                printf("ERROR CACHE: kv-cache not able to allocate requests, this should not happen because numel has been checked.\n");
            }
            if (allocate_params[0].lora_name != prev_forward_lora) {
                // 切换lora时，lora_r可能不同，所以清理BigBuffer, 重新预分配。
                printf("CUDA MEM: data_id %i clearing big buffer for lora switch: %s->%s\n", data_id, prev_forward_lora.c_str(), allocate_params[0].lora_name.c_str());
                ManagedCudaClearBigBuffer(qwen2_param.layer2deviceId[0], qwen2_param.pp_size);
                DeviceSynchronize();
            }
            std::stringstream prefill_info;
            prefill_info << "GENERATION RELOADED: unfinished_decode_ct=" << batch_inp_preparer->decode_bsz << ", new_prefill=[";
            cpu_query_starts[0] = 0;
            int dynamic_bl;
            int batch_maxlen;
            for (int prefill_i=0; prefill_i < allocate_params.size(); prefill_i++) {
                // 循环新样本拼接inputs，以及获取forward和upload所需的信息。
                std::string prefill_req_id = allocate_params[prefill_i].request_id;
                auto ctx_ref = pool->GetRes(prefill_req_id, true);
                batch_inp_preparer->AddPrefill(ctx_ref);
                dynamic_bl = batch_inp_preparer->prefill_starts[batch_inp_preparer->prefill_bsz];
                cpu_query_starts[prefill_i+1] = dynamic_bl;
                int cur_example_length = dynamic_bl - cpu_query_starts[prefill_i];
                batch_maxlen = batch_inp_preparer->prefill_batch_maxlen;
                int last_bid = batch_inp_preparer->prefill_bids[dynamic_bl-1];
                prefill_info << "(" << last_bid << "|" << dynamic_bl << "|" << batch_inp_preparer->batch_lora_name << "|" << batch_inp_preparer->prefill_seeds[prefill_i]  << "),";
            }
            std::string joined_reqs = (batch_inp_preparer->prefill_req_ids).get_list_joined();
            prefill_info << "], req_ids=[" << joined_reqs.c_str() <<"]\n";
            printf(prefill_info.str().c_str());
            int dynamic_bsz = (int)(batch_inp_preparer->prefill_req_ids.size());

            // 选择lora
            std::string prefill_lora_name = batch_inp_preparer->GetLoraName();
            LoraConfig prefill_lora_cfg = GetLora(prefill_lora_name, prefill_lora_name, lora_meta);
            // printf("lora for prefill is %s\n", prefill_lora_cfg.model_name.c_str());

            // 上传inputs， forward，以及清空preparer的信息（为BatchUpdate后的decode inputs清理lists）。
            bool batch_return_lgts = batch_inp_preparer->PrefillShouldReturnLogits(cpu_return_logits);
            batch_inp_preparer->UploadInputs(true, input_device, output_device, input_ids, inp_batch_ids, query_pos_starts, key_pos_starts, temperature, seeds_tensor, top_ps, dynamic_bsz);
            forward(&logits_last, true, batch_inp_preparer->prefill_req_ids, batch_inp_preparer->prefill_inp_ids.data(), cpu_query_starts, input_ids, inp_batch_ids, query_pos_starts, key_pos_starts, batch_maxlen, dynamic_bl, prefill_lora_cfg, &qwen2_param, full_cos, full_sin, &weights, &quant_ref_dict, kv_cache_ref, &timer);

            // 初始化随机种子，应用温度，采样token
            gpu_curand_init(output_device, qwen2_param.world_size, dynamic_bsz, 0, seeds_tensor);
            filterInvalidApplyTemperature(logitsFp32, logits_last, temperature, vocab_size, dynamic_bsz, usual_eos_id);
            topk_sampling(sampled_id, output_device, qwen2_param.world_size, 0, logitsFp32, vocab_size, top_k, top_ps, _1_pass_result, _1_psss_indices, gpu_top_logits, gpu_top_indices, sample_softmax_out, dynamic_bsz);
            
            BatchGeneratedRes gen_res = download_sampled(sampled_id, cpu_sampled_id, eos_ids, qwen2_param.eos_token_num, dynamic_bsz);
            if (batch_return_lgts) {
                batch_download_logits(top_batch_idx, top_batch_lgts, gpu_top_logits, gpu_top_indices, dynamic_bsz, top_k);
            }
            BatchLogitsRes top_lgt_info = BatchLogitsRes{batch_return_lgts, dynamic_bsz, top_k, top_batch_idx, top_batch_lgts, cpu_return_logits};
            batch_inp_preparer->PrefillUpdate(data_id, batch_inp_preparer->prefill_req_ids, gen_res.batch_eoses, gen_res.batch_tk_ids, pool, kv_cache_ref, top_lgt_info);
            prev_iteration_is_prefill = true;
            continue;
        }

        int decoding_example_ct = (int)(batch_inp_preparer->decoding_examples).size();
        int decode_bsz = batch_inp_preparer->decode_bsz;
        if (decoding_example_ct != decode_bsz) {
            printf("ERROR: decoding example ct (%i) should be equal to DynamicExample vector length (%i)\n", decode_bsz, decoding_example_ct);
        }

        if (decode_bsz == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            batch_inp_preparer->all_eos = true;
            continue;
        }

        std::string latest_lora = batch_inp_preparer->GetLoraName();
        LoraConfig decode_lora = GetLora(latest_lora, latest_lora, lora_meta);
        std::stringstream decode_info;
        decode_info << "GENERATION DECODING: len=" << decode_bsz << ", examples=[";
        decode_request_ids.clear();

        cpu_query_starts[0] = 0;
        int batch_maxlen = 0;
        int dynamic_bl = batch_inp_preparer->decode_starts[batch_inp_preparer->decode_bsz];
        for (int bi=0; bi<decode_bsz; bi++) {
            decode_request_ids.push_back(batch_inp_preparer->decode_req_ids[bi]);
            int pos_end = batch_inp_preparer->decode_starts[bi+1];
            cpu_query_starts[bi+1] = pos_end;
            int cur_example_length = pos_end - cpu_query_starts[bi];
            batch_maxlen = (batch_maxlen > cur_example_length) ? batch_maxlen : cur_example_length;
            decode_info << "(" << bi << "|" << pos_end << "),"; 
        }
        std::string joined_dec_reqs = decode_request_ids.get_list_joined();
        decode_info << "], req_ids=[" << joined_dec_reqs.c_str() <<"]\n";
        if (prev_iteration_is_prefill) {
            printf(decode_info.str().c_str());
        }
        
        // batch所需cpu端的inputs已经在上次BatchUpdate时填好。
        bool batch_logits_enabled = batch_inp_preparer->DecodeShouldReturnLogits(cpu_return_logits);
        batch_inp_preparer->UploadInputs(false, input_device, output_device, input_ids, inp_batch_ids, query_pos_starts, key_pos_starts, temperature, seeds_tensor, top_ps, decode_bsz);
        forward(&logits_last, false, decode_request_ids, batch_inp_preparer->decode_inp_ids.data(), cpu_query_starts, input_ids, inp_batch_ids, query_pos_starts, key_pos_starts, batch_maxlen, dynamic_bl, decode_lora, &qwen2_param, full_cos, full_sin, &weights, &quant_ref_dict, kv_cache_ref, &timer);
        // 清理cpu上的数据，为下次DecodeUpdate做准备。decode_request_ids和cpu_query_starts被维护好了，不受清理的影响。
        batch_inp_preparer->ClearDecode();
        // 记录当前decode的lora
        prev_forward_lora = decode_lora.model_name;

        gpu_curand_init(output_device, qwen2_param.world_size, decode_bsz, 0, seeds_tensor);
        filterInvalidApplyTemperature(logitsFp32, logits_last, temperature, vocab_size, decode_bsz, usual_eos_id);
        topk_sampling(sampled_id, output_device, qwen2_param.world_size, 0, logitsFp32, vocab_size, top_k, top_ps, _1_pass_result, _1_psss_indices, gpu_top_logits, gpu_top_indices, sample_softmax_out, decode_bsz);
        
        BatchGeneratedRes decode_res = download_sampled(sampled_id, cpu_sampled_id, eos_ids, qwen2_param.eos_token_num, decode_bsz);
        if (batch_logits_enabled) {
            batch_download_logits(top_batch_idx, top_batch_lgts, gpu_top_logits, gpu_top_indices, decode_bsz, top_k);
        }
        BatchLogitsRes top_lgt_info = BatchLogitsRes{batch_logits_enabled, decode_bsz, top_k, top_batch_idx, top_batch_lgts, cpu_return_logits};
        batch_inp_preparer->DecodeUpdate(data_id, decode_request_ids, decode_res.batch_eoses, decode_res.batch_tk_ids, pool, kv_cache_ref, top_lgt_info);
        prev_iteration_is_prefill = false;
        // std::this_thread::sleep_for(std::chrono::milliseconds(1000000));
    }
}

} // namespace liteqwen
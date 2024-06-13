#include "entities.h"
#include "core_cpu.h"
#include "core_gpu.cuh"

namespace liteqwen {

int REPEAT_THRESHOLD = 3;
int DEFAULT_EOS = 151645;

void GenerationConfig::SetRandSeed() {
    auto start_tm = std::chrono::system_clock::now();
    auto start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(start_tm.time_since_epoch()).count();
    this->seed =((int)start_ts) % 1000;
}

void BatchLogitsRes::try_insert_logits(ResponseContext* ctx_ref, int batch_id) {
    // printf("batch lgt record enabled=%i, return_lgt=%i(batch_id=%i) for req=%s\n", static_cast<int>(this->enabled), static_cast<int>(return_logits[batch_id]), batch_id, ctx_ref->request_id.c_str());
    if (this->enabled) {
        if (this->return_logits[batch_id]) {
            for (int k=0; k<this->top_k; k++) {
                int token_id = this->top_token_ids[batch_id * top_k + k];
                float lgt = this->top_logits[batch_id * top_k + k];
                int pos = ctx_ref->current_length;
                ctx_ref->AppendLogits(lgt, token_id, pos);
            }
        }
    }
}

void Q4LinearMeta::get_store_location(ParamLocation* location, std::string w_key, bool desc_act) {
    // QUANT_BUFFER: 检测到gptq quant4相关权重，需要统计最大qweight buffer大小，以及将g_idx复制到cpu中。
    // GPTQ的 g_idx 只在 config中的 desc_act=true时起作用，这时需要分配名称是 max_inner_outer_dim 的buffer。
    // 当desc_act=false时，max_inner_outer_dim=1
    // desc_act在qwen2_param里默认给false。
    int prefix_len = (int)(this->prefix.length());
    if (prefix_len < (int)(w_key).length()) {
        if (w_key.substr(0, prefix_len) == this->prefix) {
            std::string tail_str = w_key.substr(prefix_len, (int)(w_key).length());
            if (tail_str == std::string(".qweight")) {
                // return ParamLocation::QUANT_BUFFER;
                *location = ParamLocation::QUANT_BUFFER;
                return;
            }
            if (tail_str == std::string(".g_idx")) {
                if (desc_act) {
                    // return ParamLocation::CPU;
                    *location = ParamLocation::CPU;
                    return;
                } else {
                    // return ParamLocation::EMPTY;
                    *location = ParamLocation::EMPTY;
                    return;
                }
            }
            // return ParamLocation::GPU;
            *location = ParamLocation::GPU;
            return;
        }
    }
    // return ParamLocation::MISS;
    *location = ParamLocation::MISS;
    return;
}

void Qwen2Params::Init(int world_size, int data_parallel_size, std::string json_config_path, std::vector<int> base_layer2device, int max_dynamic_bsz, int max_sequence_length) {
    printf("initializing cpp Qwen1.5 model, with num_layers=%i, max_dynamic_bsz=%i\n", (int)(base_layer2device.size()), max_dynamic_bsz);
    
    // this->bos_token_id = 130004;    // V1 后期版本 bos token，可通过 config.json 覆盖
    // this->eos_token_id = 130005;    // V1 后期版本 eos token，可通过 config.json 覆盖
    // this->gmask_token_id= 150001;   // V1最初版本, 150528 tokens，部分 config.json 没有 gmask_token_id，因此取默认值。
    liteqwen::REPEAT_THRESHOLD = 5;
    this->rope = -1.0f;
    this->world_size = world_size;
    this->dp_size = data_parallel_size;
    this->pp_size = world_size/data_parallel_size;

    this->embedding_name = std::string("model.embed_tokens.weight");
    this->rotary_inv_name = std::string("transformer.rotary_pos_emb.inv_freq_TODO"); // not being used. cos and sin are calculated during initializing.
    this->eos_token_num = 3;
    this->eos_ids = std::vector<int>{151645, 151644, 151643}; // 第一个eos作为默认，其余eos都会被改写为默认eos。
    liteqwen::DEFAULT_EOS = this->eos_ids[0];
    this->gptq_desc = false;
    this->top_k = 32; // fixed, for more convenient batch inference.

    std::string error;
    this->json_config_path = json_config_path;
    std::ifstream cfg_stream(json_config_path);        std::string json_cfg_str((std::istreambuf_iterator<char>(cfg_stream)), std::istreambuf_iterator<char>());
    this->config = json11::Json::parse(json_cfg_str, error);

    this->hidden_size = this->config["hidden_size"].int_value();
    this->num_attention_heads = this->config["num_attention_heads"].int_value();
    this->num_key_value_heads = this->config["num_key_value_heads"].int_value();
    this->kv_channels = this->hidden_size / this->num_attention_heads;
    if (num_key_value_heads != num_attention_heads) {
        printf("WARNING: detected H_q(%i) != H_kv(%i). Only qwen 32b uses gqa, allowing num_attention_heads>num_key_value_heads, make sure correct model is used.\n", num_attention_heads, num_key_value_heads);
    }
    this->num_layers = this->config["num_hidden_layers"].int_value();
    this->max_dynamic_bsz = max_dynamic_bsz;
    this->max_sequence_length = max_sequence_length;
    this->ffn_hidden_size = this->config["intermediate_size"].int_value();
    this->padded_vocab_size = this->config["vocab_size"].int_value();

    if (this->hidden_size / this->num_attention_heads != this->kv_channels) {
        printf("ERROR: kv_channel should be equal to hidden_size / num_attention_heads, check for config\n");
        exit(1);
    }

    std::vector<std::map<int, int>> default_device_maps = std::vector<std::map<int, int>>(); //for each data_id, map layer_id to device_id.
    int layer_num = (int)(base_layer2device.size());
    if (layer_num !=this->num_layers) {
        printf("layer num implied by device id list is not same as num_layers in config. please modify device id list for each layer\n");
        exit(1);
    }

    // data parallel groups [[1,2], [3,4], [5,6], [7,8]] for dp=4, pp=2
    std::vector<int> data0_layer2deviceId;
    std::vector<std::pair<int, int>> stage2layer_range;
    int prev_device = 0;
    int prev_device_layer = 0;
    std::vector<std::string> data_group_print = std::vector<std::string>();
    for (int li=0; li< static_cast<int>(base_layer2device.size()); li++) {
        int data0_device_id = base_layer2device[li];
        
        if (data0_device_id >= this->pp_size) {
            printf("ERROR: device_id %i assigned for layer %i exceeds pipeline parallel boundary [0,%i). Please fix pipeline parallel configuration to make sure world_size=pp*dp, and each data_id does not use more gpu numbers than pp_size\n", data0_device_id, li, this->pp_size);
            exit(1);
        }
        data0_layer2deviceId.push_back(data0_device_id);
        
        data_group_print.push_back(std::to_string(data0_device_id));

        if (li==static_cast<int>(base_layer2device.size())-1) {
            stage2layer_range.push_back(std::pair<int, int>(prev_device_layer, li+1));       
        } else if (prev_device != data0_device_id) {
            stage2layer_range.push_back(std::pair<int, int>(prev_device_layer, li));
            prev_device_layer = li;
        }
        prev_device = data0_device_id;
    }

    if ((int)(stage2layer_range.size()) != this->pp_size) {
        printf("ERROR: pp_size(%i) should be equal to stage num implied by base_layer2device list which span over %i devices.\n", this->pp_size, (int)(stage2layer_range.size()));
        exit(1);
    }

    this->data0_layer2deviceId = data0_layer2deviceId;
    this->stage2layer_range = stage2layer_range;
    auto dev_id_list_str = liteqwen::join(data_group_print, std::string(","));
    printf("Data parallel worker data_id=0 has layers on devices [%s]\n", dev_id_list_str.c_str());

    // some redundant params...
    this->projection_size = (this->kv_channels) * this->num_attention_heads;
    this->hidden_size_per_attention_head = this->projection_size / this->num_attention_heads;
    this->multi_query_group_num = this->num_attention_heads / this->num_key_value_heads;
};

void Qwen2Params::update_data_id(int data_id) {
    this->data_id = data_id;
    std::vector<int> layer2deviceId;
    for (int li=0; li< static_cast<int>(this->data0_layer2deviceId.size()); li++) {
        int data0_device_id = this->data0_layer2deviceId[li];
        int deploying_device = data0_device_id + this->pp_size * data_id;
        layer2deviceId.push_back(deploying_device);
    }
    this->layer2deviceId = layer2deviceId;
    this->input_deviceId = layer2deviceId.front();
    this->output_deviceId = layer2deviceId.back();
    printf("data_id %i is using devices[%i, %i)\n", data_id, this->input_deviceId, this->output_deviceId+1);
}

int Qwen2Params::get_name2device(std::string w_key, int input_device, int output_device) {
    std::regex word_regex("(layers\\.)[0-9]+");
    auto words_begin = std::sregex_iterator(w_key.begin(), w_key.end(), word_regex);
    auto words_end = std::sregex_iterator();
    int layer_id = -1;
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        std::string match_str = match.str();
        if (match_str.size() > 0) {
            std::string num_str = match_str.substr(7, (int)match_str.size()-7);
            layer_id = atoi(num_str.c_str());
            break;
        }
    }
    if (layer_id >=0) {
        int device_id = (this->layer2deviceId)[layer_id];
        return device_id;
    } else if (w_key == this->embedding_name || w_key==this->rotary_inv_name) {
        if (input_device < 0) {
            return this->input_deviceId;
        }
        return input_device;
    } else {
        if (output_device < 0) {
            return this->output_deviceId;
        }
        return output_device;
    }
}

void ResponseContext::Init(std::string request_id, std::vector<int> input_ids, GenerationConfig gen_cfg, float logits_mask_base_val, float logits_mask_except_val, std::vector<int> logit_mask_except_ids, bool return_logits) {
    this->request_id = request_id;
    this->input_length = (int)(input_ids.size());
    this->current_length = (int)(this->input_length);
    this->prev_length = (int)(this->input_length);
    this->tokens = std::vector<int>(input_ids);
    this->generation_config = gen_cfg;

    this->logit_mask_base_val = logits_mask_base_val;
    this->logit_mask_except_val = logits_mask_except_val;
    this->logit_mask_except_ids = logit_mask_except_ids;

    this->return_logits = return_logits;
    this->token_logits = std::vector<TopLogitsInfo>();
    this->repeat_ct = 0;

    if (this->generation_config.seed == -1) {
        this->generation_config.SetRandSeed();
    }
}

bool ResponseContext::Append(int new_token, bool is_eos) {
    bool repeat_check_passed = true;

    int prev = this->tokens.back();
    if (prev == new_token) {
        this->repeat_ct +=1;
    } else {
        this->repeat_ct = 0;
    }

    if (this->repeat_ct >= liteqwen::REPEAT_THRESHOLD) { // single token repeat check
        repeat_check_passed = false;
    } else { // window token repeat check
        int window_size = liteqwen::REPEAT_THRESHOLD * 4;
        std::vector<int> window;
        if (this->current_length - window_size > 0) {
            std::stringstream window_info;
            window_info << "[";
            for (auto itm=(this->tokens.end() - window_size + 1); itm != this->tokens.end(); itm++) {
                int prev_tk = *itm;
                window.push_back(prev_tk);
                window_info<< prev_tk << ",";
            }
            window.push_back(new_token);
            window_info<< "]\n";
            // int window_size2 = (int)window.size();
            // printf("window= %s", window_info.str().c_str());
            for (int i=2; i<=4; i++) {
                int repeat_token_ct = 0;
                for (int j=0; j<i; j++) {
                    int same_ct = 0;
                    for (int k=1; k<(liteqwen::REPEAT_THRESHOLD+1); k++) {
                        int diff = window[window_size-1-j] - window[window_size-1-j - i*k];
                        if (diff == 0) {
                            same_ct +=1;
                        }
                    }
                    if (same_ct >= liteqwen::REPEAT_THRESHOLD) {
                        repeat_token_ct+=1;
                    }
                }
                if (repeat_token_ct == i) {
                    // all tokens in window are repeated.
                    repeat_check_passed = false;
                    break;
                }
            }
        }
    }
    
    if (!repeat_check_passed) {
        this->tokens.push_back(DEFAULT_EOS);
        this->current_length += 1;
        this->isEnding = true;
        return false;
    } else {
        // printf("appending token: %i\n", new_token);
        this->tokens.push_back(new_token);
        this->current_length += 1;
        if (is_eos){
            this->isEnding = is_eos;
        }
        return (!is_eos);
    }
}

void ResponseContext::AppendLogits(float logit, int token_id, int pos) {
    this->token_logits.push_back(TopLogitsInfo{logit, token_id, pos});
}

void ResponseContext::SetGenerateFlag(int data_id) {
    this->processing_data_id = data_id;
}

void ResponseContext::SetPrevLen(int prev_len) {
    this->prev_length = prev_len;
}

ExecuteTimer::ExecuteTimer() {
    this->chain_complete = false;
    this->event_chain = std::map<std::string, std::string>();
    this->latest_ts = std::map<std::string, std::chrono::system_clock::time_point>();
    this->span_acc = std::map<std::string, double>();
    this->span_ct = std::map<std::string, int>();
    this->prev_event_name = std::string("start");
    this->mark_start_tm = std::chrono::system_clock::now();
    this->mark_acc = std::map<std::string, double>();
    this->sync_with_cpu=false;
    this->stat_enabled = true;
}

ExecuteTimer::~ExecuteTimer() {
}

void ExecuteTimer::enable_device_sync() {
    this->sync_with_cpu=true;
}
void ExecuteTimer::disable() {
    this->stat_enabled=false;
}

void ExecuteTimer::regist(std::string event_name) {
    if (!this->stat_enabled) {
        return;
    }

    if (this->sync_with_cpu) {
        DeviceSynchronize();
    }
    
    auto cur_t = std::chrono::system_clock::now();
    this->latest_ts[event_name] = cur_t;
    std::string prev_name;
    if (! this->chain_complete) {
        if ((this->event_chain).find(event_name) == (this->event_chain.end())) {
            if (this->prev_event_name != std::string("start")) {
                this->event_chain[event_name] = this->prev_event_name;
                prev_name = this->prev_event_name;
            } else {
                prev_name = std::string("start");
            }
        } else {
            this->chain_complete = true;
            this->event_chain[event_name] = this->prev_event_name;
            prev_name =  this->prev_event_name;
        }
    } else {
        if ((this->event_chain).find(event_name) == (this->event_chain.end())) {
            prev_name = std::string("start");
        } else {
            prev_name = this->event_chain[event_name];
        }
    }

    if (prev_name != std::string("start")) {
        auto t0 = ((this->latest_ts).find(prev_name))->second;
        double dt = this->get_span(t0, cur_t);
        (this->span_acc)[event_name] = (this->span_acc)[event_name] + dt;
        (this->span_ct)[event_name] = (this->span_ct)[event_name] + 1;
    }
    this->prev_event_name = event_name;
}

void ExecuteTimer::mark(std::string mark_name) {
    if (!this->stat_enabled) {
        return;
    }

    auto cur_t = std::chrono::system_clock::now();
    if (mark_name == std::string("start")) {
        this->mark_start_tm = std::chrono::system_clock::now();
    } else if (mark_name == std::string("end")) {
        double dt = this->get_span(this->mark_start_tm, cur_t);
        (this->mark_acc)[mark_name] = (this->mark_acc)[mark_name] + dt;
        this->mark_start_tm = cur_t;
    } else {
        double dt = this->get_span(this->mark_start_tm, cur_t);
        (this->mark_acc)[mark_name] = (this->mark_acc)[mark_name] + dt;
    }
}

double ExecuteTimer::get_span(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};

void ExecuteTimer::print_stat() {
    if (!this->stat_enabled) {
        return;
    }

    for (auto ent: (this->event_chain)) {
        const std::string key = (ent.first);
        auto mean_val = ((this->span_acc).find(key)->second); // / ((this->span_ct).find(key)->second);
        int pass_ct = (this->span_ct).find(key)->second;
        float value = static_cast<float>(mean_val);
        printf("event %s->%s costs %f/%i secs.\n", (ent.second).c_str(), key.c_str(), value, pass_ct);
    }

    for (auto mark: (this->mark_acc)) {
        const std::string mkey = (mark.first);
        float mval = static_cast<float>((this->mark_acc).find(mkey)->second);
        printf("mark %s costs %f secs.\n", mkey.c_str(), mval);
    }
}

} // namespace liteqwen

#ifndef _LITEENT_H
#define _LITEENT_H
#include <string>
#include <chrono>
#include <map>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <regex>
#include "json11.h"
#include <type_traits>

namespace liteqwen{

struct BatchGeneratedRes {
    std::vector<bool> batch_eoses;
    std::vector<int> batch_tk_ids;
};

struct Qwen2Params {
    void Init(int world_size, int data_parallel_size, std::string json_config_path, std::vector<int> base_layer2device, int max_dynamic_bsz, int max_sequence_length);
    int get_name2device(std::string w_key, int input_device=-1, int output_device=-1);
    void update_data_id(int data_id);

    int data_id;
    float rope;
    int world_size;
    int dp_size;
    int pp_size;
    // bool model_initialized = false;
    int hidden_size;
    int num_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int projection_size;
    int hidden_size_per_attention_head;
    int multi_query_group_num;
    int qkv_hidden_size;
    int ffn_hidden_size;
    int kv_channels;
    int padded_vocab_size;
    int max_dynamic_bsz;
    int max_sequence_length;
    bool gptq_desc;
    int top_k;
    
    std::string json_config_path;
    std::vector<int> data0_layer2deviceId; //for each data_id, map layer_id to device_id.
    std::vector<int> layer2deviceId;
    std::vector<std::pair<int, int>> stage2layer_range;
    int input_deviceId;
    int output_deviceId;
    std::string embedding_name;
    std::string rotary_inv_name;
    json11::Json config;

    std::vector<int> eos_ids;
    int eos_token_num;
};

enum ParamLocation {
    MISS=0, IGNORE=1, EMPTY=2, QUANT_BUFFER=3, GPU=4, CPU=5
};

struct Q4LinearMeta {
    std::string prefix;
    int in_features;
    int out_features;
    int group_size;
    bool has_bias;

    void get_store_location(ParamLocation*, std::string w_key, bool desc_act);
};

struct GenerationConfig {
    float top_p = 0.8; // top_p采样
    int top_k = 32; // top_k采样
    float temperature = 0.8; // 温度参数，一般在0.1 ~ 1.0之间，设大这个参数可以带来结果的多样性
    int max_length; // kv-cache里分配显存使用的长度。
    int max_new_tokens = -1; // 如果>0，则取 min(max_sequence_length, input_length+max_new_tokens)作为eos截断条件。
    std::string adapter_name = std::string("skip"); // 默认无lora，值是skip
    int seed = -1;

    void SetRandSeed();
};

struct LoraConfig { // 推理不需要bias、dropout等参数
    std::string model_name = "default";
    bool fan_in_fan_out = false;
    float lora_alpha = 16.0f; 
    int r = 64;
    std::vector<std::string> target_modules = std::vector<std::string>(); // std::string("query_key_value&dense&dense_4h_to_h&dense_h_to_4h")
    
    static LoraConfig create(std::string _model_name, bool _fan_in_fan_out, float _lora_alpha, int _r, std::vector<std::string> _target_modules) {
        static LoraConfig _cfg = LoraConfig{_model_name, _fan_in_fan_out, _lora_alpha, _r, _target_modules};
        return _cfg;
    }

    void set_name(std::string _model_name) {
        this->model_name = _model_name;
    }
};

struct TopLogitsInfo {
    float logits;
    int token_id;
    int pos;
};

struct Response {
    int status; // 0 waiting, 1 generating, 2 finished, -1 interupted/error
    int cur_length; // 当使用增量传输时，当前token ids长度用于校验流式结果没有丢失。
    std::vector<int> response_ids;
    std::vector<TopLogitsInfo> response_logits;
};

struct ResponseContext
{
    bool isEnding = false;
    std::string request_id;
    int handle_id = -1;
    int processing_data_id = -1;
    int repeat_ct = 0;

    int input_length = 0;
    int current_length = 0;
    int prev_length = 0;
    liteqwen::GenerationConfig generation_config;
    std::vector<int> tokens;

    float logit_mask_base_val;
    float logit_mask_except_val;
    std::vector<int> logit_mask_except_ids;

    bool return_logits;
    std::vector<TopLogitsInfo> token_logits;

    void Init(std::string request_id, std::vector<int> input_ids, GenerationConfig gen_cfg, float logits_mask_base_val, float logits_mask_except_val, std::vector<int> logit_mask_except_ids, bool return_logits);
    bool Append(int new_token, bool is_eos);
    void AppendLogits(float logit, int token_id, int pos);
    void SetGenerateFlag(int data_id);
    void SetPrevLen(int prev_len);
};

class ExecuteTimer {
    public:
    ExecuteTimer();
    ~ExecuteTimer();
    bool chain_complete;
    bool sync_with_cpu;
    bool stat_enabled;
    std::string prev_event_name;
    std::map<std::string, std::string> event_chain;
    std::map<std::string, std::chrono::system_clock::time_point> latest_ts;
    std::map<std::string, double> span_acc;
    std::map<std::string, double> mark_acc;
    std::map<std::string, int> span_ct;
    std::chrono::system_clock::time_point mark_start_tm;
    void enable_device_sync();
    void disable();
    void regist(std::string event_name);
    void mark(std::string mark_name);
    void print_stat();
    double get_span(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);
};

struct AllocateParam {
    std::string request_id;
    bool successful;
    size_t bl_start;
    size_t bl_end;
    std::string lora_name;

    int get_block_len(int cache_channel) {
        return static_cast<int>((this->bl_end - this->bl_start) / cache_channel);
    }

    void set_lora(std::string lora_name) {
        this->lora_name = lora_name;
    }
};


/// std::vector<string>在大小不固定时，频繁跨方法deep copy，会导致request_id出错，影响InputPrepare，导致bug。
/// request_id贯穿InputPreparer, forward, 以及kv-cache，所以用StringArray struct手动管理这部分信息。
/// 实现一个固定大小StringArray用于容纳单个样本最大512字符，最多max_dynamic_bsz个样本的request_id。
struct StringArray {
    char* data_raw;
    int* str_lens;
    char** string_starts;
    int current_used;
    int array_size;
    void Init(int max_string_length, int array_size) {
        this->str_lens = new int[array_size];
        this->data_raw = new char[max_string_length * array_size];
        this->string_starts = new char*[array_size];
        for (int i=0; i<array_size; i++) {
            this->string_starts[i] = data_raw + (i * max_string_length);
        }
        this->current_used = 0;
        this->array_size = array_size;
    }

    void push_back(std::string item) {
        if (this->current_used < this->array_size) {
            char* new_str = this->string_starts[this->current_used];
            int str_len = item.length();
            // printf("pushing back string, len=%i\n", str_len);
            memcpy(new_str, item.c_str(), sizeof(char)*str_len);
            this->str_lens[this->current_used] = str_len;
            this->current_used += 1;
        }
    }

    std::string operator [](int idx) {
        if (idx < this->current_used) {
            size_t char_len = static_cast<size_t>(this->str_lens[idx]);
            return std::string(string_starts[idx], char_len);
        } else {
            return std::string("");
        }
    }
    std::string operator [](int idx) const {
        if (idx < this->current_used) {
            size_t char_len = static_cast<size_t>(this->str_lens[idx]);
            return std::string(string_starts[idx], char_len);
        } else {
            return std::string("");
        }
    }

    void clear() {
        for (int i=0; i<this->array_size; i++) {
            this->str_lens[i] = 0;
        }
        this->current_used = 0;
    }

    int size() {
        return this->current_used;
    }

    std::string get_list_joined() {
        if (this->current_used > 0 ) {
            std::string res = "";
            std::stringstream res_builder;
            res_builder.clear();
            for (int i=0; i<this->current_used; i++) {
                size_t char_len = static_cast<size_t>(this->str_lens[i]);
                res_builder << std::string(string_starts[i], char_len).c_str() << ",";
                // res = (res + std::string(string_starts[i], char_len) + std::string(","));
            }
            // return res;
            return res_builder.str();
        } else {
            return std::string("");
        }
    }
};


template <typename dtype>
dtype zero() {
    return static_cast<dtype>(0);
};

template<typename dtype>
struct DataArray {
    dtype* raw_data;
    int current_used;
    int array_size;

    void Init(int array_size) {
        this->raw_data = new dtype[array_size + 8]; //多声明一些，防止踩踏。
        this->current_used = 0;
        this->array_size = array_size;
    }

    void push_back(dtype item) {
        if (this->current_used < this->array_size) {
            this->raw_data[this->current_used] = item;
            this->current_used += 1;
        }
    }

    dtype operator [](int idx) {
        if (idx < 0) {
            return raw_data[this->current_used-idx];
        } else if (idx < this->current_used) {
            return raw_data[idx];
        } else {
            return zero<dtype>();
        }
    }
    dtype operator [](int idx) const {
        if (idx < 0) {
            return raw_data[this->current_used-idx];
        } else if (idx < this->current_used) {
            return raw_data[idx];
        } else {
            return zero<dtype>();
        }
    }    

    void clear() {
        this->current_used = 0;
    }

    int size() {
        return this->current_used;
    }

    dtype* data() {
        return this->raw_data;
    }

    void set_val(int idx, dtype val) {
        if (idx < 0) {
            raw_data[this->current_used-idx] =  val;
        } else if (idx < this->array_size) {
            raw_data[idx] = val;
        } else {
            printf("idx=%i index overflow with boundary=%i\n", idx, this->array_size);
        }
        if (this->current_used < idx+1) {
            this->current_used = idx+1;
        }
    }
};

} //namespace liteqwen

#endif
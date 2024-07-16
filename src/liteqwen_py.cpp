#include <cstring>
#include <thread>

#include "core_cpu.h"
#include "liteqwen.h"
#include "entities.h"
#include "json11.h"
#include "generate.h"


#ifdef WIN32
#define DLL_EXPORT _declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#include "core_cpu.h"
#include "core_gpu.cuh"
#include "pool.h"
#include "kv_cache.h"
#include "entities.h"
#include "xformer_attention.h"
#include "forward_gpu.cuh"


extern "C" {
    DLL_EXPORT int submit_request(int record_id, char* request_id, int input_length, int* input_ids, float top_p, int top_k, float temperature, int max_length, int max_new_tokens, 
    char* adapter_name, int seed, float mask_base_val, float mask_except_val, int except_ids_len, int* logit_except_ids, bool return_logits) {
        std::string adapter = std::string(adapter_name);
        std::string request_id_str = std::string(request_id);
        
        int applied_maxlen = max_length;
        // int max_new_tokens = max_new_tokens_;
        if (max_length < 1 && max_new_tokens < 1) {
            // 默认保留400生成token
            applied_maxlen = ((input_length + 400 - 1) / 32 + 1) * 32;
            max_new_tokens = applied_maxlen - input_length;
        } else if (max_length > 0 && max_new_tokens > 0) {
            // 取更小的length
            if (input_length + max_new_tokens < max_length) {
                applied_maxlen = input_length + max_new_tokens;
            } else {
                applied_maxlen = max_length;
                max_new_tokens = max_length - input_length;
                if (max_new_tokens < 1) {
                    printf("Input Length Error: providing input_length+1 < max_length, this is forbidden. aborting request.\n");
                    return false;
                }
            }
        }
        else if (max_new_tokens > 0) { // only max_new_tokens > 0
            applied_maxlen = input_length + max_new_tokens;
        } else { // only max_length > 0
            max_new_tokens = max_length - input_length;
            if (max_new_tokens < 1) {
                printf("Input Length Error: providing input_length+1 < max_length, this is forbidden. aborting request.\n");
                return false;
            }
        }

        int fixed_top_k = 32; // use fixed top k instead, same as Qwen2Params
        liteqwen::GenerationConfig config = liteqwen::GenerationConfig{top_p, fixed_top_k, temperature, applied_maxlen, max_new_tokens, adapter, seed};
        std::vector<int> inp_ids = std::vector<int>();
        for (int i=0; i<input_length; i++) {
            inp_ids.push_back(input_ids[i]);
        }

        std::vector<int> except_ids = std::vector<int>();
        if (mask_except_val != 0.0 || mask_base_val != 0.0) {
            for (int i2=0; i2<except_ids_len; i2++) {
                except_ids.push_back(logit_except_ids[i2]);
            }
        }
        
        int success = submit_inference(record_id, request_id_str, inp_ids, config, mask_base_val, mask_except_val, except_ids, return_logits);
        return success;
    }

    DLL_EXPORT void store_tensor(char *key, int shape_length, int *shape,
                              int dtype, int oriDataType, void *oriData) {
        
        std::vector<int> shape_vec = std::vector<int>(shape_length);
        for (int i = 0; i < shape_vec.size(); i++) {
            shape_vec[i] = shape[i];
        }
        liteqwen::DataType parsed_dtype = (liteqwen::DataType)dtype;
        liteqwen::DataType orig_dtype = (liteqwen::DataType)oriDataType;
        std::string key_str = std::string(key);
        add_qwen2_weight(key_str, shape_vec, parsed_dtype, orig_dtype, oriData);
        return;
    }

    DLL_EXPORT void make_q4_meta(char* prefix, int in_features, int out_features, int group_size, bool has_bias) {
        liteqwen::Q4LinearMeta meta = liteqwen::Q4LinearMeta{std::string(prefix), in_features, out_features, group_size, has_bias};
        add_q4_linear_meta(meta);
    }

    // liteqwen_lib.initialize_empty_qwen2(world_size, data_parallel_size, pipeline_parallel_size, json_path.encode(), layer_num, (ctypes.c_int * layer_num)(*layer_to_device_list), max_dynamic_bsz, max_length, data_parallel_size*max_dynamic_bsz*5, int(self.timeout_in_secs*1000))
    DLL_EXPORT void initialize_empty_qwen2(int world_size, int running_thread_num, int data_parallel_size, int pipeline_parallel, char* json_config_path, int layer_num, int* block2device_list, int max_dynamic_bsz, int max_sequence_length, int max_queue_size, int timeout, char* py_smem_name, int py_smem_size, int record_length) {
        std::string config_path = std::string(json_config_path);
        std::vector<int> block2device_list_vec; // = std::vector<int>(layer_num);
        for (int li=0; li < layer_num; li++){
            block2device_list_vec.push_back(block2device_list[li]);
        }
        std::string smem_name = std::string(py_smem_name);
        init_empty_qwen2(world_size, running_thread_num, data_parallel_size, config_path, block2device_list_vec, max_dynamic_bsz, max_sequence_length, max_queue_size, timeout, smem_name, py_smem_size, record_length);
        return;
    }

    DLL_EXPORT void add_lora_adapter_config(char* adapter_name, bool fan_in_fan_out, float lora_alpha, int r, char* target_modules_joined) {
        // std::string _model_name, bool _fan_in_fan_out, float _lora_alpha, int _r, std::vector<std::string> _target_modules
        std::string adapter_name_str = std::string(adapter_name);
        std::vector<std::string> target_modules = std::vector<std::string>();
        std::string joined_str = std::string(target_modules_joined);
        std::string delimiter = ",";
        if (joined_str.back() != delimiter.back()) {
            joined_str = joined_str + delimiter;
        }
        
        size_t pos = 0;
        std::string token;
        std::string token_trimed;
        while (((pos = joined_str.find(delimiter)) != std::string::npos)) {
            token = joined_str.substr(0, pos);
            std::string token_trimed = liteqwen::trim(token);
            // std::cout << token_trimed << std::endl;
            target_modules.push_back(token_trimed);
            joined_str.erase(0, pos + delimiter.length());
        }

        liteqwen::LoraConfig lora_config = liteqwen::LoraConfig{adapter_name_str, fan_in_fan_out, lora_alpha, r, target_modules};
        add_lora_adapter(lora_config);
    }

    DLL_EXPORT void start_loops() {
        start_thread_loops();

        while (! liteqwen::get_loading_finished()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    DLL_EXPORT void start_loop_in_dp_id(int data_id) {
        start_single_thread_loop(data_id);
        while (! liteqwen::get_loading_finished()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    DLL_EXPORT int* get_frame(char* request_id, bool is_incremental, bool force_interupt, bool no_stream) {
        std::string request_id_str = std::string(request_id);
        liteqwen::Response resp = get_generated(request_id_str, is_incremental, force_interupt, no_stream);
        int ids_length = (int)(resp.response_ids.size());
        // void* res_ptr = malloc(sizeof(int) * (ids_length+3));
        // int* int_ptr = reinterpret_cast<int*>(res_ptr);
        int* int_ptr = new int[ids_length+3];
        int_ptr[0] = resp.status;
        int_ptr[1] = resp.cur_length;
        int_ptr[2] = ids_length;
        for (int i=0; i<ids_length; i++) {
            int_ptr[i+3] = resp.response_ids[i]; 
        }
        return int_ptr;
    }

    DLL_EXPORT unsigned char* get_frame_entity(char* request_id, bool is_incremental, bool force_interupt, bool no_stream) {
        std::string request_id_str = std::string(request_id);
        liteqwen::Response resp = get_generated(request_id_str, is_incremental, force_interupt, no_stream);
        int ids_length = (int)(resp.response_ids.size());

        std::map<std::string, json11::Json> result_pack;
        int logit_result_len = resp.response_logits.size();
        result_pack.insert(std::make_pair("status", json11::Json(resp.status)));
        result_pack.insert(std::make_pair("cur_len", json11::Json(resp.cur_length)));
        result_pack.insert(std::make_pair("token_ids", json11::Json(resp.response_ids)));
        if (logit_result_len > 0) {
            std::vector<float> logits;
            std::vector<int> top_ids;
            std::vector<int> top_pos;
            for (int j=0; j<resp.response_logits.size(); j++) {
                logits.push_back((resp.response_logits[j]).logits);
                top_ids.push_back((resp.response_logits[j]).token_id);
                top_pos.push_back((resp.response_logits[j]).pos);
            }
            result_pack.insert(std::make_pair("logits", json11::Json(logits)));
            result_pack.insert(std::make_pair("top_ids", json11::Json(top_ids)));
            result_pack.insert(std::make_pair("top_pos", json11::Json(top_pos)));
        } else {
            result_pack.insert(std::make_pair("return_logits", json11::Json(false)));
        }

        json11::Json Json_d = json11::Json(result_pack);
        std::string dump_mp = Json_d.dump();

        unsigned int data_len = dump_mp.length();
        unsigned char* cstr = new unsigned char[4 + dump_mp.size()];

        unsigned long n = static_cast<unsigned long>(data_len);
        cstr[0] = static_cast<unsigned char>((n >> 24) & 0xFF);
        cstr[1] = static_cast<unsigned char>((n >> 16) & 0xFF);
        cstr[2] = static_cast<unsigned char>((n >> 8) & 0xFF);
        cstr[3] = static_cast<unsigned char>((n >> 0) & 0xFF);

        std::copy(dump_mp.data(), dump_mp.data()+data_len, cstr+4);
        return cstr;
    }

    DLL_EXPORT void free_frame_entity(unsigned char* buffer)
    {
        delete[] buffer;
    }

    DLL_EXPORT void free_frame(int* buffer)
    {
        delete[] buffer;
    }

    DLL_EXPORT void delete_request(char* request_id) {
        std::string request_id_str = std::string(request_id);
        delete_request_ctx(request_id_str);
    }
}
#ifndef _LITEQWEN_H
#define _LITEQWEN_H

#include <vector>
#include <cstdint>
#include <string>
#include <queue>
#include <list>
#include <unordered_map>
#include <functional>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h> 

#include <thread>
#include <future>
#include <chrono>
#include <mutex>
#include "json11.h"
#include "entities.h"
#include "pool.h"
#include "core_cpu.h"

namespace liteqwen {

    static Qwen2Params qwen2params;
    static std::shared_ptr<Qwen2Params>  qwen2_ptr; //模型类
    static std::shared_ptr<std::map<std::string, Data*>> originalCPUWeights;
    static std::shared_ptr<std::map<std::string, Q4LinearMeta>> Q4linearMetaMap;
    static std::shared_ptr<ContextPool> thread_pool; // 负责请求上下文的队列、任务提交。
    static std::shared_ptr<std::map<std::string, LoraConfig>> lora_adapters; // 存储所有lora adapter的config
    static std::vector<std::thread*> worker_threads; // 负责dp推理
    static std::shared_ptr<int> py_token_buffer; // python的shared memory指针

    static bool loading_finished = false;
    static int load_finish_ct = 0;
    
    bool get_loading_finished();
    void set_loading_finished(int tick);

}

void init_empty_qwen2(int world_size, int running_thread_num, int data_parallel_size, std::string json_config_path, std::vector<int> layer2device_list, int max_dynamic_bsz, int max_sequence_length, int max_queue_size, int timeout, std::string py_smem_name, int py_smem_size, int record_length);
void add_lora_adapter(liteqwen::LoraConfig);
void add_q4_linear_meta(liteqwen::Q4LinearMeta q4_meta);
void add_qwen2_weight(const std::string& weight_name, std::vector<int> shape, liteqwen::DataType dtype, liteqwen::DataType oriDataType, void *oriData);
void start_thread_loops();
void start_single_thread_loop(int data_id);
int submit_inference(int record_id, const std::string& request_id, std::vector<int> input_ids, const liteqwen::GenerationConfig& gen_cfg, float logits_mask_base_val, float logits_mask_except_val, std::vector<int> logit_mask_except_ids, bool return_logits);

liteqwen::Response get_generated(std::string request_id, bool is_incremental, bool force_interupt, bool no_stream);

void delete_request_ctx(std::string request_id);

#endif //_LITEQWEN_H
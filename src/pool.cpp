#include "pool.h"
#include "kv_cache.h"
#include "core_gpu.cuh"

std::mutex li_lock_;
std::mutex res_lock_;

namespace liteqwen {

std::string BatchInputPreparer::GetLoraName() {
    if (this->all_eos) {
        this->batch_lora_name = std::string("ANY");
        return this->batch_lora_name;
    } else {
        return this->batch_lora_name;
    }
}

void BatchInputPreparer::AddPrefill(ResponseContext* ctx_ref) { // std::string request_id, int inp_length, int max_length,
    // pool 已经检验过大小和lora符合的样本，添加到batch prefill里。
    std::string request_id = ctx_ref->request_id;
    int inp_length = ctx_ref->current_length;
    // Preparer内maxlen计算规则与KV-Cache的Reload方法内不同。max_length一般取整方便分配显存，max_new_tokens一般小于max_length，需要提前终止。
    // 这里较短的max_length在BatchUpdate时被用作eos判定。
    int max_length = this->max_seq_length;
    auto gen_cfg = ctx_ref->generation_config;
    if (gen_cfg.max_length > 1) {
        max_length = gen_cfg.max_length;
    } 
    if (gen_cfg.max_new_tokens > 1) {
        int new_token_restrict = gen_cfg.max_new_tokens + inp_length;
        if (new_token_restrict < max_length) {
            max_length = new_token_restrict;
        }
    }

    this->is_decoding = false;
    int prefill_pos = 0;
    int batch_idx = 0;
    if ((int)(this->prefill_examples.size()) > 0) {
        DynamicExample last_prefill = prefill_examples.back();
        prefill_pos = last_prefill.start_position + last_prefill.current_length;
        batch_idx = last_prefill.batch_idx + 1;
    }
    this->prefill_batch_maxlen = inp_length > this->prefill_batch_maxlen ? inp_length : this->prefill_batch_maxlen;

    this->prefill_examples.push_back(DynamicExample{request_id, -1, prefill_pos, inp_length, max_length, batch_idx, ctx_ref->generation_config.seed, ctx_ref->generation_config.temperature, ctx_ref->generation_config.top_p, ctx_ref->return_logits});
    uint8_t batch_idx8 = static_cast<uint8_t>(batch_idx);
    

    for (int i=0; i<inp_length; i++) {
        this->prefill_inp_ids.push_back((ctx_ref->tokens)[i]);
        this->prefill_bids.push_back(batch_idx8);
    }
    // this->prefill_starts[batch_idx+1] = prefill_pos+inp_length;
    prefill_starts.set_val(batch_idx+1, prefill_pos+inp_length); 
    this->prefill_bsz = batch_idx+1;

    this->prefill_temperatures.push_back(ctx_ref->generation_config.temperature);
    this->prefill_seeds.push_back(ctx_ref->generation_config.seed);
    this->prefill_req_ids.push_back(request_id);
    this->prefill_top_ps.push_back(ctx_ref->generation_config.top_p);
    this->prefill_return_lgts.push_back(ctx_ref->return_logits);
    
    if (this->batch_lora_name == std::string("ANY")) {
        this->batch_lora_name = ctx_ref->generation_config.adapter_name;
    } else {
        if (this->batch_lora_name != ctx_ref->generation_config.adapter_name) {
            printf("ERROR: Lora name obtained by InputPreparer's prev decoding is %s, not equal to adapter_name in context %s. This should not happen, but inference will continue though result may not be accurate. lora is switchable whence all_eos achieved in the batch.\n", this->batch_lora_name.c_str(), ctx_ref->generation_config.adapter_name.c_str());
        }
    }
}


void BatchInputPreparer::PrefillUpdate(int data_id, StringArray request_ids, std::vector<bool> is_eos, std::vector<int> token_ids, std::shared_ptr<ContextPool> pool, PipelineKVPool* kv_cache_ref) {
    std::vector<std::string> valid_request_ids;
    std::vector<std::string> invalid_request_ids;

    int dynamic_batch_idx = 0; // 即将更新后下次foward的bid
    int dynamic_start_position = 0;
    bool all_eos = true;
    if (this->decode_bsz > 0) {
        dynamic_batch_idx = this->decode_bsz;
        dynamic_start_position = this->decode_starts[dynamic_batch_idx];
        all_eos = false;
    }
    
    for (int bi=0; bi<request_ids.size(); bi++) {
        std::string new_req_id = this->prefill_req_ids[bi];
        if (new_req_id.length() == 0) {
            continue;
        }
        ResponseContext* double_check_ctx = pool->GetRes(new_req_id);
        if (double_check_ctx == nullptr) {
            printf("UPDATE_PREFILL: nullptr context trying to update&append prefill token for %s, possibly due to timeout, skip.\n", new_req_id.c_str());
            kv_cache_ref->free(new_req_id);
            pool->SetReloadOn(data_id);
            invalid_request_ids.push_back(new_req_id);
            continue;
        } else {
            valid_request_ids.push_back(new_req_id);
            if (this->batch_lora_name == std::string("ANY")) {
                this->batch_lora_name = double_check_ctx->generation_config.adapter_name;
            }
        }
    }

    for (auto req_item = (this->prefill_examples).begin(); req_item != (this->prefill_examples).end(); req_item++)  {
        bool still_valid_prefill = false;
        bool deleted_cache = false;
        bool example_eos;
        std::string matched_req_id;
        int example_tk_id;
        int prev_bid;  // 刚完成的推理中的bid
        // 需要prefill_examples中没被null ctx_ref删除的，且在request_ids中找到对应的。
        for (int bi=0; bi<request_ids.size(); bi++) {
            std::string req_id = request_ids[bi];
            if (req_id == req_item->request_id) {
                for (int invalid_id=0; invalid_id < invalid_request_ids.size(); invalid_id++) {
                    if (invalid_request_ids[invalid_id] == req_id) {
                        deleted_cache = true;
                        break;
                    }
                }
                still_valid_prefill = true;
                example_eos = is_eos[bi];
                example_tk_id = token_ids[bi];
                matched_req_id = req_id;
                prev_bid = bi;
                break;
            }
        }
        if (!still_valid_prefill) {
            printf("TOKEN_UPDATE WARNING: removing invalid prefill req=%s due to not in the provided Update(request_ids) param.\n", req_item->request_id.c_str());
            req_item = (--this->prefill_examples.erase(req_item));
            if (!deleted_cache) {
                kv_cache_ref->free(matched_req_id);
            }
            pool->SetReloadOn(data_id);
            continue;
        }

        // ======= 一系列eos条件判定 =========
        int example_maxlen = req_item->max_length;
        int example_curlen = req_item->current_length;
        auto ctx_ref = pool->GetRes(matched_req_id);
        if (example_eos) { // prefill直接生成eos
            printf("TOKEN_UPDATE: first frame eos for req=%s, no decoding for the request\n", matched_req_id.c_str());
            ctx_ref->Append(example_tk_id, true);
            kv_cache_ref->free(matched_req_id);
            pool->SetReloadOn(data_id);
        } else if (example_curlen >= example_maxlen ) { // prefill长度达到或超过maxlen，需要避免这样的input。
            if (example_tk_id == 0) {
                printf("TOKEN_UPDATE ERROR: generated token_id=0 for req=%s, this is unusual, should check for nan & inf values.\n", matched_req_id.c_str());
                ctx_ref->Append(example_tk_id, true);
            } else {
                printf("TOKEN_UPDATE WARNING: max_len causing early stop for req=%s\n", matched_req_id.c_str());
                ctx_ref->Append(example_tk_id, true);
            }                 
            kv_cache_ref->free(matched_req_id);
            pool->SetReloadOn(data_id);
        } else {
            if (example_tk_id == 0) {
                printf("TOKEN_UPDATE ERROR: generated token_id=0 for req=%s, this is unusual, should check for nan & inf values.\n", matched_req_id.c_str());
                ctx_ref->Append(example_tk_id, true);
                kv_cache_ref->free(matched_req_id);
                pool->SetReloadOn(data_id);
            } else {
                bool suc = ctx_ref->Append(example_tk_id, false);
                if (!suc) {
                    // ctx_ref终止生成
                    printf("TOKEN_UPDATE TERMINATE: repeated generation, stop for req=%s, this is unusual, should check for nan & inf values.\n", matched_req_id.c_str());
                    kv_cache_ref->free(matched_req_id);
                    pool->SetReloadOn(data_id);                          
                } else {
        // ======= 正常append ========
                    printf("TOKEN_UPDATE: first frame token generated for req=%s, new_pos=%i, new_len=%i, token=%i, bid=%i, appending to context\n", matched_req_id.c_str(), dynamic_start_position, example_curlen+1, example_tk_id, dynamic_batch_idx);                        
                    all_eos = false;
                    this->decode_inp_ids.push_back(example_tk_id);
                    uint8_t bid8 = static_cast<uint8_t>(dynamic_batch_idx);
                    for (int j=0; j< example_curlen+1; j++) {
                        this->decode_bids.push_back(bid8);
                    }
                    this->decode_starts.push_back(dynamic_start_position+example_curlen+1);
                    this->decode_req_ids.push_back(matched_req_id);
                    this->decode_temperatures.push_back(req_item->temperature);
                    this->decode_top_ps.push_back(req_item->top_p);
                    this->decode_seeds.push_back(req_item->seed);
                    this->decode_return_lgts.push_back(req_item->return_lgt);

                    this->decoding_examples.push_back(DynamicExample{matched_req_id, example_tk_id, dynamic_start_position, example_curlen+1, example_maxlen, dynamic_batch_idx, req_item->seed, req_item->temperature, req_item->top_p, req_item->return_lgt});
                    dynamic_start_position += (example_curlen+1);
                    dynamic_batch_idx += 1;
                }
            }
        }
    }

    this->prefill_examples.clear();
    this->decode_bsz = dynamic_batch_idx;
    this->all_eos = all_eos;
    this->is_decoding = true;
    if (all_eos) {
        this->batch_lora_name = std::string("ANY");
    }
}

void BatchInputPreparer::DecodeUpdate(int data_id, StringArray request_ids, std::vector<bool> is_eos, std::vector<int> token_ids, std::shared_ptr<ContextPool> pool, PipelineKVPool* kv_cache_ref) {
    std::vector<std::string> valid_request_ids;
    std::vector<std::string> invalid_request_ids;

    int dynamic_batch_idx = 0; // 即将更新后下次foward的bid
    int dynamic_start_position = 0;
    bool all_eos = true;

    for (int bi=0; bi<request_ids.size(); bi++) {
        std::string new_req_id = request_ids[bi];
        if (new_req_id.length() == 0) {
            continue;
        }
        
        ResponseContext* double_check_ctx = pool->GetRes(new_req_id);
        if (double_check_ctx == nullptr) {
            printf("UPDATE_DECODE: nullptr context trying to update&append prefill token for %s, possibly due to eos or timeout, skip.\n", new_req_id.c_str());
            kv_cache_ref->free(new_req_id);
            pool->SetReloadOn(data_id);
            invalid_request_ids.push_back(new_req_id);
            continue;
        } else {
            valid_request_ids.push_back(new_req_id);
            if (this->batch_lora_name == std::string("ANY")) {
                this->batch_lora_name = double_check_ctx->generation_config.adapter_name;
            }
        }
    }

    for (auto req_item = (this->decoding_examples).begin(); req_item != (this->decoding_examples).end(); req_item++) {
        bool still_valid_decode = false;
        bool deleted_cache = false;
        bool example_eos;
        std::string matched_req_id;
        int example_tk_id;
        int prev_bid;  // 刚完成的推理中的bid
        // 需要decoding_examples中没被null ctx_ref删除的，且在request_ids中找到对应的。
        for (int bi=0; bi<request_ids.size(); bi++) {
            std::string req_id = request_ids[bi];
            if (req_id == req_item->request_id) {
                for (int invalid_id=0; invalid_id < invalid_request_ids.size(); invalid_id++) {
                    if (invalid_request_ids[invalid_id] == req_id) {
                        deleted_cache = true;
                        break;
                    }
                }
                still_valid_decode = true;
                example_eos = is_eos[bi];
                example_tk_id = token_ids[bi];
                matched_req_id = req_id;
                prev_bid = bi;
                break;
            }
        }
        if (!still_valid_decode) {
            printf("TOKEN_UPDATE WARNING: removing invalid prefill req=%s due to not in the provided Update(request_ids) param.\n", req_item->request_id.c_str());
            req_item = (--this->decoding_examples.erase(req_item));
            if (!deleted_cache) {
                kv_cache_ref->free(matched_req_id);
            }
            pool->SetReloadOn(data_id);
            continue;
        }

        int example_maxlen = req_item->max_length;
        int example_curlen = req_item->current_length;
        auto ctx_ref = pool->GetRes(matched_req_id);
        if (example_eos) { // 正常eos
            int gen_len = req_item->current_length - ctx_ref->input_length;
            // 注意：这里eos时打印的batch_idx可能与prefill时不同，因为生成过程中不断有其他样本eos之后清空，使该样本的batch_idx可能减少。
            printf("TOKEN_UPDATE: EOS for req=%s, mini_bid=%i, start=%i, inp/gen=%i/%i\n", matched_req_id.c_str(), req_item->batch_idx, req_item->start_position, ctx_ref->input_length, gen_len);
            ctx_ref->Append(example_tk_id, true);
            kv_cache_ref->free(matched_req_id);
            pool->SetReloadOn(data_id);
            req_item = (--this->decoding_examples.erase(req_item));
        } else if (example_curlen >= example_maxlen ) { // 触发样本指定maxlen的终止条件。
            if (example_tk_id == 0) {
                printf("TOKEN_UPDATE ERROR: generated token_id=0 for req=%s, this is unusual, should check for nan & inf values.\n", matched_req_id.c_str());
                ctx_ref->Append(example_tk_id, true);
            } else {
                printf("TOKEN_UPDATE WARNING: max_len causing early stop for req=%s\n", matched_req_id.c_str());
                ctx_ref->Append(example_tk_id, true);
            }                 
            kv_cache_ref->free(matched_req_id);   
            pool->SetReloadOn(data_id);
            req_item = (--this->decoding_examples.erase(req_item));
        } 
        else { // 正常decode添加新token，以及当产生token_id=0（大概率推理错误，需要检查hiddens值）
            if (example_tk_id == 0) {
                printf("TOKEN_UPDATE ERROR: generated token_id=0 for req=%s, this is unusual, should check for nan & inf values.\n", matched_req_id.c_str());
                ctx_ref->Append(example_tk_id, true);
                kv_cache_ref->free(matched_req_id);
                pool->SetReloadOn(data_id);
                req_item = (--this->decoding_examples.erase(req_item));
            } else {
                bool suc = ctx_ref->Append(example_tk_id, false);
                if (!suc) {
                    // ctx_ref终止生成
                    printf("TOKEN_UPDATE TERMINATE: repeated generation, stop for req=%s, this is unusual, should check for nan & inf values.\n", matched_req_id.c_str());
                    kv_cache_ref->free(matched_req_id);
                    pool->SetReloadOn(data_id);
                    req_item = (--this->decoding_examples.erase(req_item));                            
                } else {
                    all_eos = false;
                    this->decode_inp_ids.push_back(example_tk_id);
                    uint8_t bid8 = static_cast<uint8_t>(dynamic_batch_idx);
                    for (int j=0; j< example_curlen+1; j++) {
                        this->decode_bids.push_back(bid8);
                    }
                    this->decode_starts.push_back(dynamic_start_position+example_curlen+1);
                    this->decode_req_ids.push_back(matched_req_id);
                    this->decode_top_ps.push_back(req_item->top_p);
                    this->decode_temperatures.push_back(req_item->temperature);
                    this->decode_seeds.push_back(req_item->seed);
                    this->decode_return_lgts.push_back(req_item->return_lgt);

                    req_item->next(example_tk_id, dynamic_start_position, dynamic_batch_idx);
                    dynamic_start_position += (example_curlen+1);
                    dynamic_batch_idx += 1;
                }
            }
        }
    }

    this->decode_bsz = dynamic_batch_idx;
    this->all_eos = all_eos;
    this->is_decoding = true;
    if (all_eos) {
        this->batch_lora_name = std::string("ANY");
    }
}


// void BatchInputPreparer::UploadInputs(bool is_prefill, int inp_gpu_id, int out_gpu_id, const Data& input_ids, const Data& inp_batch_ids, const Data& query_pos_starts, const Data& key_pos_starts, const Data& temperature, float* cpu_temperatures, const Data& seeds_tensor, int* cpu_seeds, const Data& top_ps, float* cpu_top_ps, int dynamic_bsz_check) {
//     SetDevice(inp_gpu_id);
//     // int dynamic_bl = (int)(this->input_bids.size());
//     // int dynamic_bsz = (int)(this->input_starts.size()) - 1;
//     int dynamic_bl = (int)(this->input_bids_ct);
//     int dynamic_bsz = this->dynamic_bsz;
//     if (dynamic_bsz != dynamic_bsz_check) {
//         printf("PREPARER ERROR: request ids len obtained in prefill fetch (%i)!= (%i) dynamic_bsz implied in preparer's input_starts vector, check whether prepare is correct.\n", dynamic_bsz_check, dynamic_bsz);
//     }
//     int upload_ids_len = is_prefill ? dynamic_bl : dynamic_bsz;
//     // int* cpu_input_ids = this->input_ids.data();
//     // uint8_t* cpu_inp_bids = this->input_bids.data();
//     // int* cpu_starts = this->input_starts.data();
//     int* cpu_input_ids = this->input_ids;
//     uint8_t* cpu_inp_bids = this->input_bids;
//     int* cpu_starts = this->input_starts;
//     QuickUploadData(DataType::INT32, (void*)input_ids.cudaData, (uint8_t*)cpu_input_ids, inp_gpu_id, 0, 0, upload_ids_len);
//     QuickUploadData(DataType::INT8, (void*)inp_batch_ids.cudaData, (uint8_t*)cpu_inp_bids, inp_gpu_id, 0, 0, dynamic_bl);
//     QuickUploadData(DataType::INT32, (void*)query_pos_starts.cudaData, (uint8_t*)cpu_starts, inp_gpu_id, 0, 0, dynamic_bsz+1);
//     QuickUploadData(DataType::INT32, (void*)key_pos_starts.cudaData, (uint8_t*)cpu_starts, inp_gpu_id, 0, 0, dynamic_bsz+1);
//     // UploadCastFp32ToFp16Data((void*)temperature.cudaData, cpu_temperatures, gpu_id, 0, 0, dynamic_bsz);
//     SetDevice(out_gpu_id);
//     QuickUploadData(DataType::FLOAT32, (void*)temperature.cudaData, (uint8_t*)cpu_temperatures, out_gpu_id, 0, 0, dynamic_bsz);
//     QuickUploadData(DataType::INT32, (void*)seeds_tensor.cudaData, (uint8_t*)cpu_seeds, out_gpu_id, 0, 0, dynamic_bsz);
//     QuickUploadData(DataType::FLOAT32, (void*)top_ps.cudaData, (uint8_t*)cpu_top_ps, out_gpu_id, 0, 0, dynamic_bsz);
//     SetDevice(inp_gpu_id);
// }

void BatchInputPreparer::UploadInputs(bool is_prefill, int inp_gpu_id, int out_gpu_id, const Data& input_ids, const Data& inp_batch_ids, const Data& query_pos_starts, const Data& key_pos_starts, const Data& temperature, const Data& seeds_tensor, const Data& top_ps, int dynamic_bsz_check) {
    SetDevice(inp_gpu_id);
    
    if (is_prefill) {
        int dynamic_bsz = this->prefill_bsz;
        int dynamic_bl = (int)(this->prefill_inp_ids.size());
        // printf("uploading prefills: dynamic_bl=%i, dynamic_bsz=%i, prefill_inp_ids[bl-1]=%i, starts[bsz]=%i\n", dynamic_bl, dynamic_bsz, this->prefill_inp_ids[dynamic_bl-1], this->prefill_starts[dynamic_bsz]);
        QuickUploadData(DataType::INT32, (void*)input_ids.cudaData, (uint8_t*)(this->prefill_inp_ids.data()), inp_gpu_id, 0, 0, dynamic_bl);
        QuickUploadData(DataType::INT8, (void*)inp_batch_ids.cudaData, (uint8_t*)(this->prefill_bids.data()), inp_gpu_id, 0, 0, dynamic_bl);
        QuickUploadData(DataType::INT32, (void*)query_pos_starts.cudaData, (uint8_t*)(this->prefill_starts.data()), inp_gpu_id, 0, 0, dynamic_bsz+1);
        QuickUploadData(DataType::INT32, (void*)key_pos_starts.cudaData, (uint8_t*)(this->prefill_starts.data()), inp_gpu_id, 0, 0, dynamic_bsz+1);
        SetDevice(out_gpu_id);
        QuickUploadData(DataType::FLOAT32, (void*)temperature.cudaData, (uint8_t*)(this->prefill_temperatures.data()), out_gpu_id, 0, 0, dynamic_bsz);
        QuickUploadData(DataType::INT32, (void*)seeds_tensor.cudaData, (uint8_t*)(this->prefill_seeds.data()), out_gpu_id, 0, 0, dynamic_bsz);
        QuickUploadData(DataType::FLOAT32, (void*)top_ps.cudaData, (uint8_t*)(this->prefill_top_ps.data()), out_gpu_id, 0, 0, dynamic_bsz);
    } else {
        int dynamic_bsz = this->decode_bsz;
        int dynamic_bl = (int)(this->decode_bids.size());
        QuickUploadData(DataType::INT32, (void*)input_ids.cudaData, (uint8_t*)(this->decode_inp_ids.data()), inp_gpu_id, 0, 0, dynamic_bsz);
        QuickUploadData(DataType::INT8, (void*)inp_batch_ids.cudaData, (uint8_t*)(this->decode_bids.data()), inp_gpu_id, 0, 0, dynamic_bl);
        QuickUploadData(DataType::INT32, (void*)query_pos_starts.cudaData, (uint8_t*)(this->decode_starts.data()), inp_gpu_id, 0, 0, dynamic_bsz+1);
        QuickUploadData(DataType::INT32, (void*)key_pos_starts.cudaData, (uint8_t*)(this->decode_starts.data()), inp_gpu_id, 0, 0, dynamic_bsz+1);
        SetDevice(out_gpu_id);
        QuickUploadData(DataType::FLOAT32, (void*)temperature.cudaData, (uint8_t*)(this->decode_temperatures.data()), out_gpu_id, 0, 0, dynamic_bsz);
        QuickUploadData(DataType::INT32, (void*)seeds_tensor.cudaData, (uint8_t*)(this->decode_seeds.data()), out_gpu_id, 0, 0, dynamic_bsz);
        QuickUploadData(DataType::FLOAT32, (void*)top_ps.cudaData, (uint8_t*)(this->decode_top_ps.data()), out_gpu_id, 0, 0, dynamic_bsz);        
    }
    SetDevice(inp_gpu_id);
}

void TimeoutFlush(std::string, ResponseContext) {
    // timeout后的回调函数，暂无使用。
    // printf("callback_fn: should flush pool and cache due to timeout.\n");
}

ContextPool::ContextPool(int max_queue_size, int timeout) {
    // size_t store_size = max_thread_num * sizeof(ResponseContext);
    // this->max_size_ = store_size;
    this->max_queue_size = max_queue_size;
    this->timeout = timeout;
    this->time_out_ = static_cast<uint32_t>(timeout);
    std::function<void(std::string, ResponseContext)> callback_fn;
    callback_fn = TimeoutFlush;
    this->expired_callback_ = callback_fn;
    this->deleting_keys = std::list<std::string>();
};

void ContextPool::SetDefaultMaxlen(int max_sequence_length) {
    this->default_maxlen = max_sequence_length;
}

void ContextPool::UnsafeDelete(std::string key) {
    if (finished_.find(key) != finished_.end())
    {
        NodeIter iter = finished_[key];
        std::deque<std::shared_ptr<ContextPool::Node>> del_nd(iter, res_list_.end());
        auto deleting_ctx = ((del_nd.front()).get())->value;
        finished_.erase(key);
        printf("POOL: queue after cleaning %s remaining queue size=%i\n", key.c_str(), (int)(finished_.size()));
        expired_callback_(key, deleting_ctx);
    }        
}

void ContextPool::DELETE(std::string key) {
    res_lock_.lock();
    // 删除指定key
    if (finished_.find(key) != finished_.end())
    {
        NodeIter iter = finished_[key];
        std::deque<std::shared_ptr<ContextPool::Node>> del_nd(iter, res_list_.end());
        auto deleting_ctx = ((del_nd.front()).get())->value;
        finished_.erase(key);
        printf("POOL: queue after cleaning %s remaining queue size=%i\n", key.c_str(), (int)(finished_.size()));
        expired_callback_(key, deleting_ctx);
    }

    //删除超时的res
    auto time_now = std::chrono::system_clock::now();
    std::vector<std::string> deleting_keys;
    for (auto res_itm=finished_.begin(); res_itm != finished_.end(); res_itm++) {
        auto res_time = (*(res_itm->second))->timestamp;
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_now - res_time);
        
        if (diff.count() > 1.5*time_out_) {
            deleting_keys.push_back(res_itm->first);
        }
    }
    for (std::string del_key :deleting_keys) {
        printf("liteqwen backend removing expired finished results %s.\n", del_key.c_str());
        this->UnsafeDelete(del_key);
    }
    res_lock_.unlock();
}

void ContextPool::Add(std::string key, ResponseContext value)
{
    li_lock_.lock();

    // // 如果队列满，删除最旧的。风险不可控，所以会在submit时阻塞等候，直到队列长度减少。
    // if (list_.size() >= max_size_+1)
    // {
    //     printf("liteqwen backend queue maximum is reached, removing earliest.");
    //     auto oldest = list_.back();
    //     list_.pop_back();
    //     map_.erase(oldest->key);
    //     expired_callback_(oldest->key, oldest->value);
    // }

    // 如果key已存在，覆盖旧内容。可能导致生成长度出问题，最好保证同一时间没有两个相同的key被加入队列。
    if (map_.find(key) != map_.end())
    {
        printf("context queue found existing key: %s, new request will overwrite previous.\n", key.c_str());
        NodeIter iter = map_[key];
        list_.erase(iter);
    }

    auto timestamp = std::chrono::system_clock::now();
    NodePtr node = std::make_shared<Node>(Node{key, value, timestamp, false});
    // printf("pushing new node: %s\n", key.c_str());
    list_.push_front(node);
    map_[key] = list_.begin();

    li_lock_.unlock();
}

ResponseContext* ContextPool::GetRes(std::string key, bool refresh)
{
    // 从结果队列获取ResponseContext的指针
    if (finished_.find(key) != finished_.end())
    {
        auto nd = (*finished_[key]);
        if (refresh) {
            // printf("refreshing ts for %s\n", key.c_str());
            auto new_ts = std::chrono::system_clock::now();
            nd.get() -> SetTimestamp(new_ts);
        }
        auto ctx_ref = nd.get()->GetValuePtr();
        // printf("found node in pool:%s, return the pointer %p\n", key.c_str(), ctx_ref);
        return ctx_ref;
    }
    else
    {
        // printf("key not found in map, returning empty ResponseContext\n");
        return nullptr;
    }
}

ResponseContext* ContextPool::GetRes(std::string key){
    return ContextPool::GetRes(key, false);
}

ResponseContext* ContextPool::GetPtr(std::string key, bool refresh)
{
    // 获取流式输入输出请求的struct在pool中的引用，一个struct对应一个request_id。refresh可以在get时顺便重置超时计时。
    if (map_.find(key) != map_.end())
    {
        auto nd = (*map_[key]);
        if (refresh) {
            // printf("refreshing ts for %s\n", key.c_str());
            auto new_ts = std::chrono::system_clock::now();
            nd.get() -> SetTimestamp(new_ts);
        }
        auto ctx_ref = nd.get()->GetValuePtr();
        // printf("found node in pool:%s, return the pointer %p\n", key.c_str(), ctx_ref);
        return ctx_ref;
    }
    else
    {
        printf("returning nullptr key=%s\n", key.c_str());
        return nullptr;
    }
}

ResponseContext* ContextPool::GetPtr(std::string key){
    return ContextPool::GetPtr(key, false);
}

std::vector<AllocateParam> ContextPool::Reload(int data_id, std::string preparer_lora_name, bool preparer_is_empty, PipelineKVPool* kv_cache_ref) {
    // 给推理中的batch重新填充新的推理请求，在BatchInputPreparer中推理样本为空或任意样本eos后，或while达到了reload轮数间隔时进行。
    // 如果加入了新prefill样本，则返回新样本的request_id list。
    // 如果没有合格的新prefill样本（由于cache满或lora不匹配或达到最大动态batch size），则返回空vector。
    // 如果BatchInputPreparer内容为空且无法加入新prefill，则返回空vector
    // 该方法有锁，应该避免不同线程的复杂逻辑长期占用。
    // 判定是否被kv-cache限制导致无法增加batch的方法在 kv_cache_ref->search_block_sequence里，会根据样本最大长度判定是否有足够大的空闲且contiguous的显存来保存该长度的kv-cache。
    // 分配方式使用了cache block链表 + 贪心的方法，所以会存在kv-cache碎片化的问题，所以建议max_length都使用2^n这样比较规范的长度，并将相同lora的较短请求放在一个batch request内提交。

    std::vector<AllocateParam> tobe_allocated;
    if (!this->CanReload(data_id)) {
        // 限制Reload频率，避免锁浪费时间。只在batch内任意样本结束推理、终止推理或达到reload间隔计数之后，才允许reload补充prefill。
        return tobe_allocated;
    }

    li_lock_.lock();
    Expired();

    bool skip_for_break = false;
    std::string infering_lora = preparer_lora_name;
    std::stringstream  batch_info_joined;

    if (!list_.empty()) {
        printf("trying to fetch from back of list, len=%i\n", (int)list_.size());
        for (auto it_oldest = list_.rbegin(); it_oldest != list_.rend(); it_oldest++) {

            std::string ctx_id = ((*it_oldest).get())->value.request_id;
            if (skip_for_break) {
                break;
            }

            if (((*it_oldest).get())->value.processing_data_id != -1) {
                // lock间隙被其他gpu线程锁定了该样本，则无视这个样本继续处理下一条。
                continue;
            } else {
                std::string ctx_adapter_name = (((*it_oldest).get())->value).generation_config.adapter_name;
                // 分桶强制要求相同lora才能在decoding阶段动态加入prefill。当队列中lora切换时，需要等已经在推理的batch的每个样本都完成，并且当前batch内没有不同的lora。
                if (infering_lora != std::string("ANY") && ctx_adapter_name != infering_lora) {
                    if (!preparer_is_empty || tobe_allocated.size() > 0) {
                        skip_for_break = true;
                        // printf("DEBUG: lora mismatched, infering until all_eos. infering=%s, ctx_adapter=%s\n", infering_lora.c_str(), ctx_adapter_name.c_str());
                        batch_info_joined << "|LORA BOUNDARY|" << infering_lora.c_str() << " != " << ctx_adapter_name.c_str();
                        break;
                    }
                }

                // 动态batch数不能超过uint8最大值，防止batch_id溢出。
                int cache_example_ct = kv_cache_ref->get_caching_count();
                if (tobe_allocated.size() + cache_example_ct + 1 > kv_cache_ref->max_dynamic_bsz) {
                    skip_for_break = true;
                    batch_info_joined << "|BSZ BOUNDARY| existing decode + new bsz = " << cache_example_ct << "+" << tobe_allocated.size();
                    break;
                }

                // 获取样本推理最大长
                auto gen_cfg = (((*it_oldest).get())->value).generation_config;
                std::string res_id = (((*it_oldest).get())->value).request_id;
                int example_maxlen;
                if (gen_cfg.max_new_tokens < 1 && gen_cfg.max_length < 1) {
                    example_maxlen = this->default_maxlen;
                } else if (gen_cfg.max_length > 0) {
                    example_maxlen = gen_cfg.max_length;
                } else {
                    int inp_len = (((*it_oldest).get())->value).input_length;
                    example_maxlen = inp_len + gen_cfg.max_new_tokens;
                }
                
                // 分桶满足后，需要cache能够容纳样本maxlen的空间。尝试逐个添加，直到cache添加失败(cache占满)。
                AllocateParam block_info = (kv_cache_ref->pipeline_caches)[0]->search_block_sequence(res_id, example_maxlen, &tobe_allocated);
                if (!block_info.successful) {
                    // printf("failure at searching for %s\n", block_info.request_id.c_str());
                    skip_for_break = true;
                    batch_info_joined << "|CACHE BOUNDARY| failed kv allocate len=" <<  example_maxlen << " req_id=" << res_id.c_str();
                    break;
                }

                infering_lora = ctx_adapter_name;
                block_info.set_lora(infering_lora);
                // printf("within reload, setting lora = %s for req=%s\n", infering_lora.c_str(), res_id.c_str());
                tobe_allocated.push_back(block_info);
                batch_info_joined << "(" << std::to_string(example_maxlen).c_str() << "," << res_id.c_str() << ") "; 

                // printf("fetching a new request %s, setting its data_id to %i\n", (((*it_oldest).get())->value).request_id.c_str(), data_id);
                (((*it_oldest).get())->value).SetGenerateFlag(data_id);
                
                // 添加至输出序列res_list_
                res_lock_.lock();
                NodePtr fetched = (*it_oldest);
                res_list_.push_front(fetched);
                finished_[res_id] = res_list_.begin();
                res_lock_.unlock();

                // 从输入序列list_删除
                list_.erase(--(it_oldest.base()));
                map_.erase(res_id);
                
            }
        } // end for
    }

    li_lock_.unlock();
    int reloaded_prefill_ct = (int)(tobe_allocated.size());
    
    if (reloaded_prefill_ct>0) {
        printf("BATCHING: %i prefills in one batch, lora=%s, batch_info=%s\n", reloaded_prefill_ct, infering_lora.c_str(), batch_info_joined.str().c_str());
    }
    batch_info_joined.clear();
    // 关闭reload持续reload_interval轮forward，或者直到任意样本eos时提前SetReloadOn.
    this->SetReloadOff(data_id);

    return tobe_allocated;
}


int ContextPool::GetLength() {
    return (int)(list_.size());
}

void ContextPool::Expired()
{
    // 清理pool中的超时request。pool从front到end按照从新到旧的顺序排列。

    auto time_now = std::chrono::system_clock::now();
    
    int list_length = (int)list_.size();
    int process_ct = 0; // 遍历计数，防止死循环
    int removal_ct = 0; // 被超时清理的struct计数
    while ((!list_.empty()) && (process_ct<=list_length))
    {
        process_ct += 1;
        auto oldest = list_.back();
        
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            time_now - oldest->timestamp);

        if (oldest->value.processing_data_id>=0) {
            // request正在被其他进程锁定fetch，直接放弃fetch。
            break;
        }

        if (oldest->time_updating && diff.count() <= time_out_){
            // 未超时的情况下time_update刷新机制被触发（GetPtr方法的refresh参数为true时），自动移至front。
            auto lifting_key = (oldest -> key);
            // printf("lifting %s to top due to ts update mark.\n", lifting_key.c_str());
            list_.pop_back();
            map_.erase(oldest->key);
            oldest->FinishingSetTimestamp();
            list_.push_front(oldest);
            map_[lifting_key] = list_.begin();
            continue;
        }
        
        if (diff.count() > time_out_)
        {
            // 超时删除
            list_.pop_back();
            int listlen = (int)list_.size();
            std::string back_id = (list_.back()->value).request_id;
            if (listlen>0) {
                printf("Removing due to timeout: %s, dt=%i, threshold=%i. Remaining list length=%i, back_id=%s\n", (oldest->key).c_str(), (int)(diff.count()), (int)time_out_, listlen, back_id.c_str());
            } else {
                printf("Removing due to timeout: %s, dt=%i, threshold=%i. Remaining list length=%i, empty list\n", (oldest->key).c_str(), (int)(diff.count()), (int)time_out_, listlen);
            }

            if (oldest->key != back_id) {
                map_.erase(oldest->key);
                expired_callback_(oldest->key, oldest->value);
                removal_ct += 1;
            } else {
                printf("ERROR: back_id is same as oldest->key: %s\n", back_id.c_str());
                continue;
            }
        }
        else
        {
            if (removal_ct > 1 || process_ct>=list_length) {
                // 节约时间，每次fetch清理最多2个struct就可以保证list能够被逐渐清理干净。
                break;
            }
        }
    }
}

void ContextPool::SetReloadOn(int data_id){
    this->reload_switch[data_id] = 0;
}

void ContextPool::SetReloadOff(int data_id){
    this->reload_switch[data_id] = this->minimum_reload_interval;
    // printf("setting countdown=max\n");
}

bool ContextPool::CanReload(int data_id) {
    for (auto item= this->reload_switch.begin(); item != this->reload_switch.end(); item++) {
        if (item->first == data_id) {
            // printf("countdown%i\n", item->second);
            if (item->second <= 0) {
                return true;
            } else {
                this->ReloadIntervalCountdown(data_id, item->second - 1);
                return false;
            }
        }
    }

    this->SetReloadOn(data_id);
    return true;
}

void ContextPool::ReloadIntervalCountdown(int data_id, int new_ct) {
    this->reload_switch[data_id] = new_ct;
}

} // namespace liteqwen
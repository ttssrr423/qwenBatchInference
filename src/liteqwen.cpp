#include "liteqwen.h"
#include "generate.h"

std::mutex load_lock;

namespace liteqwen{
    bool get_loading_finished() {
        return liteqwen::loading_finished;
    }
    
    void set_loading_finished(int ct) {
        // 每个data_id完成加载后，load_finish_ct+=1，累加直到dp_size，所有进程完成加载，释放barrier。
        load_lock.lock();
        liteqwen::load_finish_ct += ct;
        printf("loading_ct=%i\n", liteqwen::load_finish_ct);
        if (liteqwen::load_finish_ct >= liteqwen::qwen2params.dp_size) {
            liteqwen::loading_finished = true;
            printf("<============ALL DATA PARALLEL INITIALIZED==========\n");
        }
        load_lock.unlock();
    }
}

void init_empty_qwen2(int world_size, int data_parallel_size, std::string json_config_path, std::vector<int> layer2device_list, int max_dynamic_bsz, int max_sequence_length, int max_queue_size, int timeout) {
    
    liteqwen::qwen2params.Init(world_size, data_parallel_size, json_config_path, layer2device_list, max_dynamic_bsz, max_sequence_length);
    liteqwen::qwen2_ptr = (std::shared_ptr<liteqwen::Qwen2Params>)(&liteqwen::qwen2params);

    liteqwen::lora_adapters = std::make_shared<std::map<std::string, liteqwen::LoraConfig>>();
    liteqwen::originalCPUWeights = std::make_shared<std::map<std::string, liteqwen::Data*>>();
    liteqwen::Q4linearMetaMap = std::make_shared<std::map<std::string, liteqwen::Q4LinearMeta>>();

    // 下面thread_pool的存储方式是等价的，仅限class，而非struct。
    // liteqwen::thread_pool = std::make_shared<liteqwen::ContextPool>(max_queue_size, timeout, max_dynamic_bsz);
    liteqwen::thread_pool = (std::shared_ptr<liteqwen::ContextPool>)(new liteqwen::ContextPool(max_queue_size, timeout));
    return;
}

void add_qwen2_weight(const std::string& weight_name, std::vector<int> shape, liteqwen::DataType dtype, liteqwen::DataType oriDataType, void *oriData) {
    // 添加到cpu中按照ndarray的格式存储
    liteqwen::Data* data_ref = new liteqwen::Data(oriDataType, shape, -1, false);
    data_ref->Allocate();
    std::map<std::string, liteqwen::Data*>* map_ref = liteqwen::originalCPUWeights.get();
    map_ref->insert(std::pair<std::string, liteqwen::Data*>(weight_name, data_ref));
    size_t original_data_size = data_ref->numel() * data_ref->unitSize / data_ref->unitSizeDiv * sizeof(uint8_t);
    memcpy(data_ref->cpuData, oriData, original_data_size);
    return;
}

int submit_inference(const std::string& request_id, std::vector<int> input_ids, const liteqwen::GenerationConfig& gen_cfg, float logits_mask_base_val, float logits_mask_except_val, std::vector<int> logit_mask_except_ids, bool return_logits) {    
    while(liteqwen::thread_pool->GetLength() >= liteqwen::thread_pool->max_queue_size) {
        // 队列已满，提交失败。
        printf("liteqwen backend queue is full until timeout for request_id=%s, submit success=0\n", request_id.c_str());
        return 0;
    }
    liteqwen::ResponseContext waiting_ctx;
    waiting_ctx.Init(request_id, input_ids, gen_cfg, logits_mask_base_val, logits_mask_except_val, logit_mask_except_ids, return_logits);
    liteqwen::thread_pool->Add(request_id, waiting_ctx);
    printf("POOL: Added to queue: request_id=%s, queue_size=%i\n", request_id.c_str(), liteqwen::thread_pool->GetLength());
    return 1;
}

void add_lora_adapter(liteqwen::LoraConfig lora_cfg) {
    auto& lora_map = *(liteqwen::lora_adapters.get());
    lora_map.insert(std::make_pair(lora_cfg.model_name, lora_cfg));
}

void add_q4_linear_meta(liteqwen::Q4LinearMeta q4_meta) {
    auto& q4_map = *(liteqwen::Q4linearMetaMap.get());
    q4_map.insert(std::make_pair(q4_meta.prefix, q4_meta));
}

void start_thread_loops() {
    // lora以及cpu weight都加载完后调用，开始各data_id线程的加载和初始化。
    printf("lora configs loaded:\n");
    std::map<std::string, liteqwen::LoraConfig>* lora_map = liteqwen::lora_adapters.get();
    for(std::map<std::string, liteqwen::LoraConfig>::iterator it = lora_map->begin(); it != lora_map->end(); ++it) {
        std::string lora_module_str = liteqwen::join((it->second).target_modules, std::string(","));
        printf("%s: r=%i, modules=[%s]\n", (it->first).c_str(), (it->second).r, lora_module_str.c_str());
    }

    int dp_size = liteqwen::qwen2params.dp_size;
    liteqwen::Qwen2Params model_par = liteqwen::qwen2params;

    // printf("base_model num_layers=%i, max_dynamic_bsz=%i, BL=%i. Scattering to %i data parallels.\n", 0, model_par.num_layers, model_par.max_dynamic_bsz, model_par.max_sequence_length, model_par.dp_size);
    std::map<std::string, liteqwen::Data*>* cpu_weights_ref = liteqwen::originalCPUWeights.get();
    std::map<std::string, liteqwen::LoraConfig>* lora_ref = liteqwen::lora_adapters.get();
    std::map<std::string, liteqwen::Q4LinearMeta>* quant_ref = liteqwen::Q4linearMetaMap.get();

    for (int di=0; di< model_par.dp_size; di++) {
        liteqwen::worker_threads.push_back(new std::thread(liteqwen::Generate, di, liteqwen::thread_pool, model_par, cpu_weights_ref, lora_ref, quant_ref));
    }
}


liteqwen::Response get_generated(std::string request_id, bool is_incremental, bool force_interupt, bool no_stream) {
    using namespace std::chrono_literals;
    liteqwen::ResponseContext* ctx_ptr = (liteqwen::ResponseContext*)liteqwen::thread_pool->GetRes(request_id);
    if (ctx_ptr == nullptr) {
        ctx_ptr = (liteqwen::ResponseContext*)liteqwen::thread_pool->GetPtr(request_id);
    }

    int status = 0; // 0：排队，1：生成中，2：生成完毕，-1：被终止生成
    liteqwen::Response resp;
    while(status == 0) {
        if (ctx_ptr != nullptr) {
            // printf("node %s length comparing input length:%i?=%i\n", request_id.c_str(), ctx_ptr->input_length, ctx_ptr->current_length);
            if (ctx_ptr->input_length == ctx_ptr->current_length) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                ctx_ptr = (liteqwen::ResponseContext*)liteqwen::thread_pool->GetRes(request_id);
                if (ctx_ptr == nullptr) {
                    ctx_ptr = (liteqwen::ResponseContext*)liteqwen::thread_pool->GetPtr(request_id);
                }

                if (ctx_ptr == nullptr) {
                    status = -1;
                    break;
                } else if (ctx_ptr->isEnding) {
                    status = 2;
                    break;
                } else if (ctx_ptr->input_length < ctx_ptr->current_length) {
                    // printf("req_id:%s beging generating, status set to 1. inp_len=%i, prev_len=%i, cur_len=%i\n", request_id.c_str(), ctx_ptr->input_length, ctx_ptr->prev_length, ctx_ptr->current_length);
                    status = 1;
                    break;
                }
                status = 0;
                // continue; // 不能对不在生成中（status=0等待）的样本进行阻塞，会影响正在生成的流式。只能先break，返回结果给python，再在python内异步asleep。
                break;
            } else {
                if (ctx_ptr->isEnding) {
                    status = 2;
                    break;
                } else if (ctx_ptr->input_length < ctx_ptr->current_length) {
                    status = 1;
                    // printf("req_id:%s beging generating, status set to 1. inp_len=%i, prev_len=%i, cur_len=%i\n", request_id.c_str(), ctx_ptr->input_length, ctx_ptr->prev_length, ctx_ptr->current_length);
                    break;
                } else {
                    status = 0;
                    // continue; // 不能对不在生成中（status=0等待）的样本进行阻塞，会影响正在生成的流式。只能返回结果给python，再在python内异步asleep。
                    break;
                }
            }
        } else {
            status = -1;
            break;
        }
    }

    if (status == 0) {
        resp = liteqwen::Response{0, 0, std::vector<int>(), std::vector<liteqwen::TopLogitsInfo>()};
        return resp;
    }
    else if (status == -1) {
        resp = liteqwen::Response{-1, 0, std::vector<int>(), std::vector<liteqwen::TopLogitsInfo>()};
        printf("POOL: yielding status -1 for %s, most likely due to timeout.\n", request_id.c_str());
        return resp;
    } else {
        if (force_interupt) { //强制终止生成，设置eos，下一次python尝试正常获取generate内容时就会正常eos。
            printf("POOL: forcing stop: %s\n", request_id.c_str());
            ctx_ptr->isEnding = true;
        }

        while (no_stream && (!(ctx_ptr->isEnding))) { // no_stream的话等待生成完毕
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        if (ctx_ptr->isEnding) { // 最后一帧返回完整内容
            int resp_len = ctx_ptr->current_length - ctx_ptr->input_length;
            std::vector<int> response_token_list((ctx_ptr->tokens).begin() + ctx_ptr->input_length, (ctx_ptr->tokens).end());
            std::vector<liteqwen::TopLogitsInfo> response_token_logits;
            if (ctx_ptr->return_logits) {
                int delta_len = ctx_ptr->current_length - ctx_ptr->prev_length;
                int delta_lgt_ct = (ctx_ptr->generation_config.top_k) * delta_len;
                if (delta_lgt_ct > 0) {
                    int total_lgt_ct = ctx_ptr->token_logits.size();
                    // 增量返回最后一桢的logits
                    for (int lgt_idx = total_lgt_ct-delta_lgt_ct-1; lgt_idx<total_lgt_ct; lgt_idx++) {
                        response_token_logits.push_back((ctx_ptr->token_logits)[lgt_idx]);
                    }
                    printf("POOL: returning last frame %i logits\n", total_lgt_ct);
                }
            }
            resp = liteqwen::Response{2, resp_len, response_token_list, response_token_logits};
            printf("POOL: eos hit for request_id=%s, resp_token_len=%i, deleting from pool.\n", request_id.c_str(), resp_len);
            liteqwen::thread_pool->DELETE(request_id);
            return resp;
        }

        int delta_len = ctx_ptr->current_length - ctx_ptr->prev_length;
        while (delta_len <= 0) { // 确保每次yield内容非空
            // printf("waiting for new content: %s, prev_len=%i, cur_len=%i\n", request_id.c_str(), ctx_ptr->prev_length, ctx_ptr->current_length);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (ctx_ptr->isEnding) {
                break;
            }
            delta_len = ctx_ptr->current_length - ctx_ptr->prev_length;
        }

        int prev_len = ctx_ptr->prev_length;
        int cur_len = ctx_ptr->current_length;
        int inp_len = ctx_ptr->input_length;
        int resp_len = cur_len - inp_len;
        int top_k = ctx_ptr->generation_config.top_k;
        std::vector<int> response_token_list;
        if (is_incremental) {
            response_token_list = std::vector<int>((ctx_ptr->tokens).begin() + prev_len, (ctx_ptr->tokens).begin() + cur_len);
        } else {
            response_token_list = std::vector<int>((ctx_ptr->tokens).begin() + inp_len, (ctx_ptr->tokens).begin() + cur_len);
        }

        std::vector<liteqwen::TopLogitsInfo> response_token_logits;
        if (ctx_ptr->return_logits) {
            int lgt_start_offset = (prev_len - inp_len) * top_k;
            int lgt_end_offset = (cur_len - inp_len) * top_k;
            if (lgt_end_offset <= ctx_ptr->token_logits.size()) {
                response_token_logits = std::vector<liteqwen::TopLogitsInfo>((ctx_ptr->token_logits).begin()+lgt_start_offset, (ctx_ptr->token_logits).begin()+lgt_end_offset);
                // printf("return with incremental logits[%i:%i], top_k=%i\n", lgt_start_offset, lgt_end_offset, top_k);
            } else {
                printf("something wrone with logit offset calculation, total_size=%i, start_offset=%i, end_offset=%i, top_k=%i\n", ctx_ptr->token_logits.size(), prev_len - ctx_ptr->input_length, cur_len - ctx_ptr->input_length, ctx_ptr->generation_config.top_k);
                response_token_logits = std::vector<liteqwen::TopLogitsInfo>();
            }
        } else {
            response_token_logits = std::vector<liteqwen::TopLogitsInfo>();
        }


        resp = liteqwen::Response{1, resp_len, response_token_list, response_token_logits};
        ctx_ptr->SetPrevLen(cur_len);
        return resp;
    }

    return resp;
}
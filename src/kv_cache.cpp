#include "kv_cache.h"
#include "core_gpu.cuh"

namespace liteqwen {


KVBlock::KVBlock(void* global_data, size_t glob_layer_stride, size_t position_start, int max_length, int num_layers, int cache_channel) {
    this->glob_data = glob_data;
    this->glob_layer_stride = glob_layer_stride;
    this->num_layers = num_layers;
    this->position_start = position_start;
    this->cache_channel = cache_channel;
    this->max_length = max_length;
    this->numel = static_cast<size_t>(max_length) * cache_channel;
    // printf("<===adding block for each layer [%lu, %lu), numel=%lu\n", this->position_start, this->position_start+this->numel, this->numel);
}

KVBlock::~KVBlock() {
    // printf("~removing block\n");
}

size_t KVBlock::get_layer_shift(int layer_id, bool is_value) {
    // 跳转到指定laye扇区，之后前进到KVBlock对应样本的step=0对应的bl位置。
    int line_id = static_cast<int>(is_value) * this->num_layers + layer_id;
    return this->glob_layer_stride * line_id + this->position_start;
}

KVPool::KVPool(int gpu_id, int max_dynamic_bsz, int max_dynamic_length, int start_layer_id, int num_layers, int cache_channel) {
    this->pool_numel = static_cast<size_t>(max_dynamic_length) * cache_channel * num_layers * 2;
    this->position_numel = static_cast<size_t>(max_dynamic_length) * cache_channel;
    this->cache_channel = cache_channel;
    this->start_layer_id = start_layer_id;
    this->num_layers = num_layers; // local_layer_num
    this->gpu_id = gpu_id;
    this->max_dynamic_bsz = max_dynamic_bsz;

    this->cudaData_ = GetDtypeCudaMalloc(gpu_id, liteqwen::DataType::FLOAT16, this->pool_numel, false);
    this->global_layer_stride = static_cast<size_t>(max_dynamic_length) * cache_channel;
    this->layerData = new void*[num_layers * 2]; // k & v start pointer for each layer

    this->example_numel_shifts = new size_t[max_dynamic_bsz];

    DeviceSynchronize();
    this->example_numel_shifts_gpu.Init(DataType::INT64, std::vector<int>{max_dynamic_bsz}, gpu_id, false);
    example_numel_shifts_gpu.Allocate();

    int ptr_data_len = sizeof(void*) * 2 * num_layers / sizeof(uint8_t); // ptrs = [&layer1_key, &layer1_value, &layer2_key, &layer2_value, ...]
    this->gpu_layer_pointer.Init(DataType::INT8, std::vector<int>{ptr_data_len}, gpu_id, false);
    this->gpu_layer_pointer.Allocate(static_cast<size_t>(ptr_data_len+1));
    DeviceSynchronize();
    MoveToLayerStarts(this->layerData, this->gpu_layer_pointer, this->cudaData_, this->global_layer_stride, num_layers);
    printf("stage with layers[%i, %i) initialized the starting pointers for each layer.\n", this->start_layer_id, this->start_layer_id+num_layers);
    DeviceSynchronize();
    this->example_ptr_stride = ptr_data_len;

    int one_batch_ptr_len = sizeof(void*) * 2/sizeof(uint8_t);
    this->batch_ptrs.Init(DataType::INT8, std::vector<int>{num_layers, max_dynamic_bsz, one_batch_ptr_len}, gpu_id, false);
    this->batch_ptrs.Allocate();
    DeviceSynchronize();
}

void KVPool::local_scatter_example_ptrs(StringArray req_ids, bool is_first_stage, size_t* first_stage_example_shifts) {
    SetDevice(this->gpu_id);

    int dynamic_bsz = req_ids.size();
    // 样本在kv-cache中的固定位置，只在stage0的KVPool的cpu中被维护，所以其他stage复制stage0的example_numel_shifts信息。
    for (int bi=0; bi<dynamic_bsz; bi++) {
        if (is_first_stage) {
            auto block_it = this->cached_blocks.find(req_ids[bi]);
            if (block_it != this->cached_blocks.end()) {
                size_t example_cache_start_numel_shift = (block_it->second)->position_start; // pos * channel
                this->example_numel_shifts[bi] = example_cache_start_numel_shift;
            }
        } else {
            this->example_numel_shifts[bi] = first_stage_example_shifts[bi];
        }
    }
    // DeviceSynchronize();
    cudaMemcpy((size_t*)(this->example_numel_shifts_gpu.cudaData), this->example_numel_shifts, sizeof(size_t)*dynamic_bsz, cudaMemcpyHostToDevice);

    // shifting each layer starts to numel starts for each example in req_ids.
    // batch_layer_starts: [local_layer_num, max_B, 2 * uint8_size(void*)], dtype=uint8
    // example_numel_shifts_gpu: [max_dynamic_bsz] dtype=size_t
    // gpu_layer_start_ptrs: [local_layer_num * 2 * uint8_size(void*)], dtype=uint8
    ScatterLayerKVExamplePtrs(this->batch_ptrs, this->example_numel_shifts_gpu, this->gpu_layer_pointer, dynamic_bsz);
}

Data KVPool::local_get_layer_example_ptrs(int local_layer_id) {
    int single_layer_ptr_numel = sizeof(void*) * 2/sizeof(uint8_t) * this->max_dynamic_bsz;
    size_t layer_offset = static_cast<size_t>(single_layer_ptr_numel) * local_layer_id;
    Data res = Data(DataType::INT8, std::vector<int>{single_layer_ptr_numel}, this->gpu_id, this->batch_ptrs.cudaData, layer_offset);
    return res;
}

std::pair<bool, size_t> KVPool::search_block(std::string request_id, int max_length) {
    // 根据池内所有处理中的请求的cache block信息，搜索能容纳max_length的kv缓存地址的numel偏移（pos=[0, max_static_b*max_static_l) * cache_channel）
    // 只需要在pipeline parallel的stage 0搜索一次即可，因为所有stage的Pool的缓存位置偏移规则是一样的，只有local_layer扇区可能不同。
    // 使用链表 block_chain 查找cache block之间足够大的间隙，之后greedy方式返回第一个足够大的空间，左边界开始作为新cache block的起始位置。
    unsigned long numel = static_cast<unsigned long>(max_length) * this->cache_channel;

    int existing_block_ct = (int)(this->cached_blocks.size());
    auto block_chain = new CacheSpan[existing_block_ct+2];
    (block_chain + 0)->Start();
    (block_chain + existing_block_ct+1)->End(this->position_numel);
    (block_chain + 0)->set_next(block_chain + existing_block_ct+1);
    (block_chain + existing_block_ct+1)->set_prev(block_chain + 0);

    int existing_block_id = 1;
    for (auto block_iter=this->cached_blocks.begin(); block_iter != cached_blocks.end(); block_iter++) {
        unsigned long left = (block_iter->second) -> position_start;
        unsigned long right = left +  ((block_iter->second) -> numel);

        (block_chain+existing_block_id)->Init(left, right);
        // printf("search_block_seq inserting existing block: [%lu, %lu)\n", left, right);
        (block_chain+0)->insert(block_chain+existing_block_id);
        existing_block_id += 1;
    }

    auto start_block = block_chain[0];
    if (start_block.next->next == nullptr) {
        // printf("empty block chain, assign a new cache block at beginning\n");
        printf("KV_CACHING: successful allocate maxlen=%i, req_id=%s. Empty memory, numel=<<+%lu>>\n", max_length, request_id.c_str(), numel);
        delete block_chain;
        return std::pair<bool, size_t>(true, static_cast<size_t>(0));
    } else {
        auto cur_blk_ref = start_block.next;
        unsigned long limit_right = cur_blk_ref->start;
        unsigned long limit_left = (cur_blk_ref->prev)->end;
        std::stringstream memory_info;
        memory_info << "[" << (cur_blk_ref->prev)->start << ",";

        while (limit_right - limit_left < numel) {
            if (cur_blk_ref->next == nullptr) {
                break;
            } else {
                memory_info << limit_left << "]_[" << limit_right << ",";
            }
            // printf("cur=[%lu, %lu)->next[%lu, %lu)\n", cur_blk_ref->start, cur_blk_ref->end, cur_blk_ref->next->start, cur_blk_ref->next->end);
            cur_blk_ref = cur_blk_ref->next;
            limit_right = cur_blk_ref->start;
            limit_left = (cur_blk_ref->prev)->end;
            // printf("using updated search range[%i, %i)\n", limit_left, limit_right);
        }

        if (cur_blk_ref->next == nullptr && limit_right -limit_left < numel) {
            memory_info << limit_left << "]_[" << limit_right << "] no space >="<< numel;
            printf("KV_CACHING: failed allocate maxlen=%i, req_id=%s. MemoryInfo=%s\n", max_length, request_id.c_str(), memory_info.str().c_str());
            memory_info.clear();
            delete block_chain;
            return std::pair<bool, size_t>(false, static_cast<size_t>(0));
        } else {
            memory_info << limit_left << "]_<<+"<< numel << ">>_[" << limit_right << "]";
            printf("KV_CACHING: successful allocate maxlen=%i, req_id=%s. MemoryInfo=%s\n", max_length, request_id.c_str(), memory_info.str().c_str());
            memory_info.clear();
            delete block_chain;
            return std::pair<bool, size_t>(true, limit_left);
        }
    }

    delete block_chain;
    return std::pair<bool, size_t>(false, static_cast<size_t>(0));
};


AllocateParam KVPool::search_block_sequence(std::string request_id, int max_length, std::vector<AllocateParam>* pre_occupied) {
    // 根据池内所有处理中的请求的cache block信息，搜索能容纳max_length的kv缓存地址的numel偏移（pos=[0, max_static_b*max_static_l) * cache_channel）
    // 只需要在pipeline parallel的stage 0搜索一次即可，因为所有stage的Pool的缓存位置偏移规则是一样的，只有local_layer扇区可能不同。
    // 使用链表 block_chain 查找cache block之间足够大的间隙，之后greedy方式返回第一个足够大的空间，左边界开始作为新cache block的起始位置。
    unsigned long numel = static_cast<unsigned long>(max_length) * this->cache_channel;
    int pre_occupied_num = (int)(pre_occupied->size());

    int existing_block_ct = (int)(this->cached_blocks.size()) + pre_occupied_num;
    auto block_chain = new CacheSpan[existing_block_ct+2];
    // printf("block chain length = start+%i+end\n", existing_block_ct);
    (block_chain + 0)->Start();
    (block_chain + existing_block_ct+1)->End(this->position_numel);
    (block_chain + 0)->set_next(block_chain + existing_block_ct+1);
    (block_chain + existing_block_ct+1)->set_prev(block_chain + 0);

    // 真实已经分配的cache
    int existing_block_id = 1;
    if(existing_block_ct > 0) {
        for (auto block_iter=this->cached_blocks.begin(); block_iter != cached_blocks.end(); block_iter++) {
            unsigned long left = (block_iter->second) -> position_start;
            unsigned long right = left +  ((block_iter->second) -> numel);

            (block_chain+existing_block_id)->Init(left, right);
            // printf("search_block_seq inserting existing block: [%lu, %lu)\n", left, right);
            (block_chain+0)->insert(block_chain+existing_block_id);
            existing_block_id += 1;
        }
    }


    // 尚未真实分配，但在同一个batch中已经确定会被分配的样本。
    if (pre_occupied->size() > 0) {
        for (auto pre_occu_iter=pre_occupied->begin(); pre_occu_iter != pre_occupied->end(); pre_occu_iter++) {
            unsigned long left = pre_occu_iter -> bl_start;
            unsigned long right = pre_occu_iter -> bl_end; //left + static_cast<size_t>(this->cache_channel) * ((pre_occu_iter->second) -> max_length);  // left + numel

            (block_chain+existing_block_id)->Init(left, right);
            (block_chain+0)->insert(block_chain+existing_block_id);
            existing_block_id += 1;
        }
    }

    auto start_block = block_chain[0];
    if (start_block.next->next == nullptr) {
        printf("KV_CACHING: successful allocate maxlen=%i, req_id=%s. Empty memory, numel=<<+%lu>>\n", max_length, request_id.c_str(), numel);
        delete block_chain;
        return AllocateParam{request_id, true,  static_cast<size_t>(0),  static_cast<size_t>(numel)};
    } else {
        auto cur_blk_ref = start_block.next;
        unsigned long limit_right = cur_blk_ref->start;
        unsigned long limit_left = (cur_blk_ref->prev)->end;
        std::stringstream memory_info;
        memory_info << "[" << (cur_blk_ref->prev)->start << ",";

        while (limit_right - limit_left < numel) {
            if (cur_blk_ref->next == nullptr) {
                break;
            } else {
                memory_info << limit_left << "]_[" << limit_right << ",";
            }
            // printf("cur=[%lu, %lu)->next[%lu, %lu)\n", cur_blk_ref->start, cur_blk_ref->end, cur_blk_ref->next->start, cur_blk_ref->next->end);
            cur_blk_ref = cur_blk_ref->next;
            limit_right = cur_blk_ref->start;
            limit_left = (cur_blk_ref->prev)->end;
            // printf("using updated search range[%i, %i)\n", limit_left, limit_right);
        }

        if (cur_blk_ref->next == nullptr && limit_right -limit_left < numel) {
            // printf("no space available for %s, exiting without assigning block\n", request_id.c_str());
            memory_info << limit_left << "]_[" << limit_right << "] no space >="<< numel;
            printf("KV_CACHING: failed allocate maxlen=%i, req_id=%s. MemoryInfo=%s\n", max_length, request_id.c_str(), memory_info.str().c_str());
            memory_info.clear();
            delete block_chain;
            return AllocateParam{request_id, false,  static_cast<size_t>(0),  static_cast<size_t>(0)};
        } else {
            // printf("found space for [%lu, %lu) able to contain size %lu, cur_blk_ref->end=%lu\n", limit_left, limit_right, numel, cur_blk_ref->end);
            memory_info << limit_left << "]_<<+"<< numel << ">>_[" << limit_right << "]";
            printf("KV_CACHING: successful allocate maxlen=%i, req_id=%s. MemoryInfo=%s\n", max_length, request_id.c_str(), memory_info.str().c_str());
            memory_info.clear();
            delete block_chain;
            return AllocateParam{request_id, true,  limit_left,  limit_left+numel};
        }
    }

    printf("should not return here...\n");
    delete block_chain;
    return AllocateParam{request_id, false,  static_cast<size_t>(0),  static_cast<size_t>(0)};
};

void KVPool::allocate_blocks(std::string request_id, size_t bl_start, int max_length) {
    // 根据已经search到的cache内偏移，新建block信息，并与request_id绑定。KVBlock需要被手动释放（调用free_block在推理完成或超时后），无法自动清理，为了保证数据不被意外覆写。
    this->cached_blocks[request_id] = new KVBlock(this->cudaData_, this->global_layer_stride, bl_start, max_length, this->num_layers, this->cache_channel);
};

void KVPool::free_block(std::string request_id, bool should_print) {
    auto optional_req_info = this->cached_blocks.find(request_id);
    if (optional_req_info != this->cached_blocks.end()) {
        delete optional_req_info->second;
        this->cached_blocks.erase(request_id);
        if (should_print) printf("KV_CACHING: free cache block=%s, remaining block_ct=%i\n", request_id.c_str(), (int)(this->cached_blocks.size()));
    } 
    else {
        if (should_print) printf("KV_CACHING: free not found for block=%s, remaining block_ct=%i\n", request_id.c_str(), (int)(this->cached_blocks.size()));
    }
}

int KVPool::get_count() {
    return static_cast<int>(this->cached_blocks.size());
}


void KVPool::write_example_kvs_to_local_cache(bool is_prefill, int dynamic_bsz, int local_layer_id, std::pair<Data*, Data*> kv_pair, const Data& act_kstarts, const Data& batch_ids, int max_dynamic_bsz, int dynamic_bl, int kv_heads, int channels) {
    int dynamic_len;
    if (is_prefill) {
        dynamic_len = dynamic_bl;
    } else {
        dynamic_len = dynamic_bsz;
    }
    WriteKVCaches(is_prefill, local_layer_id, this->batch_ptrs, kv_pair, act_kstarts, batch_ids, dynamic_len, kv_heads, this->max_dynamic_bsz);
}

void KVPool::print_cache(std::string request_id, int local_layer_id, bool is_value, int end_step) {
    size_t layer_shift;
    if (!is_value) {
        layer_shift = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, false);
    } else {
        layer_shift = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, true);
    }

    PrintWithShift(this->cudaData_, layer_shift, end_step, this->cache_channel);
}

PipelineKVPool::PipelineKVPool(int max_dynamic_bsz, int max_dynamic_length, int cache_channel, std::vector<int> layer_id_map) {
    this->num_layers = (int)(layer_id_map.size());
    this->cache_channel = cache_channel;
    int start_rank = -1;
    int end_rank = -1;
    int peek_rank = layer_id_map[0];
    int stage_id = 0;
    int stage_start_rank = -1;
    int stage_start_layer = -1;
    this->max_dynamic_bsz = max_dynamic_bsz;
    
    for (int i=0; i< this->num_layers; i++) {
        int rank = layer_id_map[i];
        this->layer2stage_map.push_back(stage_id);

        if (i==0) {
            start_rank = rank;
        }
        if (i==this->num_layers - 1) {
            end_rank = rank;
            peek_rank = -1;
        } else {
            peek_rank = layer_id_map[i+1];
        }
        if (rank != stage_start_rank) {
            stage_start_rank = rank;
            stage_start_layer = i;
        }
        int local_layer_num = i - stage_start_layer + 1;
        
        if (peek_rank != rank) {
            this->pipeline_caches[stage_id] = new KVPool(rank, max_dynamic_bsz, max_dynamic_length, stage_start_rank, local_layer_num, cache_channel);
            this->local_stage_starts[stage_id] = stage_start_layer;
            stage_id += 1;
        }

    }
    this->pipeline_parallel_size = end_rank - start_rank + 1;
};

bool PipelineKVPool::allocate_cache(std::string request_id, int max_length) {
    size_t bl_start = 0;
    for (int stage_id=0; stage_id<(int)(this->pipeline_caches.size()); stage_id++) {
        if (stage_id == 0) {
            std::pair<bool, size_t> block_info = (this->pipeline_caches)[stage_id]->search_block(request_id, max_length);
            if (!block_info.first) {
                printf("no available cache found for %s, len=%i, waiting for block releasing...\n", request_id.c_str(), max_length);
                return false;
            }
            bl_start = block_info.second;
            printf("allocated cache blocks for req_id=%s, len=%i, start at %lu.\n", request_id.c_str(), max_length, bl_start);
        }
        (this->pipeline_caches)[stage_id]->allocate_blocks(request_id, bl_start, max_length);
    }
    return true;
}

bool PipelineKVPool::sequence_allocate_cache(std::vector<AllocateParam> verified_allocates) {
    for (int stage_id=0; stage_id<(int)(this->pipeline_caches.size()); stage_id++) { 
        for (int i=0; i<verified_allocates.size(); i++) {
            AllocateParam cand = verified_allocates[i];
            int example_maxlen = cand.get_block_len(this->cache_channel);
            (this->pipeline_caches)[stage_id]->allocate_blocks(cand.request_id, cand.bl_start, example_maxlen);
            // printf("allocated a kv cache which is searched: req=%s, len=%i: [%lu, %lu)\n", cand.request_id.c_str(), example_maxlen, cand.bl_start, cand.bl_end);
        }
    }
    return true;
}

void PipelineKVPool::free(std::string request_id) {
    for (int stage_id=0; stage_id<(int)(this->pipeline_caches.size()); stage_id++) {
        bool should_print = stage_id == 0;
        (this->pipeline_caches)[stage_id]->free_block(request_id, should_print);
    }
}

int PipelineKVPool::get_caching_count() {
    return (this->pipeline_caches)[0]->get_count();
}

void PipelineKVPool::write_example_kvs_to_cache(bool is_prefill, int dynamic_bsz, int layer_id, std::pair<Data*, Data*> kv_pair, const Data& act_kstarts, const Data& batch_ids, int max_dynamic_bsz, int dynamic_bl, int kv_heads, int channels) {
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    (this->pipeline_caches)[stage_id]->write_example_kvs_to_local_cache(is_prefill, dynamic_bsz, local_layer_id, kv_pair, act_kstarts, batch_ids, max_dynamic_bsz, dynamic_bl, kv_heads, channels);
}

void PipelineKVPool::print_cache(std::string request_id, int layer_id, bool is_value, int end_step) {
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    (this->pipeline_caches)[stage_id]->print_cache(request_id, local_layer_id, is_value, end_step);
}

void PipelineKVPool::scatter_example_ptrs(StringArray req_ids, int layer_id) {
    int stage_id = this->layer2stage_map[layer_id];
    // int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    size_t* first_stage_shifts_cpu = (this->pipeline_caches)[0]->example_numel_shifts;
    if (stage_id == 0) {
        (this->pipeline_caches)[stage_id]->local_scatter_example_ptrs(req_ids, true, first_stage_shifts_cpu);
    } else {
        (this->pipeline_caches)[stage_id]->local_scatter_example_ptrs(req_ids, false, first_stage_shifts_cpu);
    }
}

Data PipelineKVPool::get_layer_example_ptrs(int layer_id) {
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    return (this->pipeline_caches)[stage_id]->local_get_layer_example_ptrs(local_layer_id);
}

} //namespace liteqwen
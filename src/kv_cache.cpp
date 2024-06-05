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

    this->cudaData_ = GetDtypeCudaMalloc(gpu_id, liteqwen::DataType::FLOAT16, this->pool_numel, false);
    this->global_layer_stride = static_cast<size_t>(max_dynamic_length) * cache_channel;
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

void KVPool::write_local_layer_kv(std::string request_id, int local_layer_id, std::pair<Data*, Data*> kv_pair, int cache_start_step, int act_position, const Data& pos_starts, int bi, int example_len) {
    // 需要已经成功allocate过缓存，且 cache_start_step + step_num < block的max_length，才能保证新写入的数据不覆盖其他样本。
    // layer_shift_key首先考虑了【layer位移+cache allocate时的动态block位移】，之后write_offset_key再考虑绝对step下写入的位移。
    // 输出act_position则对应了activation的BL位置，不同样本的BL*channel数据应当被紧密排列。
    if (this->cached_blocks.find(request_id) != this->cached_blocks.end()) {
        int block_maxlen = this->cached_blocks[request_id]->max_length;

        size_t layer_shift_key = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, false);
        size_t layer_shift_val = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, true);

        void* pairKeyData = kv_pair.first->cudaData;
        void* pairValueData = kv_pair.second->cudaData;
        int gpu_id = kv_pair.first->gpu_id;

        if (act_position<0) { // cache_start_step=0
            size_t write_offset_key = layer_shift_key + cache_start_step * this->cache_channel;
            size_t write_offset_val = layer_shift_val + cache_start_step * this->cache_channel;
            WriteKVCacheFromBatch(true, this->cudaData_, pairKeyData, gpu_id, write_offset_key, pos_starts, bi, this->cache_channel, example_len);
            WriteKVCacheFromBatch(true, this->cudaData_, pairValueData, gpu_id, write_offset_val, pos_starts, bi, this->cache_channel, example_len);
        } else {
            // cache_start_step = -1
            size_t write_offset_key = layer_shift_key;
            size_t write_offset_val = layer_shift_val;            
            WriteKVCacheFromBatch(false, this->cudaData_, pairKeyData, gpu_id, write_offset_key, pos_starts, bi, this->cache_channel, example_len);
            WriteKVCacheFromBatch(false, this->cudaData_, pairValueData, gpu_id, write_offset_val, pos_starts, bi, this->cache_channel, example_len);
        }
    } else {
        printf("ERROR: KV cache for req=%s not found, please check whether the cache has been allocated.\n", request_id.c_str());
    }
}

void KVPool::write_local_layer_kv(std::string request_id, int local_layer_id, std::pair<Data*, Data*> kv_pair, int cache_start_step, int act_position, int step_num) {
    // 需要已经成功allocate过缓存，且 cache_start_step + step_num < block的max_length，才能保证新写入的数据不覆盖其他样本。
    // layer_shift_key首先考虑了【layer位移+cache allocate时的动态block位移】，之后write_offset_key再考虑绝对step下写入的位移。
    // 输出act_position则对应了activation的BL位置，不同样本的BL*channel数据应当被紧密排列。
    if (this->cached_blocks.find(request_id) != this->cached_blocks.end()) {
        int block_maxlen = this->cached_blocks[request_id]->max_length;

        size_t layer_shift_key = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, false);
        size_t layer_shift_val = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, true);

        size_t write_offset_key = layer_shift_key + cache_start_step * this->cache_channel;
        size_t write_offset_val = layer_shift_val + cache_start_step * this->cache_channel;

        void* pairKeyData = kv_pair.first->cudaData;
        void* pairValueData = kv_pair.second->cudaData;
        int gpu_id = kv_pair.first->gpu_id;

        printf("write to local layer=%i, layer_kshift=%lu+position_shift=%lu+step_shift=%lu\n", local_layer_id, layer_shift_key-this->cached_blocks[request_id]->position_start, this->cached_blocks[request_id]->position_start, cache_start_step * this->cache_channel);

        size_t read_offset = static_cast<size_t>(act_position) * this->cache_channel;
        size_t copy_length = static_cast<size_t>(step_num) * this->cache_channel;

        CopyGPUData(DataType::FLOAT16, this->cudaData_, pairKeyData, gpu_id, write_offset_key, read_offset, copy_length, false);
        CopyGPUData(DataType::FLOAT16, this->cudaData_, pairValueData, gpu_id, write_offset_val, read_offset, copy_length, false);
    } else {
        printf("ERROR: KV cache for req=%s not found, please check whether the cache has been allocated.\n", request_id.c_str());
    }
}

void KVPool::read_local_layer_kv(std::string request_id, int local_layer_id, std::pair<Data*, Data*> kv_outs, int act_position, int cache_start_step, int step_num) {
    if (this->cached_blocks.find(request_id) != this->cached_blocks.end()) {
        int block_maxlen = this->cached_blocks[request_id]->max_length;
        if (step_num + cache_start_step > block_maxlen) {
            printf("ERROR: skipping illegal cache read for req=%s. Cannot read step=shift(%i)+len(%i) > maxlen(%i), please allocate a larger buffer, or early stop the generation.\n", request_id.c_str(), cache_start_step, step_num, block_maxlen);
            return;
        }

        size_t layer_shift_key = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, false);
        size_t layer_shift_val = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, true);

        size_t write_offset =  static_cast<size_t>(act_position) * this->cache_channel;
        size_t read_offset_key = (layer_shift_key +  cache_start_step * this->cache_channel);
        size_t read_offset_val = (layer_shift_val +  cache_start_step * this->cache_channel);
        size_t copy_length = static_cast<size_t>(step_num) * this->cache_channel;

        void* outKeyData = kv_outs.first->cudaData;
        void* outValueData = kv_outs.second->cudaData;

        CopyGPUData(DataType::FLOAT16, outKeyData, this->cudaData_, gpu_id, write_offset, read_offset_key, copy_length, false);
        CopyGPUData(DataType::FLOAT16, outValueData, this->cudaData_, gpu_id, write_offset, read_offset_val, copy_length, false);
    } else {
        printf("ERROR: KV cache for req=%s not found, please check whether the cache has been allocated.\n", request_id.c_str());
    }
}

void KVPool::write_local_batch_kv(bool is_prefill, StringArray request_ids, int local_layer_id, std::pair<Data*, Data*> kv_pair, const Data& gpu_offsets, const Data& kstarts, const Data& batch_ids, int max_dynamic_bsz, int dynamic_bl, int kv_heads, int channels) {
    int dynamic_bsz = (int)(request_ids.size());
    int dynamic_len;
    if (is_prefill) {
        dynamic_len = dynamic_bl;
    } else {
        dynamic_len = dynamic_bsz;
    }

    if (this->cache_channel != kv_heads * channels) {
        printf("ERROR: Writing kv activation dim=%ix%i, but cache size=%i. Make sure these are equal.\n", kv_heads, channels, this->cache_channel);
        throw("");
    }
    if (channels != 128) {
        printf("ERROR: only kv channel 128 supported, if other channels are used, make sure WriteGPUKV implements copying kernel with other dim.\n");
        throw("");
    }

    size_t* shifts = (size_t*)gpu_offsets.cudaData;
    // size_t each_example_start_offset = static_cast<size_t>(read_start_step) * this->cache_channel;
    size_t* shifts_cpu = new size_t[2*max_dynamic_bsz];
    for (int bi=0; bi<dynamic_bsz; bi++) {
        std::string request_id = request_ids[bi];
        if (this->cached_blocks.find(request_id) != this->cached_blocks.end()) {
            size_t layer_shift_key = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, false);
            size_t layer_shift_val = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, true);

            size_t cache_start = this->cached_blocks[request_id]->position_start;
            // printf("found local layer=%i, req=%s, layer_kshift=%lu, layer_vshift=%lu, cache_start_pos=%lu\n", local_layer_id, request_id.c_str(), layer_shift_key, layer_shift_val, cache_start);

            shifts_cpu[bi*2] = layer_shift_key; // + each_example_start_offset;
            shifts_cpu[bi*2+1] = layer_shift_val; // + each_example_start_offset;
        } else {
            printf("ERROR: KV cache for req=%s not found, please check whether the cache has been allocated.\n", request_id.c_str());
        }        
    }
    cudaMemcpy(shifts, shifts_cpu, sizeof(size_t)*2*max_dynamic_bsz, cudaMemcpyHostToDevice);
    WriteGPUKV(is_prefill, this->cudaData_, gpu_offsets, kv_pair, kstarts, batch_ids, dynamic_len, kv_heads);
    delete shifts_cpu;
}

void KVPool::read_local_batch_kv_ref(StringArray request_ids, int local_layer_id, const Data& kv_ptrs, const Data& gpu_offsets, int max_dynamic_bsz, int read_start_step, bool shift_loaded) {
    int dynamic_bsz = (int)(request_ids.size());

    if (!shift_loaded) {
        size_t* shifts = (size_t*)gpu_offsets.cudaData;
        size_t* shifts_cpu;
        // if gpu_offsets already loaded during kv-cache write, ignore this step in attention reading step.
        size_t copy_size = sizeof(void*) * 2 * max_dynamic_bsz;
        size_t each_example_start_offset = static_cast<size_t>(read_start_step) * this->cache_channel;
        size_t void_half_ratio = sizeof(__half) / sizeof(void);

        shifts_cpu = new size_t[2*max_dynamic_bsz];
        for (int bi=0; bi<dynamic_bsz; bi++) {
            std::string request_id = request_ids[bi];
            if (this->cached_blocks.find(request_id) != this->cached_blocks.end()) {
                size_t layer_shift_key = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, false);
                size_t layer_shift_val = this->cached_blocks[request_id]->get_layer_shift(local_layer_id, true);

                size_t cache_start = this->cached_blocks[request_id]->position_start;
                // printf("found local layer=%i, req=%s, layer_kshift=%lu, layer_vshift=%lu, cache_start_pos=%lu\n", local_layer_id, request_id.c_str(), layer_shift_key, layer_shift_val, cache_start);

                shifts_cpu[bi*2] = layer_shift_key + each_example_start_offset;
                shifts_cpu[bi*2+1] = layer_shift_val + each_example_start_offset;
            } else {
                printf("ERROR: KV cache for req=%s not found, please check whether the cache has been allocated.\n", request_id.c_str());
            }
        }
        cudaMemcpy(shifts, shifts_cpu, sizeof(size_t)*2*max_dynamic_bsz, cudaMemcpyHostToDevice);
        delete shifts_cpu;
    }

    MoveGPUKVPtrs(kv_ptrs, this->cudaData_, gpu_offsets, dynamic_bsz);
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

void PipelineKVPool:: write_layer_kv(std::string request_id, int layer_id, std::pair<Data*, Data*> kv_pair, int cache_start_step, int act_position, int step_num) {
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    (this->pipeline_caches)[stage_id]->write_local_layer_kv(request_id, local_layer_id, kv_pair, cache_start_step, act_position, step_num);
}

void PipelineKVPool:: write_layer_kv(std::string request_id, int layer_id, std::pair<Data*, Data*> kv_pair, int cache_start_step, int act_position, const Data& pos_starts, int bi, int step_num) {
    // if prefill, cache_start_step=0 && act_position=-1; if decode, cache_start_step=-1 && act_position=bi
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    (this->pipeline_caches)[stage_id]->write_local_layer_kv(request_id, local_layer_id, kv_pair, cache_start_step, act_position, pos_starts, bi, step_num);
}

void PipelineKVPool:: read_layer_kv(std::string request_id, int layer_id, std::pair<Data*, Data*> kv_outs, int write_start_step, int read_start_step, int step_num) {
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    (this->pipeline_caches)[stage_id]->read_local_layer_kv(request_id, local_layer_id, kv_outs, write_start_step, read_start_step, step_num);
}

void PipelineKVPool::read_batch_kv_ref(StringArray request_ids, int layer_id, const Data& kv_ptrs, const Data& gpu_offsets, int max_dynamic_bsz, int read_start_step, bool shift_loaded) {
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    (this->pipeline_caches)[stage_id]->read_local_batch_kv_ref(request_ids, local_layer_id, kv_ptrs, gpu_offsets, max_dynamic_bsz, read_start_step, shift_loaded);
}

void PipelineKVPool::write_batch_layer_kv(bool is_prefill, StringArray request_ids, int layer_id, std::pair<Data*, Data*> kv_pair, const Data& gpu_offsets, const Data& kstarts, const Data& batch_ids, int max_dynamic_bsz, int dynamic_bl, int kv_heads, int channels) {
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    (this->pipeline_caches)[stage_id]->write_local_batch_kv(is_prefill, request_ids, local_layer_id, kv_pair, gpu_offsets, kstarts, batch_ids, max_dynamic_bsz, dynamic_bl, kv_heads, channels);
}

void PipelineKVPool::print_cache(std::string request_id, int layer_id, bool is_value, int end_step) {
    int stage_id = this->layer2stage_map[layer_id];
    int local_layer_id = layer_id - this->local_stage_starts[stage_id];
    (this->pipeline_caches)[stage_id]->print_cache(request_id, local_layer_id, is_value, end_step);
}

} //namespace liteqwen
#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <string>
#include <sstream>
#include <map>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "core_cpu.h"
#include "entities.h"

namespace liteqwen {

const int fp16_bytes=2;

struct CacheSpan {
    unsigned long start;
    unsigned long end;
    CacheSpan* prev;
    CacheSpan* next;

    void Init(unsigned long a, unsigned long b) {
        this->start = a;
        this->end = b;
        this->prev = nullptr;
        this->next = nullptr;
    }

    void Start(){
        this->start = 0;
        this->end = 0;
        this->prev = nullptr;
        this->next = nullptr;
    }
    void End(unsigned long end){
        this->start = end;
        this->end = end + 1;
        this->prev = nullptr;
        this->next = nullptr;
    }

    void set_prev(CacheSpan* span) {
        this->prev = span;
    }
    void set_next(CacheSpan* span) {
        this->next = span;
    }

    void insert(CacheSpan* span) {
        if (this->next != nullptr && this->next->start >= this->end) {
            if (this->prev == nullptr && this->next->next==nullptr) {
                span->set_next(this->next);
                span->set_prev(this);
                (this->next)->set_prev(span);
                this->set_next(span);
            } else {
                // printf("non first block, this->start(%lu) >= span->end(%lu)?\n", this->start, span->end);
                if (this->start >= span->end) {
                    // printf("found space before[%lu, %lu) for span=[%lu, %lu)\n", this->start, this->end, span->start, span->end);
                    span->set_next(this);
                    span->set_prev(this->prev);
                    (this->prev)->set_next(span);
                    this->set_prev(span);
                } else {
                    // printf("skipping a block [%i, %i)\n", this->start, this->end);
                    (this->next)->insert(span);
                }
            }
        } else if (this->next == nullptr) {
            if (this->start >= span->end) {
                // printf("end block hit, this->start=%lu, span->end=%lu\n", this->start, span->end);
                span->set_next(this);
                span->set_prev(this->prev);
                (this->prev)->set_next(span);
                (this)->set_prev(span);
            } else {
                printf("block allocated exceeding right barrier of kv_cache, check for error.\n");
            }
        }
        else {
            printf("this=[%lu, %lu), next=[%lu, %lu)], span=[%lu, %lu)]]\n", this->start, this->end, this->next->start, this->next->end, span->start, span->end);
            printf("error: spans not aranged, overlap may occure due to kv cache asignment error\n");
            return;
        }
    }
};

// 每条样本在对应GPU stage下的KV，shape=[2, M, L, H * D_kv], M是stage下的layer数，L是样本指定的最大长度。
class KVBlock {
    // storing a single request's caching data for layers mapped to the current gpu.
    public:
    void* glob_data;
    size_t glob_layer_stride;
    int num_layers;
    unsigned long position_start;
    int cache_channel;
    int max_length;
    size_t numel;
    KVBlock(){};
    KVBlock(void* glob_data, size_t glob_layer_stride, size_t position_start, int max_length, int num_layers, int cache_channel);
    ~KVBlock();

    size_t get_layer_shift(int layer_id, bool is_value);
};

// 每个stage的KV
class KVPool {
    public:
    int gpu_id;
    int start_layer_id;
    int num_layers; // pipeline layers, not global layers
    int cache_channel;
    int max_dynamic_bsz;
    size_t pool_numel;
    size_t position_numel;
    size_t global_layer_stride;
    KVPool(){};
    KVPool(int gpu_id, int max_static_batch, int max_static_length, int start_layer_id, int num_layers, int cache_channel);
    
    std::pair<bool, size_t> search_block(std::string request_id, int max_length);
    AllocateParam search_block_sequence(std::string request_id, int max_length, std::vector<AllocateParam>* pre_occupied);
    void allocate_blocks(std::string request_id, size_t bl_start, int max_length);
    void free_block(std::string request_id, bool should_print);
    int get_count();
    
    void print_cache(std::string request_id, int local_layer_id, bool is_value, int end_step);
    void write_example_kvs_to_local_cache(bool is_prefill, int dynamic_bsz, int layer_id, std::pair<Data*, Data*> kv_pair, const Data& act_kstarts, const Data& batch_ids, int max_dynamic_bsz, int dynamic_bl, int kv_heads, int channels);

    size_t* example_numel_shifts;
    Data example_numel_shifts_gpu;
    void local_scatter_example_ptrs(StringArray req_ids, bool is_first_stage, size_t* first_stage_example_shifts);
    Data local_get_layer_example_ptrs(int local_layer_id);
    private:
    void* cudaData_;
    void** layerData; // [local_layer_num * 2]
    int example_ptr_stride;
    Data batch_ptrs;
    Data gpu_layer_pointer;
    std::map<std::string, KVBlock*> cached_blocks;
};

// 每条推理样本的KV
class PipelineKVPool {
    public:
    int pipeline_parallel_size;
    int num_layers;
    int cache_channel;
    int max_dynamic_bsz;
    std::vector<int> layer_id_map;
    PipelineKVPool(int max_dynamic_bsz, int max_dynamic_length, int cache_channel, std::vector<int> layer_id_map);
    bool allocate_cache(std::string request_id, int max_length);
    bool sequence_allocate_cache(std::vector<AllocateParam> verified_allocates);
    void free(std::string request_id);
    int get_caching_count();

    void print_cache(std::string request_id, int layer_id, bool is_value, int end_step);
    void write_example_kvs_to_cache(bool is_prefill, int dynamic_bsz, int layer_id, std::pair<Data*, Data*> kv_pair, const Data& act_kstarts, const Data& batch_ids, int max_dynamic_bsz, int dynamic_bl, int kv_heads, int channels);

    // private:
    std::map<int, KVPool*> pipeline_caches;
    std::map<int, int> local_stage_starts;
    std::vector<int> layer2stage_map;

    void scatter_example_ptrs(StringArray req_ids, int layer_id);
    Data get_layer_example_ptrs(int layer_id);
};

} // namespace liteqwen

#endif // KV_CACHE_H
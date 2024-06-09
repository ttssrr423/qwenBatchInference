#ifndef _POOL_H
#define _POOL_H

#include <vector>
#include <cstdint>
#include <string>
#include <sstream>
#include <queue>
#include <list>
#include <unordered_map>
#include <functional>
#include <thread>
#include <future>
#include <chrono>
#include <mutex>
#include "entities.h"
#include "kv_cache.h"

const int uint8_maxval = 256;

namespace liteqwen {

struct DynamicExample {
    std::string request_id;
    int new_token;
    int start_position;
    int current_length;
    int max_length;
    int batch_idx;
    int seed;
    float temperature;
    float top_p;
    bool return_lgt;

    void Init(std::string request_id, int start_position, int current_length, int max_length, int batch_idx, int seed, float temperature, float top_p, bool return_lgt) {
        this->request_id = request_id;
        this->new_token = -1;
        this->start_position = start_position;
        this->current_length = current_length;
        this->max_length = max_length;
        this->batch_idx = batch_idx;
        this->seed = seed;
        this->temperature = temperature;
        this->top_p = top_p;
        this->return_lgt = return_lgt;
    }

    void next(int new_token, int dynamic_start, int new_batch_idx) {
        this->new_token = new_token;
        this->current_length += 1;
        this->start_position = dynamic_start;
        this->batch_idx = new_batch_idx;
    }
};

// 所有数据线程共享一个ContextPool
class ContextPool {

    private:
    using Timestamp = std::chrono::time_point<std::chrono::system_clock>;
    
    struct Node
    {
        std::string key;
        ResponseContext value;
        Timestamp timestamp;
        bool time_updating;

        void SetTimestamp(Timestamp new_ts) {
            this->timestamp = new_ts;
            this->time_updating = true;
        }

        void FinishingSetTimestamp() {
            this->time_updating = false;
        }

        ResponseContext* GetValuePtr() {
            return &(this->value);
        }
    };


    const int minimum_reload_interval = 20;

    public:
    // std::map<std::string, int> generating_ids; // request_id -> handle_id
    using NodePtr = std::shared_ptr<Node>;
    using NodeIter = typename std::list<NodePtr>::iterator;
    using ExpiredCallBack = std::function<void(std::string, ResponseContext)>;
    int max_queue_size;
    int default_maxlen;

    ContextPool(int max_queue_size, int timeout);
    ~ContextPool(){};
    int timeout;
    void Add(std::string key, ResponseContext value);
    ResponseContext* GetRes(std::string key, bool refresh); // get by key and refresh timeout
    ResponseContext* GetRes(std::string key);
    ResponseContext* GetPtr(std::string key, bool refresh);
    ResponseContext* GetPtr(std::string key);
    std::vector<AllocateParam> Reload(int data_id, std::string preparer_lora_name, bool preparer_is_empty, PipelineKVPool* kv_cache_ref);
    void SetDefaultMaxlen(int max_sequence_length);
    void DELETE(std::string key);
    void UnsafeDelete(std::string key);
    int GetLength();
    std::list<std::string> deleting_keys;
    // std::mutex request_locker;
    // std::thread *mainLoop = nullptr;
    void SetReloadOn(int data_id);
    void SetReloadOff(int data_id);
    bool CanReload(int data_id);
    void ReloadIntervalCountdown(int data_id, int new_ct);

    private:
    std::map<int, int> reload_switch;

    void Expired();
    std::list<NodePtr> list_;
    std::unordered_map<std::string, NodeIter> map_;
    std::list<NodePtr> res_list_;
    std::unordered_map<std::string, NodeIter> finished_;
    size_t max_size_;
    uint32_t time_out_;
    ExpiredCallBack expired_callback_;

};


// 每个数据线程独立一个BatchInputPreparer
class BatchInputPreparer {
    public:
    std::list<DynamicExample> prefill_examples;
    std::list<DynamicExample> decoding_examples;
    std::string batch_lora_name;
    bool is_decoding;
    bool all_eos; // BatchUpdate时更新，任意样本添加正常token时为false，所有样本eos或无token时为false，控制Reload的分桶时机。
    int max_batch_size;
    int max_seq_length; // 默认单个最大样本长度，并非动态block的最大长度。只在generation_config里没有给max_new_tokens生效。

    int prefill_bsz;
    int prefill_batch_maxlen;
    // std::vector<std::string> prefill_req_ids;
    StringArray prefill_req_ids;
    // std::vector<int> prefill_inp_ids;
    // std::vector<uint8_t> prefill_bids;
    // std::vector<int> prefill_starts;
    // std::vector<float> prefill_top_ps;
    // std::vector<float> prefill_temperatures;
    // std::vector<int> prefill_seeds;
    // std::vector<bool> prefill_return_lgts;
    DataArray<int> prefill_inp_ids;
    DataArray<uint8_t> prefill_bids;
    DataArray<int> prefill_starts;
    DataArray<float> prefill_top_ps;
    DataArray<float> prefill_temperatures;
    DataArray<int> prefill_seeds;
    DataArray<bool> prefill_return_lgts;

    int decode_bsz;
    // std::vector<std::string> decode_req_ids;
    StringArray decode_req_ids;
    // std::vector<int> decode_inp_ids;
    // std::vector<uint8_t> decode_bids;
    // std::vector<int> decode_starts;
    // std::vector<float> decode_top_ps;
    // std::vector<float> decode_temperatures;
    // std::vector<int> decode_seeds;
    // std::vector<bool> decode_return_lgts;

    DataArray<int> decode_inp_ids;
    DataArray<uint8_t> decode_bids;
    DataArray<int> decode_starts;
    DataArray<float> decode_top_ps;
    DataArray<float> decode_temperatures;
    DataArray<int> decode_seeds;
    DataArray<bool> decode_return_lgts;

    std::string GetLoraName();

    template<typename clear_type>
    void clear(clear_type* ref, int bound) {
        for (int i=0; i<bound; i++) {
            *(ref + i) = static_cast<clear_type>(0);
        }
    };

    void ClearPrefill() {
        this->prefill_bsz = 0;
        this->prefill_batch_maxlen = 0;
        this->prefill_req_ids.clear();
        this->prefill_inp_ids.clear();
        this->prefill_bids.clear();
        this->prefill_starts.clear();
        this->prefill_temperatures.clear();
        this->prefill_seeds.clear();
        this->prefill_top_ps.clear();
        this->prefill_return_lgts.clear();
        this->prefill_starts.push_back(0);
    };
    void ClearDecode() {
        this->decode_bsz = 0;
        this->decode_req_ids.clear();
        this->decode_inp_ids.clear();
        this->decode_bids.clear();
        this->decode_starts.clear();
        this->decode_temperatures.clear();
        this->decode_seeds.clear();
        this->decode_top_ps.clear();
        this->decode_return_lgts.clear();
        this->decode_starts.push_back(0);
    };

    void Empty() {
        this->prefill_examples.clear();
        this->decoding_examples.clear();
        this->batch_lora_name = std::string("ANY");
        this->all_eos = true;
        this->is_decoding = false;
        this->ClearPrefill();
        this->ClearDecode();
    };
    
    BatchInputPreparer(int max_batch_size, int max_seq_length) {
        if (max_batch_size > uint8_maxval) {
            printf("not able to set max batch size > uint8_t's max_val(256)\n");
        } else {
            printf("initializing BatchInputPreparer, max_B=%i, max_BL=%i\n", max_batch_size, max_seq_length);
        }
        this->max_batch_size = max_batch_size;
        this->max_seq_length = max_seq_length;
        this->prefill_req_ids.Init(512, max_batch_size);
        this->decode_req_ids.Init(512, max_batch_size);

        this->prefill_inp_ids.Init(max_seq_length);
        this->prefill_bids.Init(max_seq_length);
        this->prefill_starts.Init(max_batch_size+1);
        this->prefill_seeds.Init(max_batch_size);
        this->prefill_temperatures.Init(max_batch_size);
        this->prefill_top_ps.Init(max_batch_size);
        this->prefill_return_lgts.Init(max_batch_size);

        this->decode_inp_ids.Init(max_batch_size);
        this->decode_bids.Init(max_seq_length);
        this->decode_starts.Init(max_batch_size+1);
        this->decode_seeds.Init(max_batch_size);
        this->decode_temperatures.Init(max_batch_size);
        this->decode_top_ps.Init(max_batch_size);
        this->decode_return_lgts.Init(max_batch_size);
        this->Empty();
    };

    void AddPrefill(ResponseContext* ctx_ref);

    void PrefillUpdate(int data_id, StringArray request_ids, std::vector<bool> is_eos, std::vector<int> token_ids, std::shared_ptr<ContextPool> pool, PipelineKVPool* kv_cache_ref, BatchLogitsRes top_logits_info);
    void DecodeUpdate(int data_id, StringArray request_ids, std::vector<bool> is_eos, std::vector<int> token_ids, std::shared_ptr<ContextPool> pool, PipelineKVPool* kv_cache_ref, BatchLogitsRes top_logits_info);

    // void check_print() {
    //     // int bids_len = (int)(this->input_bids.size());
    //     int bids_len = this->input_bids_ct;
    //     int last_bid = static_cast<int>(input_bids[bids_len-1]);
    //     if (!(last_bid+1) == this->dynamic_bsz && this->dynamic_bsz > 0) {
    //         printf("assert error: last batch_id(%i) != dynamic_bsz(%i) in InputPreparer\n", (last_bid+1), this->dynamic_bsz);
    //     }
    //     // printf("current batch status: is_decodig=%i, bsz=%i, len(input_ids)=%i, len(input_bids)=%i, len(starts)=%i\n", static_cast<int>(this->is_decoding), last_bid+1, (int)(this->input_ids.size()), bids_len, (int)(this->input_starts.size()));
    //     printf("current batch status: is_decodig=%i, bsz=%i, len(input_ids)=%i, len(input_bids)=%i, len(starts)=%i\n", static_cast<int>(this->is_decoding), last_bid+1, (int)(this->input_ids_ct), bids_len, (int)(this->dynamic_bsz+1));
    //     std::string sb = std::string("");
    //     // for (int i=0; i<this->input_starts.size(); i++) {
    //     for (int i=0; i<(this->dynamic_bsz+1); i++) {
    //         sb += (std::string(",")+std::to_string(this->input_starts[i]));
    //     }
    //     // printf("starts=%s, input_ids.back()=%i, input_bids.back()=%i\n", sb.c_str(), this->input_ids.back(), static_cast<int>(this->input_bids.back()));
    //     printf("starts=%s, input_ids.back()=%i, input_bids.back()=%i\n", sb.c_str(), this->input_ids[this->input_ids_ct-1], static_cast<int>(this->input_bids[this->input_bids_ct-1]));
    // }
    // void UploadInputs(bool is_prefill, int inp_gpu_id, int out_gpu_id, const Data& input_ids, const Data& inp_batch_ids, const Data& query_pos_starts, const Data& key_pos_starts, const Data& temperature, float* cpu_temperatures, const Data& seeds_tensor, int* cpu_seeds, const Data& top_ps, float* cpu_top_ps, int dynamic_bsz);
    
    bool PrefillShouldReturnLogits(bool* batch_return_lgts);
    bool DecodeShouldReturnLogits(bool* batch_return_lgts);
    void UploadInputs(bool is_prefill, int inp_gpu_id, int out_gpu_id, const Data& input_ids, const Data& inp_batch_ids, const Data& query_pos_starts, const Data& key_pos_starts, const Data& temperature, const Data& seeds_tensor, const Data& top_ps, int dynamic_bsz_check);
};

} // namespace liteqwen

#endif //_POOL_H
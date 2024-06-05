#include "core_cpu.h"

void init_device_property_for_device(int world_size, int64_t device_start, int64_t device_end);
void xformer_self_attention_fwd(const liteqwen::Data& attended_tmp_out, const liteqwen::Data& query, const liteqwen::Data& key, const liteqwen::Data& value, const liteqwen::Data& seqstart_q, const liteqwen::Data& seqstart_k, const liteqwen::Data& seqlen_k, const liteqwen::Data& flash_attn_workspace, int dynamic_bsz, int dynamic_length, int query_heads,  int max_q_len, int max_k_len, int channel); // std::vector<int> view_shape_q=std::vector<int>(), std::vector<int> view_shape_k=std::vector<int>()

void xformer_self_attention_fwd_old(const liteqwen::Data& attended_tmp_out, const liteqwen::Data& query, const liteqwen::Data& key, const liteqwen::Data& value, const liteqwen::Data& flash_attn_workspace, int bsz, int query_heads, int kv_heads, int inp_len, int channel);

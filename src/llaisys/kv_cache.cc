#include "llaisys_kv_cache.hpp"

__C {
    llaisysKVCache_t llaisysKVCacheCreate(
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id,
        size_t num_layers,
        size_t batch_size,
        size_t num_heads,
        size_t head_dim,
        size_t max_seq_len
    ) {
        return new LlaisysKVCache{llaisys::KV_Cache::create(dtype, device_type, device_id, num_layers, batch_size, num_heads, head_dim, max_seq_len)};
    }

    void llaisysKVCacheDestroy(llaisysKVCache_t cache) {
        delete cache;
    }

    void llaisysKVCacheUpdate(
        llaisysKVCache_t cache,
        llaisysTensor_t k,
        llaisysTensor_t v,
        size_t layer_idx,
        size_t seq_len
    ) {
        return cache->kv_cache->update(k->tensor, v->tensor, layer_idx, seq_len);
    }

    void llaisysKVCacheGet(
        llaisysKVCache_t cache,
        llaisysTensor_t k_out,
        llaisysTensor_t v_out,
        size_t layer_idx
    ) {
        return cache->kv_cache->get(k_out->tensor, v_out->tensor, layer_idx);
    }

    void llaisysKVCacheReset(llaisysKVCache_t cache) {
        return cache->kv_cache->reset();
    }

    size_t llaisysKVCacheGetSize(llaisysKVCache_t cache, size_t layer_idx) {
        return cache->kv_cache->get_size(layer_idx);
    }

    void llaisysKVCacheExpand(
        llaisysKVCache_t cache,
        size_t new_max_seq_len
    ) {
        return cache->kv_cache->expand(new_max_seq_len);
    }

    size_t llaisysKVCacheGetMaxSeqLen(llaisysKVCache_t cache) {
        return cache->kv_cache->get_max_seq_len();
    }
}
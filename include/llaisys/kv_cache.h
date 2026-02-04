#ifndef LLAISYS_KV_CACHE_H
#define LLAISYS_KV_CACHE_H

#include "tensor.h"

__C {

    typedef struct LlaisysKVCache *llaisysKVCache_t;

    /**
     * @brief Create a new KV Cache
     * @param dtype Data type of the cache
     * @param device_type Device type (CPU, GPU, etc.)
     * @param device_id Device ID
     * @param num_layers Number of transformer layers
     * @param batch_size Batch size
     * @param num_heads Number of attention heads
     * @param head_dim Dimension of each attention head
     * @param max_seq_len Maximum sequence length for the cache
     * @return Pointer to the created KV Cache
     */
    __export llaisysKVCache_t llaisysKVCacheCreate(
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id,
        size_t num_layers,
        size_t batch_size,
        size_t num_heads,
        size_t head_dim,
        size_t max_seq_len
    );

    /**
     * @brief Destroy the KV Cache and free all resources
     * @param cache Pointer to the KV Cache
     */
    __export void llaisysKVCacheDestroy(llaisysKVCache_t cache);

    /**
     * @brief Update the KV Cache with new K and V tensors for a specific layer
     * @param cache Pointer to the KV Cache
     * @param k New K tensor to add to the cache
     * @param v New V tensor to add to the cache
     * @param layer_idx Index of the transformer layer
     * @param seq_len Length of the new sequence to add
     */
    __export void llaisysKVCacheUpdate(
        llaisysKVCache_t cache,
        llaisysTensor_t k,
        llaisysTensor_t v,
        size_t layer_idx,
        size_t seq_len
    );

    /**
     * @brief Get the cached K and V tensors for a specific layer
     * @param cache Pointer to the KV Cache
     * @param k_out Output tensor to store the cached K
     * @param v_out Output tensor to store the cached V
     * @param layer_idx Index of the transformer layer
     */
    __export void llaisysKVCacheGet(
        llaisysKVCache_t cache,
        llaisysTensor_t k_out,
        llaisysTensor_t v_out,
        size_t layer_idx
    );

    /**
     * @brief Reset the KV Cache to its initial state
     * @param cache Pointer to the KV Cache
     */
    __export void llaisysKVCacheReset(llaisysKVCache_t cache);

    /**
     * @brief Get the current size (sequence length) of the cached KV tensors for a specific layer
     * @param cache Pointer to the KV Cache
     * @param layer_idx Index of the transformer layer
     * @return Current size (sequence length) of the cache
     */
    __export size_t llaisysKVCacheGetSize(llaisysKVCache_t cache, size_t layer_idx);

    /**
     * @brief Expand the KV Cache to accommodate a larger sequence length
     * @param cache Pointer to the KV Cache
     * @param new_max_seq_len New maximum sequence length
     */
    __export void llaisysKVCacheExpand(
        llaisysKVCache_t cache,
        size_t new_max_seq_len
    );

    /**
     * @brief Get the current maximum sequence length of the KV Cache
     * @param cache Pointer to the KV Cache
     * @return Current maximum sequence length
     */
    __export size_t llaisysKVCacheGetMaxSeqLen(llaisysKVCache_t cache);
}

#endif // LLAISYS_KV_CACHE_H
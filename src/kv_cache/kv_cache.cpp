#include "kv_cache.hpp"

#include "../tensor/tensor.hpp"
#include "../core/llaisys_core.hpp"
#include "../utils.hpp"

#include <cassert>
#include <stdexcept>
#include <cstring>

namespace llaisys {

KV_Cache::KV_Cache(
    llaisysDataType_t dtype,
    llaisysDeviceType_t device_type,
    int device_id,
    size_t num_layers,
    size_t batch_size,
    size_t num_heads,
    size_t head_dim,
    size_t max_seq_len
) : 
    dtype_(dtype),
    device_type_(device_type),
    device_id_(device_id),
    num_layers_(num_layers),
    batch_size_(batch_size),
    num_heads_(num_heads),
    head_dim_(head_dim),
    max_seq_len_(max_seq_len),
    layer_caches_(num_layers)
{
    // Initialize layer caches
    for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        create_cache_tensor(layer_caches_[layer_idx].k_cache, layer_idx, true);
        create_cache_tensor(layer_caches_[layer_idx].v_cache, layer_idx, false);
        layer_caches_[layer_idx].current_size = 0;
    }
}

kv_cache_t KV_Cache::create(
    llaisysDataType_t dtype,
    llaisysDeviceType_t device_type,
    int device_id,
    size_t num_layers,
    size_t batch_size,
    size_t num_heads,
    size_t head_dim,
    size_t max_seq_len
) {
    return std::shared_ptr<KV_Cache>(
        new KV_Cache(dtype, device_type, device_id, num_layers, batch_size, num_heads, head_dim, max_seq_len)
    );
}



KV_Cache::~KV_Cache() {
    // Tensor destructors will handle freeing the memory
}

size_t KV_Cache::get_max_seq_len() const {
    return max_seq_len_;
}

void KV_Cache::update(llaisys::tensor_t k, llaisys::tensor_t v, size_t layer_idx, size_t seq_len) {
    check_layer_idx(layer_idx);
    
    LayerCache& cache = layer_caches_[layer_idx];
    
    // Ensure there's enough space in the cache
    ensure_enough_space(layer_idx, cache.current_size + seq_len);
    
    // Get the current position to write to
    size_t start_pos = cache.current_size;
    
    if(!k->isContiguous() || !v->isContiguous()) {
        throw std::runtime_error("Input tensors should be contiguous");
    }
    
    // Ensure the cache tensors are contiguous (they should be since we create them as contiguous)
    if (!cache.k_cache->isContiguous() || !cache.v_cache->isContiguous()) {
        throw std::runtime_error("KV Cache tensors should be contiguous");
    }

        // Check output tensor dimensions   
    if (k->shape().size() != 4 || v->shape().size() != 4) {
        throw std::runtime_error("Input tensors must be 4-dimensional");
    }
    if (k->shape()[0] != cache.k_cache->shape()[0] || 
        v->shape()[0] != cache.v_cache->shape()[0]) {
        throw std::runtime_error("Input tensor batch size must match cache batch size");
    }
    if (k->shape()[1] != seq_len || v->shape()[1] != seq_len) {
        throw std::runtime_error("Input tensor sequence length must match cache sequence length");
    }
    if (k->shape()[2] != cache.k_cache->shape()[2] || 
        v->shape()[2] != cache.v_cache->shape()[2]) {
        throw std::runtime_error("Input tensor number of heads must match cache");
    }
    if (k->shape()[3] != cache.k_cache->shape()[3] || 
        v->shape()[3] != cache.v_cache->shape()[3]) {
        throw std::runtime_error("Output tensor head dimension must match cache");
    }
    
    // Calculate the memory offset for the cache
    size_t element_size = cache.k_cache->elementSize();
    size_t batch_size = cache.k_cache->shape()[0];
    size_t num_heads = cache.k_cache->shape()[2];
    size_t head_dim = cache.k_cache->shape()[3];
    
    // Calculate the stride for the sequence dimension
    size_t seq_stride = cache.k_cache->strides()[1] * element_size;
    
    // Calculate the total size per batch element
    size_t per_batch_size = num_heads * head_dim * element_size;
    
    // Calculate the start address for the cache
    std::byte* k_cache_data = cache.k_cache->data();
    std::byte* v_cache_data = cache.v_cache->data();
    
    // Get the input data addresses
    const std::byte* k_data = k->data();
    const std::byte* v_data = v->data();
    
    // Copy data for each batch element
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // Calculate the offset for this batch and sequence position
        size_t batch_offset = batch_idx * cache.k_cache->strides()[0] * element_size;
        size_t seq_offset = start_pos * seq_stride;
        
        // Copy K tensor data
        std::byte* k_dest = k_cache_data + batch_offset + seq_offset;
        const std::byte* k_src = k_data + batch_idx * per_batch_size * seq_len;
        size_t k_copy_size = per_batch_size * seq_len;
        std::memcpy(k_dest, k_src, k_copy_size);
        
        // Copy V tensor data
        std::byte* v_dest = v_cache_data + batch_offset + seq_offset;
        const std::byte* v_src = v_data + batch_idx * per_batch_size * seq_len;
        size_t v_copy_size = per_batch_size * seq_len;
        std::memcpy(v_dest, v_src, v_copy_size);
    }
    
    // Update the current size of the cache
    cache.current_size += seq_len;
}

void KV_Cache::get(llaisys::tensor_t k_out, llaisys::tensor_t v_out, size_t layer_idx) {
    check_layer_idx(layer_idx);
    
    const LayerCache& cache = layer_caches_[layer_idx];
    size_t current_size = cache.current_size;
    
    if (current_size == 0) {
        // No data in cache, nothing to copy
        return;
    }
    
    // Ensure all tensors are contiguous
    if (!cache.k_cache->isContiguous() || !cache.v_cache->isContiguous()) {
        throw std::runtime_error("KV Cache tensors should be contiguous");
    }
    if (!k_out->isContiguous() || !v_out->isContiguous()) {
        throw std::runtime_error("Output tensors must be contiguous");
    }
    
    // Check output tensor dimensions
    if (k_out->shape().size() != 4 || v_out->shape().size() != 4) {
        throw std::runtime_error("Output tensors must be 4-dimensional");
    }
    if (k_out->shape()[0] != cache.k_cache->shape()[0] || 
        v_out->shape()[0] != cache.v_cache->shape()[0]) {
        throw std::runtime_error("Output tensor batch size must match cache batch size");
    }
    if (k_out->shape()[1] != current_size || v_out->shape()[1] != current_size) {
        throw std::runtime_error("Output tensor sequence length must match cache sequence length");
    }
    if (k_out->shape()[2] != cache.k_cache->shape()[2] || 
        v_out->shape()[2] != cache.v_cache->shape()[2]) {
        throw std::runtime_error("Output tensor number of heads must match cache");
    }
    if (k_out->shape()[3] != cache.k_cache->shape()[3] || 
        v_out->shape()[3] != cache.v_cache->shape()[3]) {
        throw std::runtime_error("Output tensor head dimension must match cache");
    }
    
    // Calculate the memory parameters
    size_t element_size = cache.k_cache->elementSize();
    size_t batch_size = cache.k_cache->shape()[0];
    size_t num_heads = cache.k_cache->shape()[2];
    size_t head_dim = cache.k_cache->shape()[3];
    
    // Calculate the total size per batch element
    size_t per_batch_size = num_heads * head_dim * element_size;
    
    // Get the data addresses
    const std::byte* k_cache_data = cache.k_cache->data();
    const std::byte* v_cache_data = cache.v_cache->data();
    std::byte* k_out_data = k_out->data();
    std::byte* v_out_data = v_out->data();
    
    // Copy data for each batch element
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        // Calculate batch offsets
        size_t cache_batch_offset = batch_idx * cache.k_cache->strides()[0] * element_size;
        size_t out_batch_offset = batch_idx * k_out->strides()[0] * element_size;
        
        // Copy all K tensor data for this batch at once
        const std::byte* k_src = k_cache_data + cache_batch_offset;
        std::byte* k_dest = k_out_data + out_batch_offset;
        size_t k_copy_size = per_batch_size * current_size;
        std::memcpy(k_dest, k_src, k_copy_size);
        
        // Copy all V tensor data for this batch at once
        const std::byte* v_src = v_cache_data + cache_batch_offset;
        std::byte* v_dest = v_out_data + out_batch_offset;
        size_t v_copy_size = per_batch_size * current_size;
        std::memcpy(v_dest, v_src, v_copy_size);

    }
}

void KV_Cache::reset() {
    // Reset the current size of all layer caches
    for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        layer_caches_[layer_idx].current_size = 0;
    }
}

size_t KV_Cache::get_size(size_t layer_idx) const {
    check_layer_idx(layer_idx);
    return layer_caches_[layer_idx].current_size;
}

void KV_Cache::expand(size_t new_max_seq_len) {
    if (new_max_seq_len <= max_seq_len_) {
        return; // No need to expand
    }
    
    // Update the maximum sequence length
    max_seq_len_ = new_max_seq_len;
    
    // Recreate all cache tensors with the new maximum sequence length
    for (size_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        // Save the current size
        size_t current_size = layer_caches_[layer_idx].current_size;
        
        // Create new cache tensors with the expanded size
        llaisys::tensor_t new_k_cache;
        llaisys::tensor_t new_v_cache;
        
        create_expanded_cache_tensor(new_k_cache, layer_idx, true);
        create_expanded_cache_tensor(new_v_cache, layer_idx, false);
        
        // If there's existing data, copy it to the new cache
        if (current_size > 0) {
            // Get the old cache tensors
            const LayerCache& old_cache = layer_caches_[layer_idx];
            
            // Calculate memory parameters
            size_t element_size = old_cache.k_cache->elementSize();
            size_t num_heads = old_cache.k_cache->shape()[2];
            size_t head_dim = old_cache.k_cache->shape()[3];
            
            // Calculate the total size per batch element per sequence
            size_t per_batch_per_seq_size = num_heads * head_dim * element_size;
            
            // Since both old and new tensors are contiguous (directly created), their sequence stride should be per_batch_per_seq_size
            // For contiguous tensors, strides[1] (sequence dimension) = num_heads * head_dim * element_size = per_batch_per_seq_size
            
            // Calculate the total size per batch element (all sequences)
            size_t per_batch_total_size = per_batch_per_seq_size * current_size;
            
            // Calculate the stride for the batch dimension in both old and new caches
            size_t old_batch_stride = old_cache.k_cache->strides()[0] * element_size;
            size_t new_batch_stride = new_k_cache->strides()[0] * element_size;
            
            // Get the data addresses
            const std::byte* old_k_data = old_cache.k_cache->data();
            const std::byte* old_v_data = old_cache.v_cache->data();
            std::byte* new_k_data = new_k_cache->data();
            std::byte* new_v_data = new_v_cache->data();
            
            // Copy data for each batch element
            for (size_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
                // Calculate batch offsets
                size_t old_batch_offset = batch_idx * old_batch_stride;
                size_t new_batch_offset = batch_idx * new_batch_stride;
                
                // Since both tensors are contiguous and have the same sequence stride, we can copy all sequences at once
                
                // Copy K tensor data for this batch
                const std::byte* old_k_src = old_k_data + old_batch_offset;
                std::byte* new_k_dest = new_k_data + new_batch_offset;
                std::memcpy(new_k_dest, old_k_src, per_batch_total_size);
                
                // Copy V tensor data for this batch
                const std::byte* old_v_src = old_v_data + old_batch_offset;
                std::byte* new_v_dest = new_v_data + new_batch_offset;
                std::memcpy(new_v_dest, old_v_src, per_batch_total_size);
            }
        }
        
        // Replace the old cache tensors with the new ones
        layer_caches_[layer_idx].k_cache = new_k_cache;
        layer_caches_[layer_idx].v_cache = new_v_cache;
        
        // Restore the current size
        layer_caches_[layer_idx].current_size = current_size;
    }
}

void KV_Cache::create_cache_tensor(llaisys::tensor_t& tensor, size_t layer_idx, bool is_key) {
    // K and V tensors have the same shape: [batch_size, max_seq_len, num_heads, head_dim]
    std::vector<size_t> shape = {batch_size_, max_seq_len_, num_heads_, head_dim_};
    
    tensor = llaisys::Tensor::create(shape, dtype_, device_type_, device_id_);
    
    if (!tensor) {
        throw std::runtime_error("Failed to create KV cache tensor");
    }
}

void KV_Cache::create_expanded_cache_tensor(llaisys::tensor_t& tensor, size_t layer_idx, bool is_key) {
    // Create a cache tensor with the expanded max_seq_len_
    std::vector<size_t> shape = {batch_size_, max_seq_len_, num_heads_, head_dim_};
    
    tensor = llaisys::Tensor::create(shape, dtype_, device_type_, device_id_);
    
    if (!tensor) {
        throw std::runtime_error("Failed to create expanded KV cache tensor");
    }
}

void KV_Cache::check_layer_idx(size_t layer_idx) const {
    if (layer_idx >= num_layers_) {
        throw std::out_of_range("Layer index out of range");
    }
}

void KV_Cache::ensure_enough_space(size_t layer_idx, size_t required_space) {
    if (required_space > max_seq_len_) {
        // Double the maximum sequence length until it's large enough
        size_t new_max_seq_len = max_seq_len_;
        while (new_max_seq_len < required_space) {
            new_max_seq_len *= 2;
        }
        
        // Expand the cache
        expand(new_max_seq_len);
    }
}

} // namespace llaisys::kv_cache
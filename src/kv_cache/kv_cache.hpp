#pragma once

#include "../tensor/tensor.hpp"

#include <vector>
#include <memory>
#include <tuple>

namespace llaisys{

class KV_Cache;

using kv_cache_t = std::shared_ptr<KV_Cache>;

class KV_Cache {
public:

    static kv_cache_t create(
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id,
        size_t num_layers,
        size_t batch_size,
        size_t num_heads,
        size_t head_dim,
        size_t max_seq_len
    );

    ~KV_Cache();

    void update(llaisys::tensor_t k, llaisys::tensor_t v, size_t layer_idx, size_t seq_len);
    void get(llaisys::tensor_t k_out, llaisys::tensor_t v_out, size_t layer_idx);
    void reset();
    size_t get_size(size_t layer_idx) const;
    void expand(size_t new_max_seq_len);
    size_t get_max_seq_len() const;

private:

    KV_Cache(
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id,
        size_t num_layers,
        size_t batch_size,
        size_t num_heads,
        size_t head_dim,
        size_t max_seq_len
    );

    struct LayerCache {
        llaisys::tensor_t k_cache;
        llaisys::tensor_t v_cache;
        size_t current_size;
    };

    llaisysDataType_t dtype_;
    llaisysDeviceType_t device_type_;
    int device_id_;
    size_t num_layers_;
    size_t batch_size_;
    size_t num_heads_;
    size_t head_dim_;
    size_t max_seq_len_;

    std::vector<LayerCache> layer_caches_;

    void create_cache_tensor(llaisys::tensor_t& tensor, size_t layer_idx, bool is_key);
    void create_expanded_cache_tensor(llaisys::tensor_t& tensor, size_t layer_idx, bool is_key);
    void check_layer_idx(size_t layer_idx) const;
    void ensure_enough_space(size_t layer_idx, size_t required_space);
};

} // namespace llaisys::kv_cache
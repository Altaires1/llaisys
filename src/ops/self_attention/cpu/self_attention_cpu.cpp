#include "self_attention_cpu.hpp"
#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace llaisys::ops::cpu {

// Template implementation for different data types
template <typename T>
void self_attention_impl(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    size_t qlen,
    size_t kvlen,
    size_t nh,
    size_t nkvh,
    size_t hd,
    size_t dv,
    float scale
) {
    const T *q_data = reinterpret_cast<const T *>(q);
    const T *k_data = reinterpret_cast<const T *>(k);
    const T *v_data = reinterpret_cast<const T *>(v);
    T *attn_val_data = reinterpret_cast<T *>(attn_val);
    
    // Use appropriate calculation type based on input type
    using FloatT = llaisys::float_type_t<T>;
    
    // Expand k and v if nkvh < nh
    size_t expand_factor = nh / nkvh;
    
    // Compute attention weights and output
    for (size_t seq_idx = 0; seq_idx < qlen; ++seq_idx) {
        for (size_t head_idx = 0; head_idx < nh; ++head_idx) {
            // Determine which kv head to use
            size_t kv_head_idx = head_idx / expand_factor;

            size_t mask_idx = kvlen - qlen + seq_idx;
            
            // Initialize attention weights with only the needed size
            std::vector<FloatT> attn_weights(mask_idx + 1, -INFINITY);
            
            // Compute Q*K^T * scale for this query and head
            for (size_t kv_idx = 0; kv_idx <= mask_idx; ++kv_idx) { // Causal mask: only look at previous positions
                FloatT sum = 0.0f;
                for (size_t dim_idx = 0; dim_idx < hd; ++dim_idx) {
                    // Q: [qlen, nh, hd]
                    FloatT q_val = llaisys::utils::cast<FloatT>(q_data[seq_idx * nh * hd + head_idx * hd + dim_idx]);
                    // K: [kvlen, nkvh, hd]
                    FloatT k_val = llaisys::utils::cast<FloatT>(k_data[kv_idx * nkvh * hd + kv_head_idx * hd + dim_idx]);
                    sum += q_val * k_val;
                }
                attn_weights[kv_idx] = sum * static_cast<FloatT>(scale);
            }
            
            // Compute softmax over attention weights
            FloatT max_val = -std::numeric_limits<FloatT>::infinity();
            for (size_t kv_idx = 0; kv_idx <= mask_idx; ++kv_idx) {
                if (attn_weights[kv_idx] > max_val) {
                    max_val = attn_weights[kv_idx];
                }
            }
            
            FloatT exp_sum = 0.0f;
            for (size_t kv_idx = 0; kv_idx <= mask_idx; ++kv_idx) {
                FloatT diff = attn_weights[kv_idx] - max_val;
                FloatT exp_val = (diff < -std::numeric_limits<FloatT>::max_exponent) 
                                ? 0.0 
                                : std::exp(diff);
                attn_weights[kv_idx] = exp_val;
                exp_sum += exp_val;
            }

            FloatT eps = std::numeric_limits<FloatT>::epsilon() * 10;

            if (exp_sum <= eps) { // 接近0时避免除零
                exp_sum = eps;
            }
            
            FloatT inv_exp_sum = 1.0 / exp_sum;

            for (size_t kv_idx = 0; kv_idx <= mask_idx; ++kv_idx) {
                attn_weights[kv_idx] *= inv_exp_sum;
            }
            
            // Compute weighted sum of values (attn_weights * V)
            for (size_t dim_idx = 0; dim_idx < dv; ++dim_idx) {
                FloatT sum = 0.0f;
                for (size_t kv_idx = 0; kv_idx <= mask_idx; ++kv_idx) {
                    // V: [kvlen, nkvh, dv]
                    FloatT v_val = llaisys::utils::cast<FloatT>(v_data[kv_idx * nkvh * dv + kv_head_idx * dv + dim_idx]);
                    sum += attn_weights[kv_idx] * v_val;
                }
                // attn_val: [qlen, nh, dv]
                attn_val_data[seq_idx * nh * dv + head_idx * dv + dim_idx] = llaisys::utils::cast<T>(sum);
            }
        }
    }
}

// Main implementation dispatcher
void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t qlen,
    size_t kvlen,
    size_t nh,
    size_t nkvh,
    size_t hd,
    size_t dv,
    float scale
) {
    switch (type) {
        case LLAISYS_DTYPE_F32:
            self_attention_impl<float>(attn_val, q, k, v, qlen, kvlen, nh, nkvh, hd, dv, scale);
            break;
        case LLAISYS_DTYPE_F64:
            self_attention_impl<double>(attn_val, q, k, v, qlen, kvlen, nh, nkvh, hd, dv, scale);
            break;
        case LLAISYS_DTYPE_F16:
            self_attention_impl<llaisys::fp16_t>(attn_val, q, k, v, qlen, kvlen, nh, nkvh, hd, dv, scale);
            break;
        case LLAISYS_DTYPE_BF16:
            self_attention_impl<llaisys::bf16_t>(attn_val, q, k, v, qlen, kvlen, nh, nkvh, hd, dv, scale);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}


} // namespace llaisys::ops::cpu
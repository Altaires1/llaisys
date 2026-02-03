#include "rope_cpu.hpp"

#include "../../../utils.hpp"


#include <cmath>
#include <cstring>

namespace llaisys::ops::cpu {

// Template implementation for different data types
template <typename T>
void rope_impl(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    size_t seq_len,
    size_t n_head,
    size_t head_dim,
    float theta
) {
    const T *in_data = reinterpret_cast<const T *>(in);
    T *out_data = reinterpret_cast<T *>(out);
    const int64_t *pos_ids_data = reinterpret_cast<const int64_t *>(pos_ids);
    
    const size_t dim_half = head_dim / 2;
    
    // Use appropriate calculation type based on input type
    using FloatT = llaisys::float_type_t<T>;
    
    for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
        const int64_t pos = pos_ids_data[seq_idx];
        
        for (size_t head_idx = 0; head_idx < n_head; ++head_idx) {
            const size_t base_idx = seq_idx * n_head * head_dim + head_idx * head_dim;
            
            for (size_t dim_idx = 0; dim_idx < dim_half; ++dim_idx) {
                // Calculate frequency: phi = pos / theta^(2j/d)
                const FloatT exponent = static_cast<FloatT>(dim_idx) * 2.0 / static_cast<FloatT>(head_dim);
                const FloatT theta_pow = std::pow(static_cast<FloatT>(theta), exponent);
                const FloatT phi = static_cast<FloatT>(pos) / theta_pow;
                
                const FloatT cos_phi = std::cos(phi);
                const FloatT sin_phi = std::sin(phi);
                
                // Get input values with appropriate conversion
                const FloatT a = llaisys::utils::cast<FloatT>(in_data[base_idx + dim_idx]);
                const FloatT b = llaisys::utils::cast<FloatT>(in_data[base_idx + dim_half + dim_idx]);
                
                // Apply rotation with appropriate precision:
                // a' = a * cos(phi) - b * sin(phi)
                // b' = b * cos(phi) + a * sin(phi)
                const FloatT a_rotated = a * cos_phi - b * sin_phi;
                const FloatT b_rotated = b * cos_phi + a * sin_phi;
                
                // Convert back to original type
                out_data[base_idx + dim_idx] = llaisys::utils::cast<T>(a_rotated);
                out_data[base_idx + dim_half + dim_idx] = llaisys::utils::cast<T>(b_rotated);
            }
        }
    }
}

// Dispatch function based on data type
void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    size_t seq_len,
    size_t n_head,
    size_t head_dim,
    float theta
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_impl<float>(out, in, pos_ids, seq_len, n_head, head_dim, theta);
        return;
    case LLAISYS_DTYPE_F64:
        rope_impl<double>(out, in, pos_ids, seq_len, n_head, head_dim, theta);
        return;
    case LLAISYS_DTYPE_F16:
        rope_impl<llaisys::fp16_t>(out, in, pos_ids, seq_len, n_head, head_dim, theta);
        return;
    case LLAISYS_DTYPE_BF16:
        rope_impl<llaisys::bf16_t>(out, in, pos_ids, seq_len, n_head, head_dim, theta);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}


} // namespace llaisys::ops::cpu
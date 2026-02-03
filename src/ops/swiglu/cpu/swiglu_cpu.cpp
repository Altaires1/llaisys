#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::ops::cpu {

// Core SwiGLU implementation template
template <typename T>
void swiglu_impl(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    size_t seq_len,
    size_t intermediate_size
) {
    const T *gate_data = reinterpret_cast<const T *>(gate);
    const T *up_data = reinterpret_cast<const T *>(up);
    T *out_data = reinterpret_cast<T *>(out);
    
    // Calculate total elements
    const size_t total_elements = seq_len * intermediate_size;
    
    // Use appropriate calculation type
    using FloatT = llaisys::float_type_t<T>;
    
    for (size_t i = 0; i < total_elements; ++i) {
        // Convert to calculation type for better precision
        const FloatT gate_val = llaisys::utils::cast<FloatT>(gate_data[i]);
        const FloatT up_val = llaisys::utils::cast<FloatT>(up_data[i]);
        
        // Compute SwiGLU: out_i = up_i * (gate_i / (1 + e^-gate_i))
        const FloatT sigmoid = gate_val / (1.0f + std::exp(-gate_val));
        const FloatT result = up_val * sigmoid;
        
        // Convert back to original type and store
        out_data[i] = llaisys::utils::cast<T>(result);
    }
}

// Dispatch function based on data type
void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t seq_len,
    size_t intermediate_size
) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_impl<float>(out, gate, up, seq_len, intermediate_size);
        return;
    case LLAISYS_DTYPE_F64:
        swiglu_impl<double>(out, gate, up, seq_len, intermediate_size);
        return;
    case LLAISYS_DTYPE_F16:
        swiglu_impl<llaisys::fp16_t>(out, gate, up, seq_len, intermediate_size);
        return;
    case LLAISYS_DTYPE_BF16:
        swiglu_impl<llaisys::bf16_t>(out, gate, up, seq_len, intermediate_size);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
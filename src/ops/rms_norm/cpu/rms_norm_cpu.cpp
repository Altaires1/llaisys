#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// 通用模板函数，用于支持原生运算符的类型
// 对于f32和f64，直接使用其类型进行计算
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t batch_size, size_t hidden_size, float eps) {
    for (size_t i = 0; i < batch_size; ++i) {
        const T *row_in = in + i * hidden_size;
        T *row_out = out + i * hidden_size;
        
        // Calculate sum of squares
        T sum_sq = 0.0;
        for (size_t j = 0; j < hidden_size; ++j) {
            sum_sq += row_in[j] * row_in[j];
        }
        
        // Calculate RMS
        T mean_sq = sum_sq / static_cast<T>(hidden_size);
        T rms = std::sqrt(mean_sq + static_cast<T>(eps));
        T rms_inv = 1.0 / rms;
        
        // Apply normalization and weight
        for (size_t j = 0; j < hidden_size; ++j) {
            T val = row_in[j] * rms_inv;
            T weighted_val = val * weight[j];
            row_out[j] = weighted_val;
        }
    }
}

// 专门为fp16_t类型的特化实现
template <>
void rms_norm_<llaisys::fp16_t>(llaisys::fp16_t *out, const llaisys::fp16_t *in, const llaisys::fp16_t *weight, size_t batch_size, size_t hidden_size, float eps) {
    for (size_t i = 0; i < batch_size; ++i) {
        const llaisys::fp16_t *row_in = in + i * hidden_size;
        llaisys::fp16_t *row_out = out + i * hidden_size;
        
        // Calculate sum of squares using float for better precision
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            sum_sq += val * val;
        }
        
        // Calculate RMS
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        float rms = std::sqrt(mean_sq + eps);
        float rms_inv = 1.0f / rms;
        
        // Apply normalization and weight
        for (size_t j = 0; j < hidden_size; ++j) {
            float val = llaisys::utils::cast<float>(row_in[j]) * rms_inv;
            float weight_val = llaisys::utils::cast<float>(weight[j]);
            float weighted_val = val * weight_val;
            row_out[j] = llaisys::utils::cast<llaisys::fp16_t>(weighted_val);
        }
    }
}

// 专门为bf16_t类型的特化实现
template <>
void rms_norm_<llaisys::bf16_t>(llaisys::bf16_t *out, const llaisys::bf16_t *in, const llaisys::bf16_t *weight, size_t batch_size, size_t hidden_size, float eps) {
    for (size_t i = 0; i < batch_size; ++i) {
        const llaisys::bf16_t *row_in = in + i * hidden_size;
        llaisys::bf16_t *row_out = out + i * hidden_size;
        
        // Calculate sum of squares using float for better precision
        float sum_sq = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            float val = llaisys::utils::cast<float>(row_in[j]);
            sum_sq += val * val;
        }
        
        // Calculate RMS
        float mean_sq = sum_sq / static_cast<float>(hidden_size);
        float rms = std::sqrt(mean_sq + eps);
        float rms_inv = 1.0f / rms;
        
        // Apply normalization and weight
        for (size_t j = 0; j < hidden_size; ++j) {
            float val = llaisys::utils::cast<float>(row_in[j]) * rms_inv;
            float weight_val = llaisys::utils::cast<float>(weight[j]);
            float weighted_val = val * weight_val;
            row_out[j] = llaisys::utils::cast<llaisys::bf16_t>(weighted_val);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t batch_size, size_t hidden_size, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), 
                    reinterpret_cast<const float *>(weight), batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_F64:
        return rms_norm_(reinterpret_cast<double *>(out), reinterpret_cast<const double *>(in), 
                    reinterpret_cast<const double *>(weight), batch_size, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), batch_size, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// 通用模板函数，用于支持原生运算符的类型
// 对于f32和f64，直接使用其类型进行计算
template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t batch_size, size_t hidden_size, float eps) {

    using FloatT = llaisys::float_type_t<T>;

    for (size_t i = 0; i < batch_size; ++i) {
        const T *row_in = in + i * hidden_size;
        T *row_out = out + i * hidden_size;
        
        // Calculate sum of squares
        FloatT sum_sq = 0.0;
        for (size_t j = 0; j < hidden_size; ++j) {
            sum_sq += llaisys::utils::cast<FloatT>(row_in[j]) * llaisys::utils::cast<FloatT>(row_in[j]);
        }
        
        // Calculate RMS
        FloatT mean_sq = sum_sq / static_cast<FloatT>(hidden_size);
        FloatT rms = std::sqrt(mean_sq + static_cast<FloatT>(eps));
        FloatT rms_inv = 1.0 / rms;
        
        // Apply normalization and weight
        for (size_t j = 0; j < hidden_size; ++j) {
            FloatT val = llaisys::utils::cast<FloatT>(row_in[j]) * rms_inv;
            FloatT weighted_val = val * llaisys::utils::cast<FloatT>(weight[j]);
            row_out[j] = llaisys::utils::cast<T>(weighted_val);
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
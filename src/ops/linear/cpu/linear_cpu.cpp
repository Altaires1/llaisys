#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// 通用模板函数，用于支持原生运算符的类型
template <typename T>
void linear_(
    T *out, const T *in, const T *weight,
    const T *bias, size_t batch_size, size_t in_features, size_t out_features) {
    using FloatT = llaisys::float_type_t<T>;
    // Matrix multiplication: Y = XW^T
    // in: (batch_size, in_features)
    // weight: (out_features, in_features)
    // out: (batch_size, out_features)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < out_features; ++o) {
            FloatT sum = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                // in[b][i] * weight[o][i]
                sum += llaisys::utils::cast<FloatT>(in[b * in_features + i]) * llaisys::utils::cast<FloatT>(weight[o * in_features + i]);
            }
            // Add bias if provided
            if (bias != nullptr) {
                sum += llaisys::utils::cast<FloatT>(bias[o]);
            }
            out[b * out_features + o] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(
    std::byte *out, const std::byte *in, const std::byte *weight,
    const std::byte *bias, llaisysDataType_t type,
    size_t batch_size, size_t in_features, size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias),
            batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias),
            batch_size, in_features, out_features);
    case LLAISYS_DTYPE_F64:
        return linear_(
            reinterpret_cast<double *>(out), reinterpret_cast<const double *>(in),
            reinterpret_cast<const double *>(weight), reinterpret_cast<const double *>(bias),
            batch_size, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias),
            batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// 通用模板函数，用于支持原生运算符的类型
template <typename T>
void linear_(
    T *out, const T *in, const T *weight,
    const T *bias, size_t batch_size, size_t in_features, size_t out_features) {
    // Matrix multiplication: Y = XW^T
    // in: (batch_size, in_features)
    // weight: (out_features, in_features)
    // out: (batch_size, out_features)
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < out_features; ++o) {
            T sum = 0;
            for (size_t i = 0; i < in_features; ++i) {
                // in[b][i] * weight[o][i]
                sum += in[b * in_features + i] * weight[o * in_features + i];
            }
            // Add bias if provided
            if (bias != nullptr) {
                sum += bias[o];
            }
            out[b * out_features + o] = sum;
        }
    }
}

// 专门为fp16_t类型的特化实现
template <>
void linear_<llaisys::fp16_t>(
    llaisys::fp16_t *out, const llaisys::fp16_t *in, const llaisys::fp16_t *weight,
    const llaisys::fp16_t *bias, size_t batch_size, size_t in_features, size_t out_features) {
    // Matrix multiplication: Y = XW^T，使用float进行计算
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                // 将fp16转换为float进行计算
                float in_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                float weight_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                sum += in_val * weight_val;
            }
            // Add bias if provided
            if (bias != nullptr) {
                float bias_val = llaisys::utils::cast<float>(bias[o]);
                sum += bias_val;
            }
            // 将结果转换回fp16
            out[b * out_features + o] = llaisys::utils::cast<llaisys::fp16_t>(sum);
        }
    }
}

// 专门为bf16_t类型的特化实现
template <>
void linear_<llaisys::bf16_t>(
    llaisys::bf16_t *out, const llaisys::bf16_t *in, const llaisys::bf16_t *weight,
    const llaisys::bf16_t *bias, size_t batch_size, size_t in_features, size_t out_features) {
    // Matrix multiplication: Y = XW^T，使用float进行计算
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t o = 0; o < out_features; ++o) {
            float sum = 0.0f;
            for (size_t i = 0; i < in_features; ++i) {
                // 将bf16转换为float进行计算
                float in_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                float weight_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                sum += in_val * weight_val;
            }
            // Add bias if provided
            if (bias != nullptr) {
                float bias_val = llaisys::utils::cast<float>(bias[o]);
                sum += bias_val;
            }
            // 将结果转换回bf16
            out[b * out_features + o] = llaisys::utils::cast<llaisys::bf16_t>(sum);
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
#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t index_size, size_t weight_cols) {
    for (size_t i = 0; i < index_size; ++i) {
        int64_t idx = index[i];
        // 索引检查应该在op.cpp中完成
        const T *src_row = weight + (static_cast<size_t>(idx) * weight_cols);
        T *dst_row = out + (i * weight_cols);
        std::memcpy(dst_row, src_row, weight_cols * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
              llaisysDataType_t type, size_t index_size, size_t weight_cols) {
    const auto index_data = reinterpret_cast<const int64_t *>(index);
    
    switch (type) {
    case LLAISYS_DTYPE_BYTE:
        return embedding_(reinterpret_cast<char *>(out), index_data,
                         reinterpret_cast<const char *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_BOOL:
        return embedding_(reinterpret_cast<bool *>(out), index_data,
                         reinterpret_cast<const bool *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_I8:
        return embedding_(reinterpret_cast<int8_t *>(out), index_data,
                         reinterpret_cast<const int8_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_I16:
        return embedding_(reinterpret_cast<int16_t *>(out), index_data,
                         reinterpret_cast<const int16_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_I32:
        return embedding_(reinterpret_cast<int32_t *>(out), index_data,
                         reinterpret_cast<const int32_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_I64:
        return embedding_(reinterpret_cast<int64_t *>(out), index_data,
                         reinterpret_cast<const int64_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_U8:
        return embedding_(reinterpret_cast<uint8_t *>(out), index_data,
                         reinterpret_cast<const uint8_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_U16:
        return embedding_(reinterpret_cast<uint16_t *>(out), index_data,
                         reinterpret_cast<const uint16_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_U32:
        return embedding_(reinterpret_cast<uint32_t *>(out), index_data,
                         reinterpret_cast<const uint32_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_U64:
        return embedding_(reinterpret_cast<uint64_t *>(out), index_data,
                         reinterpret_cast<const uint64_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), index_data,
                         reinterpret_cast<const llaisys::fp16_t *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), index_data,
                         reinterpret_cast<const float *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_F64:
        return embedding_(reinterpret_cast<double *>(out), index_data,
                         reinterpret_cast<const double *>(weight), index_size, weight_cols);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), index_data,
                         reinterpret_cast<const llaisys::bf16_t *>(weight), index_size, weight_cols);
    // F8类型目前没有对应的C++类型定义
    case LLAISYS_DTYPE_F8:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    // 复数类型目前没有对应的C++类型定义
    case LLAISYS_DTYPE_C16:
    case LLAISYS_DTYPE_C32:
    case LLAISYS_DTYPE_C64:
    case LLAISYS_DTYPE_C128:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    case LLAISYS_DTYPE_INVALID:
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

}
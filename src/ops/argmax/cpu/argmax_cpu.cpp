#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cstdint>

// 辅助函数：将std::byte*转换为实际的索引类型指针
template <typename IdxT>
IdxT* get_index_ptr(std::byte* ptr) {
    return reinterpret_cast<IdxT*>(ptr);
}

// 辅助函数：存储索引值
template <typename IdxT>
void store_index(std::byte* max_idx_ptr, size_t index) {
    *get_index_ptr<IdxT>(max_idx_ptr) = static_cast<IdxT>(index);
}

// 辅助函数：根据索引类型存储索引值
void store_index_by_type(std::byte* max_idx_ptr, llaisysDataType_t idx_dtype, size_t index) {
    switch (idx_dtype) {
        case LLAISYS_DTYPE_I8:  store_index<int8_t>(max_idx_ptr, index); break;
        case LLAISYS_DTYPE_I16: store_index<int16_t>(max_idx_ptr, index); break;
        case LLAISYS_DTYPE_I32: store_index<int32_t>(max_idx_ptr, index); break;
        case LLAISYS_DTYPE_I64: store_index<int64_t>(max_idx_ptr, index); break;
        case LLAISYS_DTYPE_U8:  store_index<uint8_t>(max_idx_ptr, index); break;
        case LLAISYS_DTYPE_U16: store_index<uint16_t>(max_idx_ptr, index); break;
        case LLAISYS_DTYPE_U32: store_index<uint32_t>(max_idx_ptr, index); break;
        case LLAISYS_DTYPE_U64: store_index<uint64_t>(max_idx_ptr, index); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(idx_dtype);
    }
}

// 通用argmax实现，用于整数类型
template <typename ValT>
void argmax_(std::byte* max_idx, std::byte* max_val, const std::byte* vals,
            llaisysDataType_t idx_dtype, size_t numel) {
    
    const ValT* vals_ptr = reinterpret_cast<const ValT*>(vals);
    ValT* max_val_ptr = reinterpret_cast<ValT*>(max_val);
    
    size_t index = 0;
    ValT value = vals_ptr[0];
    
    for (size_t i = 1; i < numel; ++i) {
        if (vals_ptr[i] > value) {
            value = vals_ptr[i];
            index = i;
        }
    }
    
    // 存储最大值
    *max_val_ptr = value;
    
    // 存储索引，支持多种索引类型
    store_index_by_type(max_idx, idx_dtype, index);
}

// 为浮点数类型的特殊处理（如fp16、bf16和f64）
void argmax_float_(std::byte* max_idx, std::byte* max_val, const std::byte* vals,
                  llaisysDataType_t idx_dtype, llaisysDataType_t val_dtype, size_t numel) {
    
    size_t index = 0;
    double value; // 使用double来处理所有浮点类型，包括f64
    double current;
    
    // 初始化最大值
    switch (val_dtype) {
        case LLAISYS_DTYPE_F32: value = *reinterpret_cast<const float*>(vals); break;
        case LLAISYS_DTYPE_F64: value = *reinterpret_cast<const double*>(vals); break;
        case LLAISYS_DTYPE_F16: value = llaisys::utils::_f16_to_f32(*reinterpret_cast<const llaisys::fp16_t*>(vals)); break;
        case LLAISYS_DTYPE_BF16: value = llaisys::utils::_bf16_to_f32(*reinterpret_cast<const llaisys::bf16_t*>(vals)); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(val_dtype); return;
    }
    
    // 查找最大值
    for (size_t i = 1; i < numel; ++i) {
        const std::byte* current_val_ptr = vals + i * llaisys::utils::dsize(val_dtype);
        
        switch (val_dtype) {
            case LLAISYS_DTYPE_F32: current = *reinterpret_cast<const float*>(current_val_ptr); break;
            case LLAISYS_DTYPE_F64: current = *reinterpret_cast<const double*>(current_val_ptr); break;
            case LLAISYS_DTYPE_F16: current = llaisys::utils::_f16_to_f32(*reinterpret_cast<const llaisys::fp16_t*>(current_val_ptr)); break;
            case LLAISYS_DTYPE_BF16: current = llaisys::utils::_bf16_to_f32(*reinterpret_cast<const llaisys::bf16_t*>(current_val_ptr)); break;
            default: EXCEPTION_UNSUPPORTED_DATATYPE(val_dtype); return;
        }
        
        if (current > value) {
            value = current;
            index = i;
        }
    }
    
    // 存储最大值
    switch (val_dtype) {
        case LLAISYS_DTYPE_F32: *reinterpret_cast<float*>(max_val) = static_cast<float>(value); break;
        case LLAISYS_DTYPE_F64: *reinterpret_cast<double*>(max_val) = value; break;
        case LLAISYS_DTYPE_F16: *reinterpret_cast<llaisys::fp16_t*>(max_val) = llaisys::utils::_f32_to_f16(static_cast<float>(value)); break;
        case LLAISYS_DTYPE_BF16: *reinterpret_cast<llaisys::bf16_t*>(max_val) = llaisys::utils::_f32_to_bf16(static_cast<float>(value)); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(val_dtype); return;
    }
    
    // 存储索引，支持多种索引类型
    store_index_by_type(max_idx, idx_dtype, index);
}

namespace llaisys::ops::cpu {
void argmax(std::byte* max_idx, std::byte* max_val, const std::byte* vals,
           llaisysDataType_t idx_dtype, llaisysDataType_t val_dtype, size_t numel) {
    // 处理特殊值类型映射
    llaisysDataType_t actual_val_dtype = val_dtype;
    switch (val_dtype) {
        case LLAISYS_DTYPE_BYTE:
            actual_val_dtype = LLAISYS_DTYPE_U8; // 将BYTE视为U8
            break;
        case LLAISYS_DTYPE_BOOL:
            actual_val_dtype = LLAISYS_DTYPE_U8; // 将BOOL视为U8
            break;
        case LLAISYS_DTYPE_F8:
        case LLAISYS_DTYPE_C16:
        case LLAISYS_DTYPE_C32:
        case LLAISYS_DTYPE_C64:
        case LLAISYS_DTYPE_C128:
            EXCEPTION_UNSUPPORTED_DATATYPE(val_dtype);
            return;
        default:
            break;
    }
    
    // 检查索引类型是否有效
    switch (idx_dtype) {
        case LLAISYS_DTYPE_I8:
        case LLAISYS_DTYPE_I16:
        case LLAISYS_DTYPE_I32:
        case LLAISYS_DTYPE_I64:
        case LLAISYS_DTYPE_U8:
        case LLAISYS_DTYPE_U16:
        case LLAISYS_DTYPE_U32:
        case LLAISYS_DTYPE_U64:
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(idx_dtype);
            return;
    }
    
    // 根据值类型调用相应的实现
    switch (actual_val_dtype) {
        // 浮点类型使用特殊处理函数
        case LLAISYS_DTYPE_F32:
        case LLAISYS_DTYPE_F64:
        case LLAISYS_DTYPE_F16:
        case LLAISYS_DTYPE_BF16:
            argmax_float_(max_idx, max_val, vals, idx_dtype, actual_val_dtype, numel);
            break;
            
        // 整数类型使用通用模板函数
        case LLAISYS_DTYPE_I8:
            argmax_<int8_t>(max_idx, max_val, vals, idx_dtype, numel);
            break;
        case LLAISYS_DTYPE_I16:
            argmax_<int16_t>(max_idx, max_val, vals, idx_dtype, numel);
            break;
        case LLAISYS_DTYPE_I32:
            argmax_<int32_t>(max_idx, max_val, vals, idx_dtype, numel);
            break;
        case LLAISYS_DTYPE_I64:
            argmax_<int64_t>(max_idx, max_val, vals, idx_dtype, numel);
            break;
        case LLAISYS_DTYPE_U8:
            argmax_<uint8_t>(max_idx, max_val, vals, idx_dtype, numel);
            break;
        case LLAISYS_DTYPE_U16:
            argmax_<uint16_t>(max_idx, max_val, vals, idx_dtype, numel);
            break;
        case LLAISYS_DTYPE_U32:
            argmax_<uint32_t>(max_idx, max_val, vals, idx_dtype, numel);
            break;
        case LLAISYS_DTYPE_U64:
            argmax_<uint64_t>(max_idx, max_val, vals, idx_dtype, numel);
            break;
            
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(actual_val_dtype);
            return;
    }
}

}

// 注意：由于argmax_模板函数仅在同一文件内被argmax函数调用，
// 编译器会自动进行隐式实例化，因此不需要显式实例化
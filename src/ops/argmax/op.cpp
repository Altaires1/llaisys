#include "op.hpp"

#include <algorithm>

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 检查参数是否为空
    CHECK_ARGUMENT(max_idx != nullptr && max_val != nullptr && vals != nullptr, "Input tensors cannot be nullptr");
    
    // 检查vals是否为1D张量
    CHECK_ARGUMENT(vals->ndim() == 1, "vals must be a 1D tensor");
    
    // 检查max_idx和max_val是否为包含单个元素的1D张量
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->numel() == 1, "max_idx must be a 1D tensor with single element");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->numel() == 1, "max_val must be a 1D tensor with single element");
    
    // 检查max_idx是否为整型
    llaisysDataType_t idx_dtype = max_idx->dtype();
    CHECK_ARGUMENT(idx_dtype == LLAISYS_DTYPE_I8 || idx_dtype == LLAISYS_DTYPE_I16 || 
                   idx_dtype == LLAISYS_DTYPE_I32 || idx_dtype == LLAISYS_DTYPE_I64 || 
                   idx_dtype == LLAISYS_DTYPE_U8 || idx_dtype == LLAISYS_DTYPE_U16 || 
                   idx_dtype == LLAISYS_DTYPE_U32 || idx_dtype == LLAISYS_DTYPE_U64, "max_idx must be integer type");
    
    // 检查max_val和vals的类型是否相同
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    
    // 获取元素数量
    size_t num_elements = vals->numel();
    if (num_elements == 0) {
        CHECK_ARGUMENT(false, "vals tensor is empty");
    }
    
    // 获取数据指针
    void* vals_data = vals->data();
    void* max_idx_data = max_idx->data();
    void* max_val_data = max_val->data();
    
    // 根据数据类型实现argmax逻辑
    llaisysDataType_t dtype = vals->dtype();
    size_t max_index = 0;
    
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            float* data = static_cast<float*>(vals_data);
            float max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<float*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_F64: {
            double* data = static_cast<double*>(vals_data);
            double max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<double*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_F16: {
            fp16_t* data = static_cast<fp16_t*>(vals_data);
            fp16_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (utils::_f16_to_f32(data[i]) > utils::_f16_to_f32(max_value)) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<fp16_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            bf16_t* data = static_cast<bf16_t*>(vals_data);
            bf16_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (utils::_bf16_to_f32(data[i]) > utils::_bf16_to_f32(max_value)) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<bf16_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_I8: {
            int8_t* data = static_cast<int8_t*>(vals_data);
            int8_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<int8_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_I16: {
            int16_t* data = static_cast<int16_t*>(vals_data);
            int16_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<int16_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_I32: {
            int32_t* data = static_cast<int32_t*>(vals_data);
            int32_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<int32_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_I64: {
            int64_t* data = static_cast<int64_t*>(vals_data);
            int64_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<int64_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_U8: {
            uint8_t* data = static_cast<uint8_t*>(vals_data);
            uint8_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<uint8_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_U16: {
            uint16_t* data = static_cast<uint16_t*>(vals_data);
            uint16_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<uint16_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_U32: {
            uint32_t* data = static_cast<uint32_t*>(vals_data);
            uint32_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<uint32_t*>(max_val_data) = max_value;
            break;
        }
        case LLAISYS_DTYPE_U64: {
            uint64_t* data = static_cast<uint64_t*>(vals_data);
            uint64_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *static_cast<uint64_t*>(max_val_data) = max_value;
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    
    // 根据max_idx的数据类型存储索引
    switch (idx_dtype) {
        case LLAISYS_DTYPE_I8:
            *static_cast<int8_t*>(max_idx_data) = static_cast<int8_t>(max_index);
            break;
        case LLAISYS_DTYPE_I16:
            *static_cast<int16_t*>(max_idx_data) = static_cast<int16_t>(max_index);
            break;
        case LLAISYS_DTYPE_I32:
            *static_cast<int32_t*>(max_idx_data) = static_cast<int32_t>(max_index);
            break;
        case LLAISYS_DTYPE_I64:
            *static_cast<int64_t*>(max_idx_data) = static_cast<int64_t>(max_index);
            break;
        case LLAISYS_DTYPE_U8:
            *static_cast<uint8_t*>(max_idx_data) = static_cast<uint8_t>(max_index);
            break;
        case LLAISYS_DTYPE_U16:
            *static_cast<uint16_t*>(max_idx_data) = static_cast<uint16_t>(max_index);
            break;
        case LLAISYS_DTYPE_U32:
            *static_cast<uint32_t*>(max_idx_data) = static_cast<uint32_t>(max_index);
            break;
        case LLAISYS_DTYPE_U64:
            *static_cast<uint64_t*>(max_idx_data) = static_cast<uint64_t>(max_index);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(idx_dtype);
    }
}
} // namespace llaisys::ops
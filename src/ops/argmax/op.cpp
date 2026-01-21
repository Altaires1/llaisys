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
    
    // 获取数据指针 - 使用host_data函数
    auto vals_host_data = vals->host_data();
    size_t max_index = 0;
    
    // 根据数据类型实现argmax逻辑
    llaisysDataType_t dtype = vals->dtype();
    
    // 存储最大值的变量
    std::vector<uint8_t> max_val_buf(vals->elementSize());
    
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            const float* data = reinterpret_cast<const float*>(vals_host_data.get());
            float max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<float*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_F64: {
            const double* data = reinterpret_cast<const double*>(vals_host_data.get());
            double max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<double*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_F16: {
            const fp16_t* data = reinterpret_cast<const fp16_t*>(vals_host_data.get());
            fp16_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (utils::_f16_to_f32(data[i]) > utils::_f16_to_f32(max_value)) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<fp16_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            const bf16_t* data = reinterpret_cast<const bf16_t*>(vals_host_data.get());
            bf16_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (utils::_bf16_to_f32(data[i]) > utils::_bf16_to_f32(max_value)) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<bf16_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_I8: {
            const int8_t* data = reinterpret_cast<const int8_t*>(vals_host_data.get());
            int8_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<int8_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_I16: {
            const int16_t* data = reinterpret_cast<const int16_t*>(vals_host_data.get());
            int16_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<int16_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_I32: {
            const int32_t* data = reinterpret_cast<const int32_t*>(vals_host_data.get());
            int32_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<int32_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_I64: {
            const int64_t* data = reinterpret_cast<const int64_t*>(vals_host_data.get());
            int64_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<int64_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_U8: {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(vals_host_data.get());
            uint8_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<uint8_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_U16: {
            const uint16_t* data = reinterpret_cast<const uint16_t*>(vals_host_data.get());
            uint16_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<uint16_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_U32: {
            const uint32_t* data = reinterpret_cast<const uint32_t*>(vals_host_data.get());
            uint32_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<uint32_t*>(max_val_buf.data()) = max_value;
            break;
        }
        case LLAISYS_DTYPE_U64: {
            const uint64_t* data = reinterpret_cast<const uint64_t*>(vals_host_data.get());
            uint64_t max_value = data[0];
            for (size_t i = 1; i < num_elements; ++i) {
                if (data[i] > max_value) {
                    max_value = data[i];
                    max_index = i;
                }
            }
            *reinterpret_cast<uint64_t*>(max_val_buf.data()) = max_value;
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    
    // 使用load函数将最大值加载到max_val张量中
    max_val->load(max_val_buf.data());
    
    // 根据max_idx的数据类型存储索引 - 使用load函数
    std::vector<uint8_t> max_idx_buf(max_idx->elementSize());
    
    switch (idx_dtype) {
        case LLAISYS_DTYPE_I8:
            *reinterpret_cast<int8_t*>(max_idx_buf.data()) = static_cast<int8_t>(max_index);
            break;
        case LLAISYS_DTYPE_I16:
            *reinterpret_cast<int16_t*>(max_idx_buf.data()) = static_cast<int16_t>(max_index);
            break;
        case LLAISYS_DTYPE_I32:
            *reinterpret_cast<int32_t*>(max_idx_buf.data()) = static_cast<int32_t>(max_index);
            break;
        case LLAISYS_DTYPE_I64:
            *reinterpret_cast<int64_t*>(max_idx_buf.data()) = static_cast<int64_t>(max_index);
            break;
        case LLAISYS_DTYPE_U8:
            *reinterpret_cast<uint8_t*>(max_idx_buf.data()) = static_cast<uint8_t>(max_index);
            break;
        case LLAISYS_DTYPE_U16:
            *reinterpret_cast<uint16_t*>(max_idx_buf.data()) = static_cast<uint16_t>(max_index);
            break;
        case LLAISYS_DTYPE_U32:
            *reinterpret_cast<uint32_t*>(max_idx_buf.data()) = static_cast<uint32_t>(max_index);
            break;
        case LLAISYS_DTYPE_U64:
            *reinterpret_cast<uint64_t*>(max_idx_buf.data()) = static_cast<uint64_t>(max_index);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(idx_dtype);
    }
    
    // 使用load函数将索引加载到max_idx张量中
    max_idx->load(max_idx_buf.data());
}
} // namespace llaisys::ops
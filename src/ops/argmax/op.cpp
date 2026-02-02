#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

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
    
    // 检查设备一致性
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    
    // 检查contiguousness
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), 
           "Argmax: all tensors must be contiguous.");
    
    // 获取元素数量
    size_t num_elements = vals->numel();
    if (num_elements == 0) {
        CHECK_ARGUMENT(false, "vals tensor is empty");
    }
    
    // CPU implementation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                          max_idx->dtype(), vals->dtype(), num_elements);
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                          max_idx->dtype(), vals->dtype(), num_elements);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
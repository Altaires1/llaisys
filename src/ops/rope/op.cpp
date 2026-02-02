#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // Check that all tensors are on the same device
    CHECK_SAME_DEVICE(out, in, pos_ids);
    
    // Check that input and output have the same shape
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    
    // Validate input tensor shape: should be [seqlen, nhead, d] or [seqlen, nkvhead, d]
    ASSERT(out->ndim() == 3, "RoPE: input tensor must be 3-dimensional.");
    
    // Validate pos_ids shape: should be [seqlen]
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids tensor must be 1-dimensional.");
    ASSERT(pos_ids->shape()[0] == out->shape()[0], "RoPE: pos_ids length must match input sequence length.");
    // Validate pos_ids data type: should be int64
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids tensor must be int64 dtype.");
    
    // Check that head dimension is even
    const size_t head_dim = out->shape()[2];
    ASSERT(head_dim % 2 == 0, "RoPE: head dimension must be even.");
    
    // Check that all tensors are contiguous
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");
    
    // Check that input and output have the same dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    // Get dimensions
    const size_t seq_len = out->shape()[0];
    const size_t n_head = out->shape()[1];
    
    // Set device context
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    // Handle other device types if supported
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, n_head, head_dim, theta);
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
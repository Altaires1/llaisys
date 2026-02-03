#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // Check that all tensors are on the same device
    CHECK_SAME_DEVICE(out, gate, up);
    
    // Check that all tensors have the same shape
    CHECK_SAME_SHAPE(out->shape(), gate->shape());
    CHECK_SAME_SHAPE(out->shape(), up->shape());
    
    // Validate input tensor shape: should be 2-dimensional
    ASSERT(out->ndim() == 2, "SwiGLU: input tensor must be 2-dimensional.");
    
    // Check that all tensors are contiguous
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");
    
    // Check that all tensors have the same dtype
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype());
    CHECK_SAME_DTYPE(out->dtype(), up->dtype());
    
    // Get dimensions
    const size_t seq_len = out->shape()[0];
    const size_t intermediate_size = out->shape()[1];
    
    // Set device context
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    // Handle other device types if supported
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), seq_len, intermediate_size);
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
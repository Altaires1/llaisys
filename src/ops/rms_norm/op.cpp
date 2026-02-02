#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // Check that all tensors are on the same device
    CHECK_SAME_DEVICE(out, in, weight);
    
    // Check that input and output have the same shape
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    
    // Check that weight has the correct shape (should match the hidden size)
    ASSERT(weight->shape().size() == 1, "RMSNorm: weight must be 1D tensor.");
    ASSERT(weight->shape()[0] == in->shape().back(), "RMSNorm: weight size must match input's last dimension.");
    
    // Check that all tensors are contiguous
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");
    
    // Check that all tensors have the same dtype
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    
    // Get batch size and hidden size
    size_t batch_size = in->shape()[0];
    size_t hidden_size = in->shape()[1];
    
    // Set device context
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    // Handle other device types if supported
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), batch_size, hidden_size, eps);
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
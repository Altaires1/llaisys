#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    
    // Check shapes
    ASSERT(in->ndim() == 2, "Input must be 2D tensor");
    ASSERT(weight->ndim() == 2, "Weight must be 2D tensor");
    ASSERT(out->ndim() == 2, "Output must be 2D tensor");
    
    size_t batch_size = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    
    ASSERT(in_features == weight->shape()[1], "Input features must match weight's input features");
    ASSERT(out->shape()[0] == batch_size, "Output batch size must match input batch size");
    ASSERT(out->shape()[1] == out_features, "Output features must match weight's output features");
    
    if (bias != nullptr) {
        ASSERT(bias->ndim() == 1, "Bias must be 1D tensor");
        ASSERT(bias->shape()[0] == out_features, "Bias features must match output features");
    }
    
    // Check data types
    CHECK_SAME_DTYPE(in->dtype(), out->dtype(), weight->dtype());
    if (bias != nullptr) {
        CHECK_SAME_DTYPE(in->dtype(), bias->dtype());
    }
    
    // Check continuity
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "All tensors must be contiguous");
    if (bias != nullptr) {
        ASSERT(bias->isContiguous(), "Bias must be contiguous");
    }
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(), in->data(), weight->data(),
            bias != nullptr ? bias->data() : nullptr,
            out->dtype(), batch_size, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
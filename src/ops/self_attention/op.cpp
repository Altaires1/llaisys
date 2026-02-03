#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // Check that all tensors are on the same device
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    // Check tensor shapes
    ASSERT(q->ndim() == 3, "SelfAttention: query tensor must be 3-dimensional.");
    ASSERT(k->ndim() == 3, "SelfAttention: key tensor must be 3-dimensional.");
    ASSERT(v->ndim() == 3, "SelfAttention: value tensor must be 3-dimensional.");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: output tensor must be 3-dimensional.");
    
    // Validate shapes match
    ASSERT(q->shape()[0] == attn_val->shape()[0], "SelfAttention: query and output sequence lengths must match.");
    ASSERT(q->shape()[1] == attn_val->shape()[1], "SelfAttention: query and output number of heads must match.");
    ASSERT(k->shape()[0] == v->shape()[0], "SelfAttention: key and value sequence lengths must match.");
    ASSERT(k->shape()[1] == v->shape()[1], "SelfAttention: key and value number of heads must match.");
    ASSERT(q->shape()[2] == k->shape()[2], "SelfAttention: query and key head dimensions must match.");
    ASSERT(v->shape()[2] == attn_val->shape()[2], "SelfAttention: value and output head dimensions must match.");
    
    // Get dimensions
    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t hd = q->shape()[2];
    
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    size_t dv = v->shape()[2];

    ASSERT(qlen > 0, "SelfAttention: query sequence length must be greater than zero.");
    ASSERT(kvlen > 0, "SelfAttention: key/value sequence length must be greater than zero.");
    ASSERT(hd > 0, "SelfAttention: head dimension must be greater than zero.");
    ASSERT(dv > 0, "SelfAttention: value head dimension must be greater than zero.");
    ASSERT(nh > 0, "SelfAttention: number of heads must be greater than zero.");
    ASSERT(nkvh > 0, "SelfAttention: number of key/value heads must be greater than zero.");
    ASSERT(scale > 0.0f, "SelfAttention: scale must be greater than zero.");


    // Check that nh is divisible by nkvh
    ASSERT(q->shape()[1] % k->shape()[1] == 0, "SelfAttention: number of query heads must be divisible by number of key/value heads.");
    
    // Check that all tensors are contiguous
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "SelfAttention: all tensors must be contiguous.");
    
    // Set device context
    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    
    // Call the appropriate implementation based on device type
    switch (attn_val->deviceType()) {
        case LLAISYS_DEVICE_CPU:
            llaisys::ops::cpu::self_attention(
                attn_val->data(),
                q->data(),
                k->data(),
                v->data(),
                attn_val->dtype(),
                qlen,
                kvlen,
                nh,
                nkvh,
                hd,
                dv,
                scale
            );
            break;
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
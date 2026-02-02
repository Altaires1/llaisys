#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // Check dimensions
    CHECK_ARGUMENT(index->ndim() == 1, "index tensor must be 1-dimensional");
    CHECK_ARGUMENT(weight->ndim() == 2, "weight tensor must be 2-dimensional");
    CHECK_ARGUMENT(out->ndim() == 2, "output tensor must be 2-dimensional");

    // Check index data type
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "index tensor must be Int64 type");

    // Check shapes
    size_t index_size = index->shape()[0];
    size_t weight_rows = weight->shape()[0];
    size_t weight_cols = weight->shape()[1];
    CHECK_ARGUMENT(out->shape()[0] == index_size, "output tensor's first dimension must match index size");
    CHECK_ARGUMENT(out->shape()[1] == weight_cols, "output tensor's second dimension must match weight's second dimension");

    // Check data type consistency
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "output and weight tensors must have the same data type");

    // Check device consistency
    CHECK_SAME_DEVICE(out, index, weight);
    
    // Check contiguousness
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), 
           "Embedding: all tensors must be contiguous.");

    // 验证索引值是否在有效范围内 - 对所有设备类型都需要检查
    const auto index_data = static_cast<const int64_t*>(reinterpret_cast<const void*>(index->data()));
    for (size_t i = 0; i < index_size; ++i) {
        int64_t idx = index_data[i];
        CHECK_ARGUMENT(idx >= 0 && idx < static_cast<int64_t>(weight_rows), "index value out of bounds");
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             out->dtype(), index_size, weight_cols);
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
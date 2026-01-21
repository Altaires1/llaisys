#include "op.hpp"

#include <cstring>

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

    // Get element size
    size_t elem_size = utils::dsize(out->dtype());

    // Get data on host
    const auto index_data = static_cast<const int64_t*>(reinterpret_cast<const void*>(index->host_data().get()));
    const auto weight_data = static_cast<const std::byte*>(weight->host_data().get());

    // Allocate temporary memory for output on host
    std::byte* out_data = new std::byte[out->numel() * elem_size];

    // Perform embedding lookup
    for (size_t i = 0; i < index_size; ++i) {
        int64_t idx = index_data[i];
        CHECK_ARGUMENT(idx >= 0 && idx < static_cast<int64_t>(weight_rows), "index value out of bounds");
        
        // Copy the idx-th row from weight to output
        const std::byte* src_row = weight_data + (static_cast<size_t>(idx) * weight_cols * elem_size);
        std::byte* dst_row = out_data + (i * weight_cols * elem_size);
        std::memcpy(dst_row, src_row, weight_cols * elem_size);
    }

    // Load result back to output tensor
    out->load(out_data);

    // Free temporary memory
    delete[] out_data;
}
} // namespace llaisys::ops
#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

// Helper function to calculate the memory offset for a given index
inline size_t calculate_offset(const std::vector<size_t> &index, const std::vector<ptrdiff_t> &strides) {
    size_t offset = 0;
    for (size_t i = 0; i < index.size(); ++i) {
        offset += static_cast<size_t>(strides[i]) * index[i];
    }
    return offset;
}

// Recursive function to iterate through all elements of the tensor
template <typename T>
void rearrange_recursive(
    T *out, const T *in,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &in_strides,
    const std::vector<ptrdiff_t> &out_strides,
    std::vector<size_t> &index,
    size_t dim) {
    if (dim == shape.size()) {
        // Base case: we've reached the last dimension, copy the element
        size_t in_offset = calculate_offset(index, in_strides);
        size_t out_offset = calculate_offset(index, out_strides);
        out[out_offset] = in[in_offset];
        return;
    }

    // Recursively iterate through the current dimension
    for (size_t i = 0; i < shape[dim]; ++i) {
        index[dim] = i;
        rearrange_recursive(out, in, shape, in_strides, out_strides, index, dim + 1);
    }
}

// Template function to handle different data types
template <typename T>
void rearrange_impl(
    std::byte *out,
    const std::byte *in,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &in_strides,
    const std::vector<ptrdiff_t> &out_strides) {
    T *out_ptr = reinterpret_cast<T *>(out);
    const T *in_ptr = reinterpret_cast<const T *>(in);
    std::vector<size_t> index(shape.size(), 0);
    rearrange_recursive(out_ptr, in_ptr, shape, in_strides, out_strides, index, 0);
}

// Main function that dispatches to the appropriate template instantiation
void rearrange(
    std::byte *out,
    const std::byte *in,
    llaisysDataType_t type,
    size_t size,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &in_strides,
    const std::vector<ptrdiff_t> &out_strides) {
    switch (type) {
    // Integer types
    case LLAISYS_DTYPE_BYTE:
        return rearrange_impl<std::byte>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_BOOL:
        return rearrange_impl<bool>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_I8:
        return rearrange_impl<int8_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_I16:
        return rearrange_impl<int16_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_I32:
        return rearrange_impl<int32_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_I64:
        return rearrange_impl<int64_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_U8:
        return rearrange_impl<uint8_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_U16:
        return rearrange_impl<uint16_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_U32:
        return rearrange_impl<uint32_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_U64:
        return rearrange_impl<uint64_t>(out, in, shape, in_strides, out_strides);
    
    // Floating point types with native support
    case LLAISYS_DTYPE_F16:
        return rearrange_impl<llaisys::fp16_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_BF16:
        return rearrange_impl<llaisys::bf16_t>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_F32:
        return rearrange_impl<float>(out, in, shape, in_strides, out_strides);
    case LLAISYS_DTYPE_F64:
        return rearrange_impl<double>(out, in, shape, in_strides, out_strides);
    
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
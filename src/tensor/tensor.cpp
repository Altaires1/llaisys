#include "tensor.hpp"

#include "../utils.hpp"

#include "../ops/rearrange/op.hpp" 
#include <functional> 
#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::shared_ptr<const std::byte> Tensor::host_data() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    
    size_t data_size = this->numel() * this->elementSize();
    
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // If already on CPU, return a shared_ptr that doesn't own the memory
        // Note: This is safe because the tensor still owns the memory
        return std::shared_ptr<const std::byte>(this->data(), [](const std::byte*) {});
    } else {
        // Allocate host memory
        void* host_mem = std::malloc(data_size);
        if (!host_mem) {
            EXCEPTION_OUT_OF_MEMORY;
        }
        
        // Wrap in shared_ptr with custom deleter
        std::shared_ptr<const std::byte> result(
            static_cast<const std::byte*>(host_mem), 
            [](const std::byte* ptr) {
                std::free(const_cast<std::byte*>(ptr));
            }
        );
        
        // Copy data from device to host
        core::context().runtime().api()->memcpy_sync(
            const_cast<std::byte*>(result.get()),
            this->data(),
            data_size,
            LLAISYS_MEMCPY_D2H
        );
        
        return result;
    }
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t ndim = this->ndim();
    if (ndim == 0) {
        return true; // 标量张量始终是连续的
    }
    
    // 计算预期的步长，从最后一个维度开始
    ptrdiff_t expected_stride = 1; // 最后一个维度的步长应为1（考虑元素大小前）
    
    // 从倒数第二个维度开始检查
    for (size_t i = ndim - 1; i > 0; --i) {
        // 检查当前维度的步长是否等于预期步长
        if (this->strides()[i] != expected_stride) {
            return false;
        }
        // 更新下一个维度的预期步长
        expected_stride *= this->shape()[i];
    }
    
    // 检查第一个维度的步长是否等于预期步长
    return this->strides()[0] == expected_stride;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if(this->ndim() != order.size()){
        EXCEPTION_INVALID_PERMUTE_ORDER;
    }

    TensorMeta new_meta = this->_meta;
    new_meta.shape.clear();
    new_meta.strides.clear();

    std::unordered_map<size_t,bool> tag;

    size_t ndim = this->ndim();
    
    for(size_t idx : order){
        if(idx < 0 || idx >= ndim){
            EXCEPTION_INVALID_PERMUTE_ORDER;
        }
        if(tag[idx]){
            EXCEPTION_INVALID_PERMUTE_ORDER;
        }
        tag[idx] = true;
        new_meta.shape.emplace_back(this->shape()[idx]);
        new_meta.strides.emplace_back(this->strides()[idx]);
    }
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    
    //首先判断数量是否匹配
    size_t new_numel = 1;
    for (size_t s : shape) {
        new_numel *= s;
    }
    if (new_numel != this->numel()) {
        EXCEPTION_INVALID_VIEW_SHAPE;
    }

    if(isContiguous()){
        TensorMeta new_meta = this->_meta;
        new_meta.shape = shape;
        new_meta.strides.resize(shape.size());
        int64_t expected_stride = 1;
        size_t new_shape_siz = shape.size();
        for(int64_t i = new_shape_siz - 1; i >= 0; --i){
            new_meta.strides[i] = expected_stride;
            expected_stride *= shape[i];
        }
        return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
    }
    else{
        TensorMeta new_meta = this->_meta;
        new_meta.shape = shape;
        new_meta.strides.clear();
        auto& new_strides = new_meta.strides;
        //先合并连续的维度
        std::vector<std::pair<int64_t,int64_t>> merged_dims;
        size_t dim = this->ndim();
        int64_t expected_stride = this->strides()[dim - 1];
        int64_t current_shape = 1;
        int64_t current_stride = this->strides()[dim - 1];
        
        for(int64_t i = dim - 1; i >= 0 ; --i){
            if(expected_stride == this->strides()[i]){
                current_shape *= this->shape()[i];
                expected_stride *= this->shape()[i];
            }
            else{
                merged_dims.emplace_back(current_shape,current_stride);
                current_shape = this->shape()[i];
                current_stride = this->strides()[i];
            }
        }

        merged_dims.emplace_back(current_shape,current_stride);

        for(int64_t t_shape : shape){
            if(t_shape == merged_dims.back().first){
                new_strides.emplace_back(merged_dims.back().second);
                merged_dims.pop_back();
            }
            else if(t_shape > merged_dims.back().first){
                EXCEPTION_INVALID_VIEW_SHAPE;
            }
            else{
                if(merged_dims.back().first % t_shape != 0){
                    EXCEPTION_INVALID_VIEW_SHAPE;
                }
                size_t left_shape = t_shape;
                size_t right_shape = merged_dims.back().first / left_shape;
                size_t right_stride = merged_dims.back().second;
                size_t left_stride = right_stride * right_shape;
                new_strides.emplace_back(left_stride);
                merged_dims.pop_back();
                merged_dims.emplace_back(right_shape,right_stride);
            }
        }

        if(!merged_dims.empty()){
            EXCEPTION_INVALID_VIEW_SHAPE;
        }

        return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
    }

}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    if (dim >= this->ndim()) {
        EXCEPTION_INVALID_SLICE_DIM;
    }
    
    if (start > end || end > this->shape()[dim]) {
        EXCEPTION_INVALID_SLICE_RANGE;
    }
    
    TensorMeta new_meta = this->_meta;
    new_meta.shape[dim] = end - start;
    
    size_t new_offset = this->_offset + start * this->strides()[dim] * this->elementSize();
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    size_t data_size = numel() * elementSize();
    llaisysDeviceType_t device_type = this->deviceType();
    
    if (device_type == LLAISYS_DEVICE_CPU) {
        // CPU tensor, directly copy from src_ to tensor memory
        std::memcpy(data(), src_, data_size);
    } else {
        // GPU or other device tensor, use device API to copy from host to device
        core::context().setDevice(device_type, this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            data(),
            src_,
            data_size,
            LLAISYS_MEMCPY_H2D
        );
    }
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        // If already contiguous, return a new Tensor object that shares the same underlying storage
        // and metadata. This is a lightweight copy of the Tensor object itself, not its data.
        return std::shared_ptr<Tensor>(new Tensor(this->_meta, this->_storage, this->_offset));
    }

    // Create a new tensor with contiguous memory layout.
    // The `create` method ensures the default strides for the new_tensor are contiguous.
    tensor_t new_tensor = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());

    // Create a temporary tensor_t that points to the same underlying Tensor object as `this`.
    // This allows passing `this` (a const Tensor*) as a `tensor_t` (shared_ptr<Tensor>) to `rearrange`.
    // This new Tensor object will share the `_storage` with `this`.
    tensor_t current_tensor_as_shared = std::shared_ptr<Tensor>(new Tensor(this->_meta, this->_storage, this->_offset));

    // Use the rearrange operator to copy data from the current (potentially non-contiguous)
    // tensor to the newly created contiguous tensor.
    // The rearrange op handles the strided access for 'in' and writes contiguously to 'out'.
    llaisys::ops::rearrange(new_tensor, current_tensor_as_shared);

    return new_tensor;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
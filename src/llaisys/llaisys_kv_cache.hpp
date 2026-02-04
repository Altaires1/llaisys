#pragma once
#include "llaisys/kv_cache.h"
#include "llaisys_tensor.hpp"

#include "../kv_cache/kv_cache.hpp"

__C {
    typedef struct LlaisysKVCache {
        llaisys::kv_cache_t kv_cache;
    } LlaisysKVCache;
}
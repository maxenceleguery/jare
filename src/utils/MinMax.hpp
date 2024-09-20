#pragma once

#include <cuda_runtime.h>

namespace Utils {
    template<typename T>
    __host__ __device__ T min(const T t1, const T t2) {
        return t1<t2 ? t1 : t2;
    }

    template<typename T>
    __host__ __device__ T max(const T t1, const T t2) {
        return t1>t2 ? t1 : t2;
    }

    template<typename T>
    __host__ __device__ float smoothStep(const T min, const T max, const float t) {
        return min*(1-t) + max*t;
    }
};
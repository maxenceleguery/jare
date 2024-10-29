#pragma once

#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "../Vector.hpp"
#include "../Matrix4x4.hpp"

namespace Logger {
    __host__ __device__ static void debug(const char* msg) {
        #ifdef __CUDA_ARCH__
            if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
                printf("[DEBUG] : %s\n", msg);
            }
        #else
            printf("[DEBUG] : %s\n", msg);
        #endif
    }

    __host__ __device__ static void debug(const char* msg, const int value) {
        #ifdef __CUDA_ARCH__
            if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
                printf("[DEBUG] : %s : %d\n", msg, value);
            }
        #else
            printf("[DEBUG] : %s : %d\n", msg, value);
        #endif
    }

    __host__ __device__ static void debug(const char* msg, const uint value) {
        #ifdef __CUDA_ARCH__
            if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
                printf("[DEBUG] : %s : %u\n", msg, value);
            }
        #else
            printf("[DEBUG] : %s : %u\n", msg, value);
        #endif
    }

    __host__ __device__ static void debug(const char* msg, const float value) {
        #ifdef __CUDA_ARCH__
            if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
                printf("[DEBUG] : %s : %f\n", msg, value);
            }
        #else
            printf("[DEBUG] : %s : %f\n", msg, value);
        #endif
    }

    __host__ __device__ static void debug(const char* msg, const double value) {
        #ifdef __CUDA_ARCH__
            if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
                printf("[DEBUG] : %s : %f\n", msg, value);
            }
        #else
            printf("[DEBUG] : %s : %f\n", msg, value);
        #endif
    }

    template<typename T>
    __host__ __device__ static void debug(const char* msg, const Vector<T> value) {
        #ifdef __CUDA_ARCH__
            if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
                printf("[DEBUG] : %s ", msg);
                value.printCoordDevice();
            }
        #else
            printf("[DEBUG] : %s ", msg);
            value.printCoordDevice();
        #endif
    }

    __host__ __device__ static void debug(const char* msg, const Matrix4x4 value) {
        #ifdef __CUDA_ARCH__
            if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
                printf("[DEBUG] : %s ", msg);
                value.printGPU();
            }
        #else
            printf("[DEBUG] : %s ", msg);
            value.print();
        #endif
    }
}

#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#define cudaErrorCheck(call){cudaAssert(call,__FILE__,__LINE__);}

void cudaAssert(const cudaError err, const char *file, const int line);

class Shader {
    protected:
        unsigned int W, H, blocksize, nblocks, nthreads;

    public:
        __host__ __device__ Shader(const unsigned int W, const unsigned int H) : W(W), H(H) {
            blocksize = 256; // 1024 at most
            nthreads = 1;
            nblocks = nthreads*H*W / blocksize;
        };
        
        __host__ __device__ unsigned int getW() const {
            return W;
        }
        __host__ __device__ unsigned int getH() const {
            return H;
        }
        __host__ __device__ unsigned int getBlocksize() const {
            return blocksize;
        }
        __host__ __device__ unsigned int getNblocks() const {
            return nblocks;
        }
        __host__ __device__ unsigned int getNthreads() const {
            return nthreads;
        }

        //__device__ void shader(int idx, int state); // Function to override
};

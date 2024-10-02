#pragma once

#include <cuda_runtime.h>

class CudaReady {
    public:
        virtual __host__ void cuda() = 0;
        virtual __host__ void cpu() = 0;
        virtual __host__ void sync_to_cpu() = 0;
        virtual __host__ void free() = 0;
};

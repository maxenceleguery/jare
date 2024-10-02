#pragma once

#include "../Camera.hpp"

#include "Shader.hpp"

struct AggregShaderParams {
    Camera cam;
};

class AggregShader : public Shader {
    private:
        AggregShaderParams params;
    public:
        __host__ __device__ AggregShader(unsigned int W, unsigned int H) : Shader(W, H) {};
        __host__ __device__ void setParams(const AggregShaderParams _params) {
            params = _params;
        }
        __device__ void shader(int idx, int state);
        __host__ __device__ uint getMaxIndex() const {
            return H*W;
        }
};

__global__ void kernel(AggregShader shader, int state);

void compute_shader(AggregShader shader, int state);
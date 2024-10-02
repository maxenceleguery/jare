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
        __host__ __device__ AggregShader(AggregShaderParams _params) : Shader(_params.cam.getWidth(), _params.cam.getHeight()) {
            params = _params;
        };
        __device__ void shader(const int idx, int state);
        __host__ __device__ uint getMaxIndex() const {
            return H*W;
        }
};

__global__ void kernel(AggregShader shader, int state);

void compute_shader(AggregShader shader, int state);
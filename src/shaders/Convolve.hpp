#pragma once
#include "../Camera.hpp"

#include "Shader.hpp"

struct ConvolutionShaderParams {
    int kernelMatrix[3][3];
    Camera cam;
};

class ConvolutionShader : public Shader {
    private:
        ConvolutionShaderParams params;
    public:
        __host__ __device__ ConvolutionShader(const ConvolutionShaderParams _params) : Shader(_params.cam.getWidth(), _params.cam.getHeight(), 0) {
            params = _params;
        };
        __device__ void shader(const int idx);
};

__global__ void kernel(ConvolutionShader shader);

void compute_shader(ConvolutionShader shader);
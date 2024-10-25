#pragma once
#include "../Camera.hpp"

#include "Shader.hpp"

enum OpeType {
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION
};

struct OperationShaderParams {
    OpeType ope;
    float value;
    Camera cam;
};

class OperationShader : public Shader {
    private:
        OperationShaderParams params;
    public:
        __host__ __device__ OperationShader(const OperationShaderParams _params) : Shader(_params.cam.getWidth(), _params.cam.getHeight(), 0) {
            params = _params;
        };
        __device__ void shader(const int idx);
};

__global__ void kernel(OperationShader shader);

void compute_shader(OperationShader shader);
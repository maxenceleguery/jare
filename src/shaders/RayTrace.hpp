#pragma once
#include "../Ray.hpp"
#include "../Triangle.hpp"
#include "../Hit.hpp"
#include "../BVH.hpp"
#include "../utils/Array.hpp"
#include "../Camera.hpp"

#include "Shader.hpp"

struct RayTraceShaderParams {
    Array<BVH> bvhs;
    Camera cam;
    uint samplesByThread;
};

class RayTraceShader : public Shader, RandomInterface {
    private:
        RayTraceShaderParams params;
    public:
        __host__ __device__ RayTraceShader(const RayTraceShaderParams _params, unsigned long seed) : Shader(_params.cam.getWidth(), _params.cam.getHeight(), seed) {
            params = _params;
        };
        __device__ void shader(const int idx);
};

__global__ void kernel(RayTraceShader shader);

void compute_shader(RayTraceShader shader);
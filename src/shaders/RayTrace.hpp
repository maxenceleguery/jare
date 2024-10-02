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

class RayTraceShader : public Shader {
    private:
        RayTraceShaderParams params;
    public:
        __host__ __device__ RayTraceShader(const RayTraceShaderParams _params) : Shader(_params.cam.getWidth(), _params.cam.getHeight()) {
            params = _params;
        };
        __device__ void shader(const int idx, int state);
        __host__ __device__ uint getMaxIndex() const {
            return H*W*nthreads;
        }
};

__global__ void kernel(RayTraceShader shader, int state);

void compute_shader(RayTraceShader shader, int state);
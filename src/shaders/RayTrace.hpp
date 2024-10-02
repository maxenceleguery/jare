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
        __host__ __device__ RayTraceShader(unsigned int W, unsigned int H) : Shader(W, H) {};
        __host__ __device__ void setParams(const RayTraceShaderParams _params) {
            params = _params;
        }
        __device__ void shader(int idx, int state);
        __host__ __device__ uint getMaxIndex() const {
            return H*W*nthreads;
        }
};

__global__ void kernel(RayTraceShader shader, int state);

void compute_shader(RayTraceShader shader, int state);
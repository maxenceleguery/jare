#pragma once
#include "../Ray.hpp"
#include "../Triangle.hpp"
#include "../Hit.hpp"
#include "../BVH.hpp"
#include "../utils/Array.hpp"
#include "../Camera.hpp"

#include "Shader.hpp"

struct RasterizeShaderParams {
    Array<BVH> bvhs;
    Camera cam;
};

class RasterizeShader : public Shader {
    private:
        RasterizeShaderParams params;
    public:
        __host__ __device__ RasterizeShader(const RasterizeShaderParams _params, unsigned long seed) : Shader(_params.cam.getWidth(), _params.cam.getHeight(), seed) {
            params = _params;
        };
        __device__ void shader(const int idx);
        __host__ __device__ uint getMaxIndex() const {
            return H*W*nthreads;
        }
};

__global__ void kernel(RasterizeShader shader);

void compute_shader(RasterizeShader shader);
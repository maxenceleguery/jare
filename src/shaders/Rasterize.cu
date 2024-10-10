#include "Rasterize.hpp"

__device__ void RasterizeShader::shader(const int idx) {
    uint idx2 = idx%(W*H);
    uint w = idx2%W;
    uint h = idx2/W;

    Ray ray = params.cam.generate_ray(w, h);
    Vector<float> incomingLight = ray.rasterizeBVHDevice(ray, params.bvhs);

    params.cam.updatePixel(idx, Pixel(incomingLight));
}

__global__ void kernel(RasterizeShader shader) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < shader.getMaxIndex()) {
        shader.shader(idx);
    }
}

void compute_shader(RasterizeShader shader) {
    kernel<<<shader.getNblocks(), shader.getBlocksize()>>>(shader);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaDeviceSynchronize() ); // Checks for execution error
}
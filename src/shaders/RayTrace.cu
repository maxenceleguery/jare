#include "RayTrace.hpp"

__device__ void RayTraceShader::shader(const int idx, int state) {
    Vector<float> incomingLight;
    uint idx2 = idx%(W*H);
    uint w = idx2%W;
    uint h = idx2/W;
    for (int i=0;i<params.samplesByThread;i++) {
        Ray ray = params.cam.generate_ray(w, h);
        incomingLight += ray.rayTraceBVHDevice(state*499+idx*956+i*7, ray, params.bvhs);
    }
    incomingLight /= params.samplesByThread;
    params.cam.updatePixel(idx, Pixel(incomingLight));
}

__global__ void kernel(RayTraceShader shader, int state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < shader.getMaxIndex()) {
        shader.shader(idx, state);
    }
}

void compute_shader(RayTraceShader shader, int state) {
    kernel<<<shader.getNblocks(), shader.getBlocksize()>>>(shader, state);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaDeviceSynchronize() ); // Checks for execution error
}
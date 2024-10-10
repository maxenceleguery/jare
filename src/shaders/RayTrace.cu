#include "RayTrace.hpp"

__device__ void RayTraceShader::shader(const int idx) {
    Vector<float> incomingLight;
    uint idx2 = idx%(W*H);
    uint w = idx2%W;
    uint h = idx2/W;
    for (int i=0;i<params.samplesByThread;i++) {
        Ray ray = params.cam.generate_ray(w, h);
        //state = state*144965205+i*68524+idx*57635273;
        int state = rand_gen.randomValue()*100000000;
        //if (idx == 0) printf("%d\n", state);
        incomingLight += ray.rayTraceBVHDevice(state, ray, params.bvhs);
    }
    incomingLight /= params.samplesByThread;
    params.cam.updatePixel(idx, Pixel(incomingLight));
}

__global__ void kernel(RayTraceShader shader) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < shader.getMaxIndex()) {
        shader.shader(idx);
    }
}

void compute_shader(RayTraceShader shader) {
    kernel<<<shader.getNblocks(), shader.getBlocksize()>>>(shader);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaDeviceSynchronize() ); // Checks for execution error
}
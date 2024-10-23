#include "RayTrace.hpp"
#include "../Tracing.hpp"

__device__ void RayTraceShader::shader(const int idx) {
    Vector<float> incomingLight;
    CoordsPair pair = params.cam.indexToCoord(idx);
    const uint w = pair.width;
    const uint h = pair.height;
    
    for (int i=0;i<params.samplesByThread;i++) {
        Ray ray = params.cam.generate_ray(w, h);
        //randomValue(state);
        uint state = seed+484585*(idx+1)+956595*(i+1);
        state = 10000000*randomValue(state);
        //if (idx == 0) printf("%u : %u -> %f\n", idx, state, randomValue(state));
        incomingLight += Tracing::rayTraceBVHDevice(seed+484585*idx+956595*i, ray, params.bvhs);
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
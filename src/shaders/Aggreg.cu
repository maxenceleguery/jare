#include "Aggreg.hpp"

__device__ void AggregShader::shader(const int idx, int state) {
    Vector<float> partialColor;
    for(int i=0; i<nthreads; i++)
        partialColor += params.cam.getPixel(idx+i*H*W).toVector();
    partialColor /= nthreads;
    params.cam.updatePixel(idx, Pixel(partialColor));
}

__global__ void kernel(AggregShader shader, int state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < shader.getMaxIndex()) {
        shader.shader(idx, state);
    }
}

void compute_shader(AggregShader shader, int state) {
    kernel<<<shader.getNblocks(), shader.getBlocksize()>>>(shader, state);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaDeviceSynchronize() ); // Checks for execution error
}
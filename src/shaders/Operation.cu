#include "Operation.hpp"

__device__ void OperationShader::shader(const int idx) {
    Pixel pixel = params.cam.getPixel(idx);
    switch (params.ope) {
        case ADDITION:
            params.cam.setPixel(idx, Pixel(pixel.toVector()+params.value));
            break;
        case SUBTRACTION:
            params.cam.setPixel(idx, Pixel(pixel.toVector()-params.value));
            break;
        case MULTIPLICATION:
            params.cam.setPixel(idx, Pixel(pixel.toVector()*params.value));
            break;
        case DIVISION:
            params.cam.setPixel(idx, Pixel(pixel.toVector()/params.value));
            break;
    }
}

__global__ void kernel(OperationShader shader) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < shader.getMaxIndex()) {
        shader.shader(idx);
    }
}

void compute_shader(OperationShader shader) {
    kernel<<<shader.getNblocks(), shader.getBlocksize()>>>(shader);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaDeviceSynchronize() ); // Checks for execution error
}
#include "Convolve.hpp"

__device__ void ConvolutionShader::shader(const int idx) {
    CoordsPair pair = params.cam.indexToCoord(idx);
    const uint w = pair.width;
    const uint h = pair.height;

    int offset = 1;

    Vector<float> center;
    for (int x=-offset; x<offset+1; x++) {
        for (int y=-offset; y<offset+1; y++) {
            if (0 < w+x && w+x < W && 0 < h+y && h+y < H)
                center+=params.cam.getPixel(params.cam.coordToIndex(w+x, h+y)).toVector()*(params.kernelMatrix[x+offset][y+offset]/16.f);
        }
    }
    params.cam.setPixel(idx, Pixel(center));
}

__global__ void kernel(ConvolutionShader shader) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < shader.getMaxIndex()) {
        shader.shader(idx);
    }
}

void compute_shader(ConvolutionShader shader) {
    kernel<<<shader.getNblocks(), shader.getBlocksize()>>>(shader);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaDeviceSynchronize() ); // Checks for execution error
}
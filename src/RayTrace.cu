#include "RayTrace.cuh"

#define cudaErrorCheck(call){cudaAssert(call,__FILE__,__LINE__);}

void cudaAssert(const cudaError err, const char *file, const int line) {
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}

__global__ void rayTraceBVHCuda(Array<Ray> rays, Array<BVH> bvhs, Array<Pixel> colors, uint W, uint H, uint samplesByThread, uint threadsByRay, int state) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = idx%(W*H);

    if (idx < threadsByRay*W*H) {
        Vector<float> incomingLight;
        for (uint i=0;i<samplesByThread;i++)
            incomingLight += rays[idx2].rayTraceBVHDevice(idx*(i+1)*(i+2)*state, rays[idx2], bvhs);
        incomingLight/=samplesByThread;
        colors[idx] = Pixel(incomingLight);
    }
}

__global__ void threadsAggreg(Array<Pixel> colors, uint threadsByRay) {
    int idx = blockIdx.x * blockDim.x;

    if (threadIdx.x == 0) {
        Vector<float> partialColor;
        for(uint i=0;i<threadsByRay;i++)
            partialColor+=colors[idx+i].toVector();
        partialColor/=threadsByRay;
        colors[idx] = Pixel(partialColor);
    }
}

void rayTraceBVH(Array<Ray> rays, Array<BVH> bvhs, Array<Pixel> colors, uint nblocks,uint blocksize, uint W, uint H, uint samplesByThread, uint threadsByRay, int state) {
    rayTraceBVHCuda<<<nblocks,blocksize>>>(rays,bvhs,colors,W,H,samplesByThread,threadsByRay,state);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaDeviceSynchronize() ); // Checks for execution error

    threadsAggreg<<<nblocks,blocksize>>>(colors, threadsByRay);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaDeviceSynchronize() ); // Checks for execution error
}
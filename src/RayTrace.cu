#include "RayTrace.cuh"

#define cudaErrorCheck(call){cudaAssert(call,__FILE__,__LINE__);}

void cudaAssert(const cudaError err, const char *file, const int line) {
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}
__global__ void rayTraceCuda(Ray* rays, Triangle* triangles, Pixel* colors, uint nbTriangles, uint W, uint H, uint samplesByThread, uint threadsByRay, int state) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = idx%(W*H);

    if (idx < threadsByRay*W*H) {
        Vector<double> incomingLight;
        for (uint i=0;i<samplesByThread;i++)
            incomingLight += rays[idx2].rayTrace3(idx*(i+1)*state,rays[idx2],triangles,nbTriangles);
        incomingLight/=samplesByThread;
        colors[idx] = Pixel(incomingLight);
    }
}

__global__ void rayTraceBVHCuda(Ray* rays, Array<BVH>* bvhs, Pixel* colors, uint W, uint H, uint samplesByThread, uint threadsByRay, int state) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = idx%(W*H);

    if (idx < threadsByRay*W*H) {
        Vector<double> incomingLight;
        for (uint i=0;i<samplesByThread;i++)
            incomingLight += rays[idx2].rayTraceBVHDevice(idx*(i+1)*state,rays[idx2],bvhs);
        incomingLight/=samplesByThread;
        colors[idx] = Pixel(incomingLight);
    }
}

__global__ void threadsAggreg(Pixel* colors, uint threadsByRay) {
    int idx = blockIdx.x * blockDim.x;

    if (threadIdx.x == 0) {
        Vector<double> partialColor;
        for(uint i=0;i<threadsByRay;i++)
            partialColor+=colors[idx+i].toVector();
        partialColor/=threadsByRay;
        colors[idx] = Pixel(partialColor);
    }
}

void rayTrace(Ray* rays, Triangle* triangles, Pixel* colors, uint nbTriangles,uint nblocks,uint blocksize,uint W,uint H,uint samplesByThread,uint threadsByRay,int state) {
    rayTraceCuda<<<nblocks,blocksize>>>(rays,triangles,colors,nbTriangles,W,H,samplesByThread,threadsByRay,state);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error

    threadsAggreg<<<nblocks,blocksize>>>(colors, threadsByRay);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error
}

void rayTraceBVH(Ray* rays, Array<BVH>* bvhs, Pixel* colors, uint nblocks,uint blocksize, uint W, uint H, uint samplesByThread, uint threadsByRay, int state) {
    rayTraceBVHCuda<<<nblocks,blocksize>>>(rays,bvhs,colors,W,H,samplesByThread,threadsByRay,state);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error

    threadsAggreg<<<nblocks,blocksize>>>(colors, threadsByRay);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error
}
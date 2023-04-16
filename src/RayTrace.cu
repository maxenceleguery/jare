#include "RayTrace.cuh"

#define cudaErrorCheck(call){cudaAssert(call,__FILE__,__LINE__);}

void cudaAssert(const cudaError err, const char *file, const int line)
{ 
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}
__global__ void rayTraceCuda(Ray* rays, FaceCuda* faces, Pixel* colors, uint nbFaces, uint W, uint H, uint samplesByThread, uint threadsByRay, int state) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idx2 = idx%(W*H);

    if (idx < threadsByRay*W*H) {
        Vector<double> incomingLight;
        for (uint i=0;i<samplesByThread;i++)
            incomingLight += rays[idx2].rayTrace3(idx*(i+1)*state,rays[idx2],faces,nbFaces);
        incomingLight/=samplesByThread;
        colors[idx] = Pixel(incomingLight);
    }
}

void rayTrace(Ray* rays, FaceCuda* faces, Pixel* colors, uint nbFaces,uint nblocks,uint blocksize,uint W,uint H,uint samplesByThread,uint threadsByRay,int state) {
    rayTraceCuda<<<nblocks,blocksize>>>(rays,faces,colors,nbFaces,W,H,samplesByThread,threadsByRay,state);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    //cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error
}
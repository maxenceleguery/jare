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

__global__ void rayTraceCuda(Ray* rays, FaceCuda* faces, Pixel* colors, uint nbFaces, uint W, uint H) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < W*H) {
        Ray ray = rays[idx];
        Vector<double> incomingLight = Vector<double>();
        Vector<double> rayColor = Vector<double>(1.,1.,1.);
        for (uint bounce=0;bounce<ray.getMaxBounce();bounce++) {
            Hit hit = ray.simpleTraceDevice(faces, nbFaces);
            if (hit.getHasHit()) {
                ray = Ray(hit.getPoint(),ray.getDiffusionDirection(hit.getNormal(),idx));
                Vector<double> emittedLight = hit.getMaterial().getColor().toVector() * hit.getMaterial().getEmissionStrengh();
                incomingLight += emittedLight.productTermByTerm(rayColor);
                rayColor = rayColor.productTermByTerm(hit.getMaterial().getColor().toVector())*(hit.getNormal()*ray.getDirection());

            } else {
                break;
            }
        }
        colors[idx] = Pixel(incomingLight);
    }
}

void rayTrace3(Ray* rays, FaceCuda* faces, Pixel* colors, uint nbFaces,uint nblocks,uint blocksize,uint W,uint H) {
    rayTraceCuda<<<nblocks,blocksize>>>(rays,faces,colors,nbFaces,W,H);
    cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
    //cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error
}
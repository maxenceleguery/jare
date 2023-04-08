#pragma once
#include "Ray.hpp"
#include "FaceCuda.hpp"
#include "Pixel.hpp"
#include "Hit.hpp"


void cudaAssert(const cudaError err, const char *file, const int line);

__global__ void rayTraceCuda(Ray* rays, FaceCuda* faces, Pixel* colors, uint nbFaces);

void rayTrace3(Ray* rays, FaceCuda* faces, Pixel* colors, uint nbFaces,uint nblocks,uint blocksize,uint W,uint H);

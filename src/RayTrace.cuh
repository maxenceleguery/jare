#pragma once
#include "Ray.hpp"
#include "Triangle.hpp"
#include "Pixel.hpp"
#include "Hit.hpp"
#include "BVH.hpp"
#include "utils/Array.hpp"


void cudaAssert(const cudaError err, const char *file, const int line);

__global__ void rayTraceCuda(Ray* rays, Triangle* triangles, Pixel* colors, uint nbTriangles, uint W, uint H, uint samplesByThread, uint threadsByRay, int state);

__global__ void rayTraceBVHCuda(Ray* rays, Array<BVH>* bvhs, Pixel* colors, uint W, uint H, uint samplesByThread, uint threadsByRay, int state);

__global__ void threadsAggreg(Ray* rays, Triangle* triangles, Pixel* colors, uint nbTriangles, uint W, uint H, uint samplesByThread, uint threadsByRay, int state);

void rayTrace(Ray* rays, Triangle* triangles, Pixel* colors, uint nbTriangles,uint nblocks,uint blocksize,uint W,uint H, uint samplesByThread, uint threadsByRay, int state);

void rayTraceBVH(Ray* rays, Array<BVH>* bvhs, Pixel* colors, uint W, uint H, uint samplesByThread, uint threadsByRay, int state);

#pragma once

#include "Ray.hpp"

namespace Tracing {

    __host__ __device__ static void rayTriangleBVHs(Ray& ray, const Array<BVH>& bvhs, Hit& hit) {
        for (int i = 0; i<bvhs.size(); i++) {
            ray.rayTriangleBVH(bvhs[i], 0, 0, hit);
        }
    }

    __host__ static Hit simpleTraceHost(Ray& ray, const Meshes& meshes) {
        Hit finalHit;
        for (int i=0;i<meshes.size();i++) {
            for (int j=0; j<meshes[i].size(); j++) {
                Hit hit = ray.rayTriangle(meshes[i][j]);
                finalHit.update(hit);
            }
        }
        return finalHit;
    }

    __device__ static Hit simpleTraceDevice(Ray& ray, Triangle* triangles, const uint nbTriangles) {
        Hit finalHit;
        for (int i=0;i<nbTriangles;i++) {
            Hit hit = ray.rayTriangle(triangles[i]);
            finalHit.update(hit);
        }
        return finalHit;
    }

    __host__ static Pixel simpleRayTraceHost(Ray& ray, Meshes& meshes, const Pixel& backgroundColor) {
        Hit hit = simpleTraceHost(ray, meshes);
        if (hit.getHasHit())
            return hit.getMaterial().getColor();
        else
            return backgroundColor;
    }

    __host__ static Pixel rayTraceHost(Ray& ray, Meshes& meshes, uint state) {
        Vector<float> incomingLight = Vector<float>();
        Vector<float> rayColor = Vector<float>(1.,1.,1.);
        for (int bounce=0;bounce<ray.getMaxBounce();bounce++) {
            Hit hit = simpleTraceHost(ray, meshes);
            if (hit.getHasHit()) {
                ray.updateRay(hit, state);
                ray.updateLight(hit, &incomingLight, &rayColor);
                //const float p = rayColor.max();
                //if (random_gen.randomValue(state) >= p) {
                //    break;
                //}
                //rayColor *= 1.0f / p;
            } else {
                incomingLight += ray.envLight().productTermByTerm(rayColor);
                break;
            }
        }
        return Pixel(incomingLight);
    }

    __device__ static Vector<float> rayTraceDevice(uint state, Ray& ray, Triangle* triangles, uint nbTriangles) {
        Vector<float> incomingLight = Vector<float>();
        Vector<float> rayColor = Vector<float>(1.,1.,1.);
        for (int bounce=0;bounce<ray.getMaxBounce();bounce++) {
            Hit hit = simpleTraceDevice(ray, triangles, nbTriangles);
            if (hit.getHasHit()) {
                ray.updateRay(hit, state);
                ray.updateLight(hit, &incomingLight, &rayColor);
                //const float p = rayColor.max();
                //if (random_gen.randomValue(uidx) >= p) {
                //    break;
                //}
                //rayColor *= 1.0f / p;
            } else {
                incomingLight += ray.envLight().productTermByTerm(rayColor);
                break;
            }
        }
        return incomingLight;
    }

    __host__ static Pixel rayTraceBVHHost(Ray& ray, const Array<BVH>& bvhs, uint state) {
        Vector<float> incomingLight = Vector<float>();
        Vector<float> rayColor = Vector<float>(1.,1.,1.);
        for (int bounce=0;bounce<ray.getMaxBounce();bounce++) {
            Hit hit = Hit();
            rayTriangleBVHs(ray, bvhs, hit);
            if (hit.getHasHit()) {
                ray.updateRay(hit, state);
                ray.updateLight(hit, &incomingLight, &rayColor);
                //const float p = rayColor.max();
                //if (random_gen.randomValue(state) >= p) {
                //    break;
                //}
                //rayColor *= 1.0f / p;
            } else {
                incomingLight += ray.envLight().productTermByTerm(rayColor);
                break;
            }
        }
        return Pixel(incomingLight);
    }

    __device__ static Vector<float> rayTraceBVHDevice(uint state, Ray& ray, Array<BVH>& bvhs) {
        Vector<float> incomingLight = Vector<float>();
        Vector<float> rayColor = Vector<float>(1.,1.,1.);
        for (int bounce=0;bounce<ray.getMaxBounce();bounce++) {
            Hit hit = Hit();
            rayTriangleBVHs(ray, bvhs, hit);
            if (hit.getHasHit()) {
                ray.updateRay(hit, state);
                ray.updateLight(hit, &incomingLight, &rayColor);
                
                //const float p = rayColor.max();
                //if (p < 0.5) {
                //    break;
                //}
            } else {
                //incomingLight += ray.envLight().productTermByTerm(rayColor);
                //incomingLight.clamp(0.f, 1.f);
                break;
            }
        }
        return incomingLight;
    }

    __device__ static Vector<float> rasterizeBVHDevice(Ray& ray, Array<BVH> bvhs) {
        Vector<float> incomingLight = Vector<float>();
        Hit hit = Hit();
        rayTriangleBVHs(ray, bvhs, hit);
        /*
        while (!hit.getHasHit() || hit.getMaterial().getSpecularProb() > 0) {
            ray.updateRay(hit, 0);
            rayTriangleBVHs(ray, bvhs, hit);
        }*/
        if (hit.getHasHit()) incomingLight = hit.getMaterial().getColor().toVector();
        return incomingLight;
    }
    
}

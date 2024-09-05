#pragma once

#include "Vector.hpp"
#include "Line.hpp"
#include "Material.hpp"
#include "Triangle.hpp"
#include "Triangle.hpp"
#include "Hit.hpp"
#include "Mesh.hpp"
#include "BVH.hpp"

#include <cuda_runtime.h>
#include <curand.h>


class Ray : public Line {
    private:
        uint maxBounce = 5;

        __host__ __device__ double randomValue(uint state) const {
            state = state*747796405 + 2891336453;
            uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
            result = (result >> 22) ^ result;
            return result / 4294967295.0;
        }

        __host__ __device__ double randomValueNormalDistribution(const uint state) const { 
            double theta = 2 * PI * randomValue(state);
            double rho = std::sqrt(-2*std::log(randomValue(state*state)));
            return rho*std::cos(theta);
        }

        __host__ __device__ Vector<double> randomDirection(const uint state) const {
            double x;  double y;  double z;
            do {
                x = randomValueNormalDistribution(state);
                y = randomValueNormalDistribution(state*42);
                z = randomValueNormalDistribution(state*77);
            } while ( std::abs(x)<1E-5 && std::abs(y)<1E-5 && std::abs(z)<1E-5);            
            return Vector<double>(x,y,z).normalize();
        }

        __host__ __device__ int sign(const double number) const {
            if (number<0.)
                return -1;
            else if (number>0.)
                return 1;
            else
                return 0;
        }
    
    public:
        __host__ __device__ Ray() {};
        __host__ __device__ Ray(Vector<double> point0, Vector<double> direction0) : Line(point0, direction0) {};

        __host__ __device__ uint getMaxBounce() const {
            return maxBounce;
        }

        __host__ __device__ Vector<double> getDiffusionDirection(const Vector<double>& normal, uint state) const {
            Vector<double> dir = randomDirection(state);
            return dir*sign(dir*normal);
        }

        __host__ __device__ Vector<double> getSpecularDirection(const Vector<double>& normal) const {
            return direction - normal*2*(direction*normal);
        }

        __host__ __device__ void updateRay(Ray& ray, const Hit& hit, uint state) {
            Material mat = hit.getMaterial();
            Vector<double> diffusionDir = ray.getDiffusionDirection(hit.getNormal(), state);
            Vector<double> specularDir = ray.getSpecularDirection(hit.getNormal());
            Vector<double> finalDirection = diffusionDir.lerp(specularDir, mat.getSpecularSmoothness()).normalize();
            // New ray after bounce
            ray.setPoint(hit.getPoint());
            ray.setDirection(finalDirection);
        }

        __host__ __device__ void updateLight(Ray& ray, const Hit& hit, Vector<double>& incomingLight, Vector<double>& rayColor) {
            Material mat = hit.getMaterial();
            Vector<double> emittedLight = mat.getColor().toVector() * mat.getEmissionStrengh();
            incomingLight += emittedLight.productTermByTerm(rayColor);
            rayColor = rayColor.productTermByTerm(mat.getColor().toVector())*(hit.getNormal()*ray.getDirection()) * 2;
        }

        // Thanks to https://tavianator.com/2011/ray_box.html
        __host__ __device__ double distToBounds(const BoundingBox& bounds) {
            const Vector<double> tMin = (bounds.getMin() - point).productTermByTerm(invDir);
            const Vector<double> tMax = (bounds.getMax() - point).productTermByTerm(invDir);
            const Vector<double> t1 = tMin.min(tMax);
            const Vector<double> t2 = tMin.max(tMax);
            const double tNear = Utils::max(Utils::max(t1.getX(), t1.getY()), t1.getZ());
            const double tFar = Utils::min(Utils::min(t2.getX(), t2.getY()), t2.getZ());

            bool didHit = tFar >= tNear && tFar > 0;
            const double dst = didHit ? tNear > 0 ? tNear : 0 : INFINITY;
            return dst;
        }

        __host__ __device__ Hit rayTriangle(const Triangle& tri) {           
            const Vector<double> edgeAB = tri.getVertex(1) - tri.getVertex(0);
            const Vector<double> edgeAC = tri.getVertex(2) - tri.getVertex(0);
            const Vector<double> normalVector = edgeAB.crossProduct(edgeAC);
            const Vector<double> ao = point - tri.getVertex(0);
            const Vector<double> dao = ao.crossProduct(direction);

            const double determinant = -direction*normalVector;
            const double invDet = 1.0 / determinant;

            // Calculate dst to triangle & barycentric coordinates of intersection point
            const double dst = (ao*normalVector) * invDet;
            const double u = (edgeAC*dao) * invDet;
            const double v = -(edgeAB*dao) * invDet;
            const double w = 1.0 - u - v;

            // Initialize hit info
            Hit hit;
            const Vector<double> intersection = point + direction * dst;
            hit.setHasHit(std::abs(determinant) >= 1E-8 && dst >= 1E-8 && u >= 1E-8 && v >= 1E-8 && w >= 1E-8);
            hit.setPoint(intersection);
            hit.setNormal(tri.getNormalVector());
            hit.setMaterial(tri.getMaterial());
            hit.setDistance(dst);
            return hit;
        }

        __host__ __device__ void rayTriangleBVH(const BVH& bvh, const uint nodeOffset, const uint triOffset, Hit& hit) {
            Hit finalHit;
            uint stack[128];
            uint stackIndex = 0;
            stack[stackIndex++] = nodeOffset + 0;

            while (stackIndex > 0) {
                const Node node = bvh.allNodes[stack[--stackIndex]];
                const bool isLeaf = node.getTriangleCount() > 0;

                if (isLeaf) {
                    for (uint j=0; j<node.getTriangleCount(); j++) {
                        Hit hit_tmp = rayTriangle(bvh.allTriangles[triOffset + node.getTriangleIndex() + j]);
                        finalHit.update(hit_tmp);
                    }
                } else {
                    const uint childIndexA = nodeOffset + node.getChildIndex() + 0;
                    const uint childIndexB = nodeOffset + node.getChildIndex() + 1;

                    const double dstA = distToBounds(bvh.allNodes[childIndexA].getBoundingBox());
                    const double dstB = distToBounds(bvh.allNodes[childIndexB].getBoundingBox());
                    
                    // We want to look at closest child node first, so push it last
                    const bool isNearestA = dstA <= dstB;
                    const double dstNear = isNearestA ? dstA : dstB;
                    const double dstFar = isNearestA ? dstB : dstA;
                    const uint childIndexNear = isNearestA ? childIndexA : childIndexB;
                    const uint childIndexFar = isNearestA ? childIndexB : childIndexA;

                    if (dstFar < finalHit.getDistance()) {
                        stack[stackIndex++] = childIndexFar;
                    }
                    if (dstNear < finalHit.getDistance()) {
                        stack[stackIndex++] = childIndexNear;
                    }
                }
            }
            hit.update(finalHit);
        }

        __host__ Hit simpleTraceHost(const Meshes& meshes) {
            Hit finalHit;
            for (uint i=0;i<meshes.size();i++) {
                for (uint j=0; j<meshes[i].size(); j++) {
                    Hit hit = rayTriangle(meshes[i][j]);
                    finalHit.update(hit);
                }
            }
            return finalHit;
        }

        __device__ Hit simpleTraceDevice(Triangle* triangles, const uint nbTriangles) {
            Hit finalHit;
            for (uint i=0;i<nbTriangles;i++) {
                Hit hit = rayTriangle(triangles[i]);
                finalHit.update(hit);
            }
            return finalHit;
        }

        __host__ Pixel simpleRayTraceHost(Meshes& meshes, const Pixel& backgroundColor) {
            Hit hit = simpleTraceHost(meshes);
            if (hit.getHasHit())
                return hit.getMaterial().getColor();
            else
                return backgroundColor;
        }

        __host__ Pixel rayTraceHost(Meshes& meshes, uint state) {
            Vector<double> incomingLight = Vector<double>();
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<maxBounce;bounce++) {
                Hit hit = simpleTraceHost(meshes);
                if (hit.getHasHit()) {
                    updateRay(*this, hit, state);
                    updateLight(*this, hit, incomingLight, rayColor);
                } else {
                    break;
                }
            }
            return Pixel(incomingLight);
        }

        __device__ Vector<double> rayTraceDevice(const int idx, Ray ray, Triangle* triangles, uint nbTriangles) {
            Vector<double> incomingLight = Vector<double>();
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<ray.getMaxBounce();bounce++) {
                Hit hit = ray.simpleTraceDevice(triangles, nbTriangles);
                if (hit.getHasHit()) {
                    updateRay(ray, hit, idx);
                    updateLight(ray, hit, incomingLight, rayColor);
                } else {
                    break;
                }
            }
            return incomingLight;
        }

        __host__ Pixel rayTraceBVHHost(const Array<BVH>& bvhs, uint state) {
            Vector<double> incomingLight = Vector<double>();
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<maxBounce;bounce++) {
                Hit hit = Hit();
                for (uint i = 0; i<bvhs.size(); i++) {
                    rayTriangleBVH(bvhs[i], 0, 0, hit);
                }
                if (hit.getHasHit()) {
                    updateRay(*this, hit, state);
                    updateLight(*this, hit, incomingLight, rayColor);
                } else {
                    break;
                }
            }
            return Pixel(incomingLight);
        }

        __device__ Vector<double> rayTraceBVHDevice(const int idx, Ray ray, Array<BVH> bvhs) {
            Vector<double> incomingLight = Vector<double>();
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<ray.getMaxBounce();bounce++) {
                Hit hit = Hit();
                for (uint i = 0; i<bvhs.size(); i++) {
                    ray.rayTriangleBVH(bvhs[i], 0, 0, hit);
                }
                if (hit.getHasHit()) {
                    updateRay(ray, hit, idx);
                    updateLight(ray, hit, incomingLight, rayColor);
                } else {
                    break;
                }
            }
            return incomingLight;
        }

};
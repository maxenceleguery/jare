#pragma once

#include "Vector.hpp"
#include "Line.hpp"
#include "Material.hpp"
#include "Triangle.hpp"
#include "Hit.hpp"
#include "Mesh.hpp"
#include "BVH.hpp"
#include "utils/MinMax.hpp"
#include "utils/Random.hpp"


class Ray : public Line {
    private:
        uint maxBounce = 10;
        RayInfo ray_info = {1.000293f, false};
    
    public:
        __host__ __device__ Ray() {};
        __host__ __device__ Ray(Vector<float> point0, Vector<float> direction0) : Line(point0, direction0) {};

        __host__ __device__ uint getMaxBounce() const {
            return maxBounce;
        }

        // TODO Background light
        __host__ __device__ Vector<float> envLight() {
            if (true) {
                // Global illumination
                return Vector<float>(1.f, 1.f, 1.f);
            }
            const Vector<float> groundColor = Pixel(200,200,200).toVector();
            const Vector<float> skyColorHorizon = Pixel(157, 232, 229).toVector();
            const Vector<float> skyColorZenith = Pixel(255, 255, 255).toVector();
            const Vector<float> sunLightDirection = Vector<float>(0, 100.f, 100.f);
            const float sunFocus = 0.01f;
            const float sunIntensity = 0.1f;


            const float skyGradientT = std::pow(Utils::smoothStep(0.0f, 0.8f, direction.getZ()), 0.35f);
            const Vector<float> skyGradient = skyColorHorizon.lerp(skyColorZenith, skyGradientT);
            const float sun = std::pow(Utils::max(0.f, direction*sunLightDirection), sunFocus) * sunIntensity;

            const float groundToSkyT = Utils::smoothStep(-0.01f, 0.0f, direction.getZ());
            return groundColor.lerp(skyGradient, groundToSkyT) + sun * (groundToSkyT >= 1);
        }

        __host__ __device__ void updateRay(const Hit& hit, uint state) {
            Material mat = hit.getMaterial();
            const Vector<float> finalDirection = mat.trace(direction, hit.getNormal(), ray_info, state);
            // New ray after bounce
            setPoint(hit.getPoint());
            setDirection(finalDirection);
        }

        __host__ __device__ void updateLight(const Hit& hit, Vector<float>* incomingLight, Vector<float>* rayColor) const {
            Material mat = hit.getMaterial();
            mat.shade(incomingLight, rayColor, direction, hit.getNormal(), hit.getDistanceTraveled());
        }

        // Thanks to https://tavianator.com/2011/ray_box.html
        __host__ __device__ float distToBounds(const BoundingBox& bounds) const {
            const Vector<float> tMin = (bounds.getMin() - point).productTermByTerm(invDir);
            const Vector<float> tMax = (bounds.getMax() - point).productTermByTerm(invDir);
            const Vector<float> t1 = tMin.min(tMax);
            const Vector<float> t2 = tMin.max(tMax);
            const float tNear = Utils::max(Utils::max(t1.getX(), t1.getY()), t1.getZ());
            const float tFar = Utils::min(Utils::min(t2.getX(), t2.getY()), t2.getZ());

            bool didHit = tFar >= tNear && tFar > 0;
            const float dst = didHit ? tNear > 0 ? tNear : 0 : INFINITY;
            return dst;
        }

        __host__ __device__ Hit rayTriangle(const Triangle& tri) const {           
            const Vector<float> edgeAB = tri.getVertex(1) - tri.getVertex(0);
            const Vector<float> edgeAC = tri.getVertex(2) - tri.getVertex(0);
            const Vector<float> normalVector = edgeAB.crossProduct(edgeAC);
            const Vector<float> ao = point - tri.getVertex(0);
            const Vector<float> dao = ao.crossProduct(direction);

            const float determinant = -direction*normalVector;
            const float invDet = 1.0f / determinant;

            // Calculate dst to triangle & barycentric coordinates of intersection point
            const float dst = (ao*normalVector) * invDet;
            const float u = (edgeAC*dao) * invDet;
            const float v = -(edgeAB*dao) * invDet;
            const float w = 1.0f - u - v;

            // Initialize hit info
            Hit hit;
            const Vector<float> intersection = point + direction * dst;
            hit.setHasHit(std::abs(determinant) >= 1E-8f && dst >= 1E-8f && u >= 1E-8f && v >= 1E-8f && w >= 1E-8f);
            hit.setPoint(intersection);
            hit.setNormal(tri.getNormalVector(u, v, w));
            hit.setMaterial(tri.getMaterial());
            hit.setDistance(dst);
            return hit;
        }

        __host__ __device__ Hit rayTriangleBVH(const int bvh_index, const BVH& bvh, const uint nodeOffset, const uint triOffset) {
            Hit finalHit;
            uint stack[128];
            uint stackIndex = 0;
            stack[stackIndex++] = nodeOffset + 0;

            while (stackIndex > 0) {
                if (stackIndex >= 128) {
                    printf("Stack overflow : %d/127", stackIndex);
                }
                const Node node = bvh.allNodes[stack[--stackIndex]];
                const bool isLeaf = node.getTriangleCount() > 0;

                if (isLeaf) {
                    for (int j=0; j<node.getTriangleCount(); j++) {
                        Hit hit_tmp = rayTriangle(bvh[triOffset + node.getTriangleIndex() + j]);
                        finalHit.update(hit_tmp, bvh_index);
                    }
                } else {
                    const uint childIndexA = nodeOffset + node.getChildIndex() + 0;
                    const uint childIndexB = nodeOffset + node.getChildIndex() + 1;

                    const float dstA = distToBounds(bvh.allNodes[childIndexA].getBoundingBox());
                    const float dstB = distToBounds(bvh.allNodes[childIndexB].getBoundingBox());
                    
                    // We want to look at closest child node first, so push it last
                    const bool isNearestA = dstA <= dstB;
                    const float dstNear = isNearestA ? dstA : dstB;
                    const float dstFar = isNearestA ? dstB : dstA;
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
            return finalHit;
        }
};
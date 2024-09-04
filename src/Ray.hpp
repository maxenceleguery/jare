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
            
            /*
            double tx1 = (bounds.getMin().getX() - point.getX())*invDir.getX();
            double tx2 = (bounds.getMax().getX() - point.getX())*invDir.getX();

            double tmin = Utils::min(tx1, tx2);
            double tmax = Utils::max(tx1, tx2);

            double ty1 = (bounds.getMin().getY() - point.getY())*invDir.getY();
            double ty2 = (bounds.getMax().getY() - point.getY())*invDir.getY();

            tmin = Utils::max(tmin, Utils::min(ty1, ty2));
            tmax = Utils::min(tmax, Utils::max(ty1, ty2));

            double tz1 = (bounds.getMin().getZ() - point.getZ())*invDir.getZ();
            double tz2 = (bounds.getMax().getZ() - point.getZ())*invDir.getZ();

            tmin = Utils::max(tmin, Utils::min(tz1, tz2));
            tmax = Utils::min(tmax, Utils::max(tz1, tz2));
            
            bool didHit = tmax >= Utils::max(0.0, tmin) && tmin < INFINITY;
            return didHit ? tmin : INFINITY;
            */
        }

        __host__ __device__ Hit rayTriangle(const Triangle& tri) {
            /*
            Hit hit;
            Vector<double> intersectionPoint = tri.getIntersection((Line)(*this));
            double distance = std::sqrt((intersectionPoint-point).normSquared());
            if (intersectionPoint != Vector<double>() && distance<hit.getDistance()) {
                if (hit.getFirstDistance() < 0)
                    hit.setFirstDistance(distance);
                hit.setDistance(distance);
                hit.setHasHit(true);
                hit.setMaterial(tri.getMaterial());
                hit.setNormal(tri.getNormalVector());
                hit.setPoint(intersectionPoint);
            }
            return hit;*/
            
            Vector<double> edgeAB = tri.getVertex(1) - tri.getVertex(0);
            Vector<double> edgeAC = tri.getVertex(2) - tri.getVertex(0);
            Vector<double> normalVector = edgeAB.crossProduct(edgeAC);
            Vector<double> ao = point - tri.getVertex(0);
            Vector<double> dao = ao.crossProduct(direction);

            double determinant = -direction*normalVector;
            double invDet = 1.0 / determinant;

            // Calculate dst to triangle & barycentric coordinates of intersection point
            double dst = (ao*normalVector) * invDet;
            double u = (edgeAC*dao) * invDet;
            double v = -(edgeAB*dao) * invDet;
            double w = 1.0 - u - v;

            // Initialize hit info
            Hit hit;
            Vector<double> intersection = point + direction * dst;
            //const bool edge = intersection == tri.getVertex(0) || intersection == tri.getVertex(1) || intersection == tri.getVertex(2);
            hit.setHasHit(/*!edge &&*/ determinant >= 1E-8 && dst >= 1E-6 && u >= 1E-6 && v >= 1E-6 && w >= 1E-6);
            hit.setPoint(intersection);
            hit.setNormal(tri.getNormalVector());
            hit.setMaterial(tri.getMaterial());
            hit.setDistance(dst);
            return hit;
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

        /*
        __host__ __device__ void rayTriangleBVHRecursive(Node* node, Hit& hit) {
            if (distToBounds(node->getBoundingBox()) < INFINITY) {
                if (node->childA == nullptr && node->childB == nullptr) {
                    Mesh mesh = node->getMesh();
                    for (uint j=0; j<mesh.size(); j++) {
                        rayTriangle(mesh[j], hit);
                    }
                } else {
                    rayTriangleBVH(node->childA, hit);
                    rayTriangleBVH(node->childB, hit);
                }
            }
        }
        */

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

        __device__ Hit simpleTraceDevice(Triangle* triangles, const uint nbTriangles) {
            Hit finalHit;
            for (uint i=0;i<nbTriangles;i++) {
                Hit hit = rayTriangle(triangles[i]);
                finalHit.update(hit);
            }
            return finalHit;
        }

        __host__ Pixel rayTrace1(Meshes& meshes, const Pixel& backgroundColor) {
            Hit hit = simpleTraceHost(meshes);
            if (hit.getHasHit())
                return hit.getMaterial().getColor();
            else
                return backgroundColor;
        }

        __host__ Pixel rayTrace2(Meshes& meshes, uint state) {
            Vector<double> incomingLight = Vector<double>();
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<maxBounce;bounce++) {
                Hit hit = simpleTraceHost(meshes);
                if (hit.getHasHit()) {
                    point = hit.getPoint();
                    direction = getDiffusionDirection(hit.getNormal(),state);
                    Vector<double> emittedLight = hit.getMaterial().getColor().toVector() * hit.getMaterial().getEmissionStrengh();
                    incomingLight += emittedLight.productTermByTerm(rayColor);
                    rayColor = rayColor.productTermByTerm(hit.getMaterial().getColor().toVector())*(hit.getNormal()*direction);

                } else {
                    break;
                }
            }
            return Pixel(incomingLight);
        }

        __device__ Vector<double> rayTrace3(const int idx, Ray ray, Triangle* triangles, uint nbTriangles) {
            Vector<double> incomingLight = Vector<double>();
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<ray.getMaxBounce();bounce++) {
                Hit hit = ray.simpleTraceDevice(triangles, nbTriangles);
                if (hit.getHasHit()) {
                    Material mat = hit.getMaterial();
                    Vector<double> diffusionDir = ray.getDiffusionDirection(hit.getNormal(),idx);
                    Vector<double> specularDir = ray.getSpecularDirection(hit.getNormal());
                    Vector<double> finalDirection = (diffusionDir*(1-mat.getSpecularSmoothness()) + specularDir*mat.getSpecularSmoothness()).normalize();
                    ray = Ray(hit.getPoint(),finalDirection);
                    Vector<double> emittedLight = mat.getColor().toVector() * mat.getEmissionStrengh();
                    incomingLight += emittedLight.productTermByTerm(rayColor);
                    rayColor = rayColor.productTermByTerm(mat.getColor().toVector())*(hit.getNormal()*ray.getDirection()) * 2;
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
                    // New ray after bounce
                    setPoint(hit.getPoint());
                    setDirection(getDiffusionDirection(hit.getNormal(),state));

                    Vector<double> emittedLight = hit.getMaterial().getColor().toVector() * hit.getMaterial().getEmissionStrengh();
                    incomingLight += emittedLight.productTermByTerm(rayColor);
                    rayColor = rayColor.productTermByTerm(hit.getMaterial().getColor().toVector())*(hit.getNormal()*direction);
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
                    Material mat = hit.getMaterial();
                    Vector<double> diffusionDir = ray.getDiffusionDirection(hit.getNormal(),idx);
                    Vector<double> specularDir = ray.getSpecularDirection(hit.getNormal());
                    Vector<double> finalDirection = (diffusionDir*(1-mat.getSpecularSmoothness()) + specularDir*mat.getSpecularSmoothness()).normalize();
                    
                    // New ray after bounce
                    ray = Ray(hit.getPoint(), finalDirection);

                    Vector<double> emittedLight = mat.getColor().toVector() * mat.getEmissionStrengh();
                    incomingLight += emittedLight.productTermByTerm(rayColor);
                    rayColor = rayColor.productTermByTerm(mat.getColor().toVector())*(hit.getNormal()*ray.getDirection()) * 2;
                } else {
                    break;
                }
            }
            return incomingLight;
        }

};
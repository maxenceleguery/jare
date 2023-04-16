#pragma once

#include "Vector.hpp"
#include "Line.hpp"
#include "Material.hpp"
#include "Face.hpp"
#include "FaceCuda.hpp"
#include "Hit.hpp"

#include <cuda_runtime.h>
#include <curand.h>


class Ray : public Line {
    public:
        using Faces = std::vector<Face>;
    private:
        uint maxBounce = 5;

        __host__ __device__ double randomValue(uint state) const {
            state = state*747796405 + 2891336453;
            uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
            result = (result >> 22) ^ result;
            return result / 4294967295.0;
        }

        __host__ __device__ double randomValueNormalDistribution(uint state) const { 
            double theta = 2 * PI * randomValue(state);
            double rho = std::sqrt(-2*std::log(randomValue(state*state)));
            return rho*std::cos(theta);
        }

        __host__ __device__ Vector<double> randomDirection(uint state) const {
            double x;  double y;  double z;
            do {
                x = randomValueNormalDistribution(state);
                y = randomValueNormalDistribution(state*42);
                z = randomValueNormalDistribution(state*77);
            } while ( std::abs(x)<1E-5 && std::abs(y)<1E-5 && std::abs(z)<1E-5);            
            return Vector<double>(x,y,z).normalize();
        }

        __host__ __device__ int sign(double number) const {
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
            return direction - direction*2*(direction*normal);
        }

        __host__ Hit simpleTraceHost(Faces& faces) {
            Hit hit = Hit();
            for (uint i=0;i<faces.size();i++) {
                Vector<double> intersectionPoint = (faces[i]).getIntersection((Line)(*this));
                double distance = std::sqrt((intersectionPoint-point).normSquared());
                if (intersectionPoint != Vector(0.,0.,0.) && distance<hit.getDistance()) {
                    hit.setDistance(distance);
                    hit.setDistanceTraveled(distance+hit.getDistanceTraveled());
                    hit.setHasHit(true);
                    hit.setMaterial(faces[i].getMaterial());
                    hit.setNormal(faces[i].getNormalVector());
                    hit.setPoint(intersectionPoint);
                }
            }
            return hit;
        }

        __device__ Hit simpleTraceDevice(FaceCuda* faces, uint nbFaces) {
            Hit hit = Hit();
            for (uint i=0;i<nbFaces;i++) {
                Vector<double> intersectionPoint = (faces[i]).getIntersection((Line)(*this));
                double distance = std::sqrt((intersectionPoint-point).normSquared());
                if (intersectionPoint != Vector(0.,0.,0.) && distance<hit.getDistance()) {
                    hit.setDistance(distance);
                    hit.setDistanceTraveled(distance+hit.getDistanceTraveled());
                    hit.setHasHit(true);
                    hit.setMaterial(faces[i].getMaterial());
                    hit.setNormal(faces[i].getNormalVector());
                    hit.setPoint(intersectionPoint);
                }
            }
            return hit;
        }

        __host__ Pixel rayTrace1(Faces faces, Pixel& backgroundColor) {
            Hit hit = simpleTraceHost(faces);
            if (hit.getHasHit())
                return hit.getMaterial().getColor();
            else
                return backgroundColor;
        }

        __host__ Pixel rayTrace2(Faces& faces, uint state) {
            Vector<double> incomingLight = Vector<double>();
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<maxBounce;bounce++) {
                Hit hit = simpleTraceHost(faces);
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
            Pixel finalColor = Pixel(incomingLight);
            return finalColor;
        }

        __device__ Vector<double> rayTrace3(int idx, Ray ray, FaceCuda* faces, uint nbFaces) {
            Vector<double> incomingLight = Vector<double>(0.,0.,0.);
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<ray.getMaxBounce();bounce++) {
                Hit hit = ray.simpleTraceDevice(faces, nbFaces);
                if (hit.getHasHit()) {
                    Material mat = hit.getMaterial();
                    Vector<double> diffusionDir = ray.getDiffusionDirection(hit.getNormal(),idx);
                    Vector<double> specularDir = ray.getSpecularDirection(hit.getNormal());
                    Vector<double> finalDirection = diffusionDir*(1-mat.getSpecularSmoothness()) + specularDir*mat.getSpecularSmoothness();
                    ray = Ray(hit.getPoint(),diffusionDir);
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
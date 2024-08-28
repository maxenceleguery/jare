#pragma once

#include "Vector.hpp"
#include "Material.hpp"

#include <cuda_runtime.h>

class Hit {
    private:
        Material mat = Material();
        Vector<double> point = Vector<double>();
        Vector<double> normal = Vector<double>();
        double distance = INFINITY;
        double distanceTraveled = 0.;
        double firstDistance = -1.;
        bool hasHit = false;
        
    public:
        __host__ __device__ Hit() {};
        __host__ __device__ ~Hit() {};

        __host__ __device__ void update(const Hit& hit) {
            if (hit.getHasHit() && hit.getDistance() < distance) {
                if (firstDistance < 0)
                    firstDistance = hit.distance;
                setDistance(hit.distance);
                setHasHit(true);
                setMaterial(hit.mat);
                setNormal(hit.normal);
                setPoint(hit.point);
            }
        }
        
        // getters
        __host__ __device__ Material getMaterial() const {
            return mat;
        }
        
        __host__ __device__ Vector<double> getPoint() const {
            return point;
        }
        
        __host__ __device__ Vector<double> getNormal() const {
            return normal;
        }
        
        __host__ __device__ double getDistance() const {
            return distance;
        }
        
        __host__ __device__ double getFirstDistance() const {
            return firstDistance;
        }

        __host__ __device__ double getDistanceTraveled() const {
            return distanceTraveled;
        }
        
        __host__ __device__ bool getHasHit() const {
            return hasHit;
        }
        
        // setters
        __host__ __device__ void setMaterial(const Material& m) {
            mat = m;
        }
        
        __host__ __device__ void setPoint(const Vector<double>& p) {
            point = p;
        }
        
        __host__ __device__ void setNormal(const Vector<double>& n) {
            normal = n;
        }
        
        __host__ __device__ void setDistance(const double& d) {
            distance = d;
            distanceTraveled += d;
        }

        __host__ __device__ void setFirstDistance(const double& d) {
            firstDistance = d;
        }
        
        __host__ __device__ void setHasHit(const bool& h) {
            hasHit = h;
        }
};


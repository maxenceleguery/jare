#pragma once

#include "../Vector.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#define PI 3.14159

class RandomGenerator {
    private:
        
    public:
         __host__ __device__ RandomGenerator() {};
         __host__ __device__ ~RandomGenerator() {};

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
};


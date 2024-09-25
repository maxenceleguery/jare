#pragma once

#include "../Vector.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#define PI 3.141592653589f

class RandomGenerator {
    private:
        uint inner_state = 1;
    public:
         __host__ __device__ RandomGenerator() {};
         __host__ __device__ ~RandomGenerator() {};

        __host__ __device__ float randomValue(uint state) {
            state = state*747796405 + 2891336453*inner_state;
            inner_state = state;
            uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
            result = (result*inner_state >> 22) ^ result;
            return result / 4294967295.f;
        }

        __host__ __device__ float randomValueNormalDistribution(const uint state) { 
            float theta = 2 * PI * randomValue(state);
            const float rho = std::sqrt(-2*std::log(randomValue(state*state)));
            return rho*std::cos(theta);
        }

        __host__ __device__ Vector<float> randomDirection(const uint state) {
            float x;  float y;  float z;
            do {
                x = randomValueNormalDistribution(state);
                y = randomValueNormalDistribution(state*42);
                z = randomValueNormalDistribution(state*77);
            } while ( std::abs(x)<1E-5 && std::abs(y)<1E-5 && std::abs(z)<1E-5);            
            return Vector<float>(x,y,z).normalize();
        }
};


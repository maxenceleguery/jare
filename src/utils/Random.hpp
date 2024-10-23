#pragma once

#include "../Vector.hpp"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <exception>

#define PI 3.141592653589f

class RandomGenerator {
    private:
        unsigned long inner_state = 0;
    public:
        __host__ __device__ RandomGenerator() {};
        __host__ __device__ RandomGenerator(const unsigned long seed) : inner_state(seed) {};
        __host__ __device__ ~RandomGenerator() {};

        __host__ __device__ unsigned long getSeed() const {
            return inner_state;
        }

        __host__ __device__ void updateSeed(const unsigned long seed) {
            inner_state += seed;
        }

        __host__ __device__ float randomValue(uint& state) {
            //if (inner_state == 0) inner_state = state + 1;

            /*
            inner_state ^= (inner_state << 13);
            inner_state ^= (inner_state >> 17);
            inner_state ^= (inner_state << 5);
            return static_cast<float>(state) / static_cast<float>(UINT32_MAX);
            */

            /*
            const unsigned long a = 1664525; // multiplier
            const unsigned long c = 1013904223; // increment
            const unsigned long m = 4294967296; // 2^32
            inner_state = (a * state + c) % m;
            return static_cast<float>(state) / (m - 1);
            */

            state = state*747796405 + 2891336453;//*inner_state;
            //inner_state = state;
            uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
            result = (result >> 22) ^ result;
            return result / 4294967295.f;

            /*
            unsigned int t;
            t = (state->v[0] ^ (state->v[0] >> 2));
            state->v[0] = state->v[1];
            state->v[1] = state->v[2];
            state->v[2] = state->v[3];
            state->v[3] = state->v[4];
            state->v[4] = (state->v[4] ^ (state->v[4] <<4)) ^ (t ^ (t << 1));
            state->d += 362437;
            unsigned int x = state->v[4] + state->d;
            printf("%f\n", x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f));
            return x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);*/
        }

        __host__ __device__ float randomValueNormalDistribution(uint& state) { 
            float theta = 2 * PI * randomValue(state);
            const float rho = std::sqrt(-2*std::log(randomValue(state)));
            return rho*std::cos(theta);
        }

        __host__ __device__ Vector<float> randomDirection(uint& state) {
            float x;  float y;  float z;
            do {
                x = randomValueNormalDistribution(state);
                y = randomValueNormalDistribution(state);
                z = randomValueNormalDistribution(state);
            } while ( std::abs(x)<1E-5f && std::abs(y)<1E-5f && std::abs(z)<1E-5f);            
            return Vector<float>(x,y,z).normalize();
        }


        __host__ __device__ float randomValue() {
            const unsigned long a = 1664525; // multiplier
            const unsigned long c = 1013904223; // increment
            const unsigned long m = 4294967296; // 2^32
            inner_state = (a * inner_state + c) % m;
            return static_cast<float>(inner_state) / (m - 1);
        }

        __host__ __device__ float randomValueNormalDistribution() { 
            float theta = 2 * PI * randomValue();
            const float rho = std::sqrt(-2*std::log(randomValue()));
            return rho*std::cos(theta);
        }

        __host__ __device__ Vector<float> randomDirection() {
            float x;  float y;  float z;
            do {
                x = randomValueNormalDistribution();
                y = randomValueNormalDistribution();
                z = randomValueNormalDistribution();
            } while ( std::abs(x)<1E-5f && std::abs(y)<1E-5f && std::abs(z)<1E-5f);            
            return Vector<float>(x,y,z).normalize();
        }
};

class RandomInterface {
    public:
        __host__ __device__ float randomValue(uint state) {
            #ifdef  __CUDA_ARCH__
                state *= (clock64() % 10) ^ 156;
            #else
                state = rand();
            #endif

            state = state*747796405 + 2891336453;
            uint result = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
            result = (result >> 22) ^ result;
            return result / 4294967295.0f;
        }

        __host__ __device__ float randomValueNormalDistribution(uint state) { 
            float theta = 2 * PI * randomValue(state);
            float rho = std::sqrt(-2*std::log(randomValue(state*state)));
            return rho*std::cos(theta);
        }

        __host__ __device__ Vector<float> randomDirection(uint state) {
            float x;  float y;  float z;
            do {
                x = randomValueNormalDistribution(state);
                state = randomValue(state)*100000000000000;
                y = randomValueNormalDistribution(state);
                state = randomValue(state)*100000000000000;
                z = randomValueNormalDistribution(state);
            } while ( std::abs(x)<1E-5f && std::abs(y)<1E-5f && std::abs(z)<1E-5f);            
            return Vector<float>(x,y,z).normalize();
        }
};

class Khi2Error : public std::runtime_error {
    private:
        std::string what_message;
    public:
        Khi2Error() : std::runtime_error("Khi² exception") {};
        Khi2Error(const double value, const double threshold) : std::runtime_error("Khi² exception") {
            what_message = "Khi² exception : " + std::to_string(value) + " > " + std::to_string(threshold);
        };
        virtual ~Khi2Error() noexcept = default;
    public:
        const char* what() const noexcept override {
            return what_message.c_str();
        }
};
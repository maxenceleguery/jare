#pragma once
#include <iostream>
#include "Pixel.hpp"

#include <cuda_runtime.h>

class Material {
    private:
        Pixel emissionColor;
        double diffusion;
        double reflexion;
        double emissionStrengh = 0.;
    public:
        __host__ __device__ Material() : emissionColor(Pixel(0,0,0)), diffusion(0), reflexion(0) {};
        __host__ __device__ Material(Pixel color0) : emissionColor(color0), diffusion(0), reflexion(0) {};
        __host__ __device__ ~Material() {};

    __host__ __device__ Pixel getColor() const {
        return emissionColor;
    }
    __host__ __device__ void setColor(const Pixel color) {
        emissionColor=color;
    }

    __host__ __device__ double getDiffusion() const {
        return diffusion;
    }
    __host__ __device__ void setDiffusion(const double d) {
        diffusion=d;
    }

    __host__ __device__ double getReflexion() const {
        return reflexion;
    }
    __host__ __device__ void setReflexion(const double r) {
        reflexion=r;
    }

    __host__ __device__ double getEmissionStrengh() const {
        return emissionStrengh;
    }
    __host__ __device__ void setEmissionStrengh(const double s) {
        emissionStrengh=s;
    }
};

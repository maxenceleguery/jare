#pragma once
#include <iostream>
#include "Pixel.hpp"

#include <cuda_runtime.h>

class Material {
    private:
        Pixel emissionColor;
        double diffusion = 0.;
        double specularSmoothness = 0.;
        double emissionStrengh = 0.;
    public:
        __host__ __device__ Material() : emissionColor(Pixel(0,0,0)), diffusion(0), specularSmoothness(0) {};
        __host__ __device__ Material(Pixel color0) : emissionColor(color0), diffusion(0), specularSmoothness(0) {};
        __host__ __device__ Material(Pixel color0, double diffusion) : emissionColor(color0), diffusion(diffusion), specularSmoothness(0) {};
        __host__ __device__ Material(Pixel color0, double diffusion, double specularSmoothness) : emissionColor(color0), diffusion(diffusion), specularSmoothness(specularSmoothness) {};
        __host__ __device__ Material(Pixel color0, double diffusion, double specularSmoothness, double emissionStrengh) : emissionColor(color0), diffusion(diffusion), specularSmoothness(specularSmoothness), emissionStrengh(emissionStrengh) {};
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

        __host__ __device__ double getSpecularSmoothness() const {
            return specularSmoothness;
        }
        __host__ __device__ void setSpecularSmoothness(const double ss) {
            specularSmoothness=ss;
        }

        __host__ __device__ double getEmissionStrengh() const {
            return emissionStrengh;
        }
        __host__ __device__ void setEmissionStrengh(const double s) {
            emissionStrengh=s;
        }
};

namespace Materials {
    const Material LIGHT = Material(Pixel(255,255,255), 0, 0, 1.);
    const Material MIRROR = Material(Pixel(255,255,255), 0, 1, 0);
}

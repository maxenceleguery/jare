#pragma once
#include <iostream>
#include "Pixel.hpp"

#include <cuda_runtime.h>

enum MaterialType {
    DEFAULT,
    MIRROR,
    LIGHT,
};

class Material {
    private:
        Pixel emissionColor;
        Pixel specularColor;
        float diffusion;
        float specularSmoothness;
        float specularProb;
        float emissionStrengh;
    public:
        __host__ __device__ Material() : emissionColor(Pixel(0,0,0)), specularColor(Pixel(0,0,0)), diffusion(0), specularSmoothness(0), specularProb(0), emissionStrengh(0) {};
        __host__ __device__ Material(Pixel color0) : emissionColor(color0), specularColor(color0), diffusion(0), specularSmoothness(0), specularProb(0), emissionStrengh(0) {};
        __host__ __device__ Material(Pixel color0, MaterialType mat_type) : emissionColor(color0), specularColor(color0) {
            switch(mat_type) {
                case MIRROR:
                    diffusion = 0;
                    specularSmoothness = 1;
                    specularProb = 0;
                    emissionStrengh = 0;
                    break;

                case LIGHT:
                    diffusion = 0;
                    specularSmoothness = 0;
                    specularProb = 0;
                    emissionStrengh = 1;
                    break;

                default:
                    diffusion = 0;
                    specularSmoothness = 0;
                    specularProb = 0;
                    emissionStrengh = 0;
            }
        };
        __host__ __device__ ~Material() {};

        __host__ __device__ Pixel getColor() const {
            return emissionColor;
        }
        __host__ __device__ void setColor(const Pixel color) {
            emissionColor=color;
        }

        __host__ __device__ Pixel getSpecularColor() const {
            return specularColor;
        }
        __host__ __device__ void setSpecularColor(const Pixel color) {
            specularColor=color;
        }

        __host__ __device__ float getDiffusion() const {
            return diffusion;
        }
        __host__ __device__ void setDiffusion(const float d) {
            diffusion=d;
        }

        __host__ __device__ float getSpecularSmoothness() const {
            return specularSmoothness;
        }
        __host__ __device__ void setSpecularSmoothness(const float ss) {
            specularSmoothness=ss;
        }

        __host__ __device__ float getSpecularProb() const {
            return specularProb;
        }
        __host__ __device__ void setSpecularProb(const float sp) {
            specularProb=sp;
        }

        __host__ __device__ float getEmissionStrengh() const {
            return emissionStrengh;
        }
        __host__ __device__ void setEmissionStrengh(const float s) {
            emissionStrengh=s;
        }
};

namespace Materials {
    const Material LIGHT = Material(Pixel(255,255,255), MaterialType::LIGHT);
    const Material MIRROR = Material(Pixel(255,255,255), MaterialType::MIRROR);
}

#pragma once
#include <iostream>
#include "Pixel.hpp"
#include "utils/Random.hpp"

#include <cuda_runtime.h>

enum MaterialType {
    DEFAULT,
    MIRROR,
    LIGHT,
    GLASS,
    WATER,
};

struct RayInfo {
    float env_refractive_index;
    bool isInside;
};


class Material : public RandomInterface {
    private:
        Pixel emissionColor;
        float emissionStrengh = 0;

        Pixel specularColor;
        float specularSmoothness = 0;
        float specularProb = 0;

        bool isTransparent = false;
        float transparency = 0;
        float refractive_index = 1.000293;

        __host__ __device__ int sign(const float number) const {
            return number<0 ? -1 : 1;
        }

    public:
        __host__ __device__ Material() : emissionColor(Pixel(0,0,0)), specularColor(Pixel(0,0,0)) {};
        __host__ __device__ Material(Pixel color0) : emissionColor(color0), specularColor(color0) {};
        __host__ __device__ Material(Pixel color0, MaterialType mat_type) : emissionColor(color0), specularColor(color0) {
            switch(mat_type) {
                case MIRROR:
                    specularSmoothness = 1;
                    specularProb = 1;
                    break;

                case LIGHT:
                    emissionStrengh = 1;
                    break;

                case GLASS:
                    isTransparent = true;
                    transparency = 0.9;
                    refractive_index = 1.52;
                    specularSmoothness = 1;
                    break;

                case WATER:
                    isTransparent = true;
                    transparency = 1;
                    refractive_index = 1.333;
                    specularSmoothness = 1;
                    break;

                case DEFAULT:
                    break;
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

        __host__ __device__ Vector<float> getDiffusionDirection(const Vector<float>& ray_direction, Vector<float> normal, uint state) {
            Vector<float> dir = randomDirection(state);
            //normal *= sign(-ray_direction*normal);
            return dir*sign(dir*normal);
        }

        // Specular or reflexion direction
        __host__ __device__ Vector<float> getSpecularDirection(const Vector<float>& ray_direction, Vector<float> normal) const {
            //normal *= sign(-ray_direction*normal);
            return (ray_direction - normal*2*(ray_direction*normal)).normalize();
        }
        
        // Refraction direction
        __host__ __device__ Vector<float> getRefractionDirection(const Vector<float>& ray_direction, Vector<float> normal, RayInfo& ray_info) const {
            normal *= sign(-ray_direction*normal);
            const float r = ray_info.isInside ? refractive_index/ray_info.env_refractive_index : ray_info.env_refractive_index/refractive_index;
            const float cos_theta_1 = -normal*ray_direction;
            const float sin_theta_2_squared = r*r*(1-cos_theta_1*cos_theta_1);

            ray_info.isInside = !ray_info.isInside;

            if (sin_theta_2_squared <= 1) {
                return (ray_direction*r + normal*(r*cos_theta_1 - std::sqrt(1-sin_theta_2_squared))).normalize(); // Snell-Descartes
            } else { // Total reflexion
                return getSpecularDirection(ray_direction, normal);
            }
        }

        __host__ __device__ Vector<float> trace(const Vector<float>& ray_direction, Vector<float> normal, RayInfo& ray_info, uint state) {
            // Diffusion
            Vector<float> finalDirection = getDiffusionDirection(ray_direction, normal, state);
            // Specular
            if (specularProb >= randomValue(state)) {
                Vector<float> specularDir = getSpecularDirection(ray_direction, normal);
                finalDirection = finalDirection.lerp(specularDir, specularSmoothness).normalize();
            }
            // Refraction
            if (transparency >= randomValue(state)) {
                Vector<float> refractionDir = getRefractionDirection(ray_direction, normal, ray_info);
                finalDirection = refractionDir;
            }
            return finalDirection;
        }

        __host__ __device__ void shade(Vector<float>* incomingLight, Vector<float>* rayColor, const Vector<float>& ray_direction, Vector<float> normal, const float dist) const {
            Vector<float> emittedLight = emissionColor.toVector() * emissionStrengh;
            *incomingLight += emittedLight.productTermByTerm(*rayColor);//* 5./(dist*dist);
            //incomingLight->clamp(0.f, 1.f);
            *rayColor = rayColor->productTermByTerm(emissionColor.toVector()*(normal*ray_direction) * 2);
        }
};

namespace Materials {
    const Material LIGHT = Material(Pixel(255,255,255), MaterialType::LIGHT);
    const Material MIRROR = Material(Pixel(255,255,255), MaterialType::MIRROR);
}

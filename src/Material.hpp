#pragma once
#include <iostream>
#include "Pixel.hpp"

class Material {
    private:
        Pixel emissionColor;
        double diffusion;
        double reflexion;
        double emissionStrengh = 0.;
    public:
        Material() : emissionColor(Pixel(0,0,0)), diffusion(0), reflexion(0) {};
        Material(Pixel color0) : emissionColor(color0), diffusion(0), reflexion(0) {};
        ~Material() {};

    Pixel getColor() const {
        return emissionColor;
    }
    void setColor(const Pixel color) {
        emissionColor=color;
    }

    double getDiffusion() const {
        return diffusion;
    }
    void setDiffusion(const double d) {
        diffusion=d;
    }

    double getReflexion() const {
        return reflexion;
    }
    void setReflexion(const double r) {
        reflexion=r;
    }

    double getEmissionStrengh() const {
        return emissionStrengh;
    }
    void setEmissionStrengh(const double s) {
        emissionStrengh=s;
    }
};

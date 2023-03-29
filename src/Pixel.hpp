#pragma once
#include <cstdint>
#include <iostream>
#include <fstream>

class Pixel {
    private:
        uint r;
        uint g;
        uint b;
        uint a;

    public:
        Pixel() : r(0), g(0), b(0), a(0) {};
        Pixel(uint r0, uint g0, uint b0) : r(r0), g(g0), b(b0), a(0) {
            if (r0 > 255 || g0 > 255 || b0 > 255)
                throw std::invalid_argument("Pixel's values must be between 0 and 255");
        };
        Pixel(uint r0, uint g0, uint b0, uint a0) : r(r0), g(g0), b(b0), a(a0) {
            if (r0 > 255 || g0 > 255 || b0 > 255 || a0 > 255)
                throw std::invalid_argument("Pixel's values must be between 0 and 255");
        };
        ~Pixel();

        void renderPixel(std::ofstream& imageFlux) const {
            imageFlux << r << " " << g << " " << b << " ";
        }
};


Pixel::~Pixel() {
}

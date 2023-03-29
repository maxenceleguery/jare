#pragma once
#include <cstdint>
#include <iostream>
#include <fstream>

class Pixel {
    private:
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;

    public:
        Pixel() : r(0), g(0), b(0), a(0) {};
        Pixel(uint8_t r0, uint8_t g0, uint8_t b0) : r(r0), g(g0), b(b0), a(0) {
            /*if (r0 > 255 || g0 > 255 || b0 > 255)
                throw std::invalid_argument("Pixel's values must be between 0 and 255");*/
        };
        Pixel(uint8_t r0, uint8_t g0, uint8_t b0, uint8_t a0) : r(r0), g(g0), b(b0), a(a0) {
            /*if (r0 > 255 || g0 > 255 || b0 > 255 || a0 > 255)
                throw std::invalid_argument("Pixel's values must be between 0 and 255");*/
        };
        ~Pixel();

        void renderPixel(std::ofstream& imageFlux) const {
            imageFlux << r << " " << g << " " << b << " ";
        }

        uint8_t getR() const {
            return r;
        }

        uint8_t getG() const {
            return g;
        }

        uint8_t getB() const {
            return b;
        }
};


Pixel::~Pixel() {
}

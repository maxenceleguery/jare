#ifndef PIXEL_HPP
#define PIXEL_HPP

#include "Vector.hpp"
#include <cstdint>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

class Pixel {
    private:
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;

        template<typename T, typename U>
        __host__ __device__ T min(T number1, U number2) {
            if (number1 < number2)
                return number1;
            else
                return number2;
        }

    public:
        __host__ __device__ Pixel() : r(0), g(0), b(0), a(0) {};
        __host__ __device__ Pixel(uint8_t r0, uint8_t g0, uint8_t b0) : r(r0), g(g0), b(b0), a(0) {
            /*if (r0 > 255 || g0 > 255 || b0 > 255)
                throw std::invalid_argument("Pixel's values must be between 0 and 255");*/
        };
        __host__ __device__ Pixel(uint8_t r0, uint8_t g0, uint8_t b0, uint8_t a0) : r(r0), g(g0), b(b0), a(a0) {
            /*if (r0 > 255 || g0 > 255 || b0 > 255 || a0 > 255)
                throw std::invalid_argument("Pixel's values must be between 0 and 255");*/
        };
        // Vector values has to be between 0 and 1
        __host__ __device__ Pixel(const Vector<float>& vec0) : r(min(255,vec0.getX()*255)), g(min(255,vec0.getY()*255)), b(min(255,vec0.getZ()*255)) {};
        __host__ __device__ ~Pixel() {};

        __host__ void renderPixel(std::ofstream& imageFlux) const {
            imageFlux << r << " " << g << " " << b << " ";
        }

        __host__ __device__ uint8_t getR() const {
            return r;
        }

        __host__ __device__ uint8_t getG() const {
            return g;
        }

        __host__ __device__ uint8_t getB() const {
            return b;
        }

        __host__ __device__ void setR(const uint8_t r0) {
            r=r0;
        }

        __host__ __device__ void setG(const uint8_t g0) {
            g=g0;
        } 

        __host__ __device__ void setB(const uint8_t b0) {
            b=b0;
        } 

        __host__ __device__ Pixel& operator*= (const Pixel& p) {
            this->r *= p.r;
            this->g *= p.g;
            this->b *= p.b;
            return *this;
        }

        __host__ __device__ Pixel operator+ (const Pixel& other) const {
            return Pixel(r+other.r,g+other.g,b+other.b);
        }

        __host__ __device__ Pixel operator- (const Pixel& other) const {
            return Pixel(r-other.r,g-other.g,b-other.b);
        }

        __host__ __device__ Pixel operator* (const float number) const {
            return Pixel(r*number,g*number,b*number);
        }

        __host__ __device__ Pixel operator/ (const float number) const {
            return Pixel(r/number,g/number,b/number);
        }

        __host__ __device__ Vector<float> toVector() const {
            return Vector<float>(r,g,b)/255.f;
        }

        __host__ __device__ static float sRGBtoLin(const float colorChannel) {
            if (colorChannel <= 0.04045f) {
                return colorChannel / 12.92f;
            } else {
                return std::pow((( colorChannel + 0.055f)/1.055f), 2.4f);
            }
        }

        __host__ __device__ float getLuminance() const {
            return 0.2126f*sRGBtoLin(r/255.f) + 0.7152f*sRGBtoLin(g/255.f) + 0.0722f*sRGBtoLin(b/255.f);
        }

        __host__ __device__ float getPerceivedLightness() const {
            const float Y = getLuminance();
            if (Y <= 216.f/24389.f) {       // The CIE standard states 0.008856 but 216/24389 is the intent for 0.008856451679036
                return Y * (24389.f/27.f);  // The CIE standard states 903.3, but 24389/27 is the intent, making 903.296296296296296
            } else {
                return pow(Y,(1/3)) * 116 - 16;
            }
        }
};

namespace Colors {
    const Pixel RED = Pixel(255,0,0);
    const Pixel GREEN = Pixel(0,255,0);
    const Pixel BLUE = Pixel(0,0,255);
    const Pixel YELLOW = Pixel(255,255,0);
    const Pixel CYAN = Pixel(0,255,255);
    const Pixel MAGENTA = Pixel(255,0,255);
    const Pixel BLACK = Pixel(0,0,0);
    const Pixel WHITE = Pixel(255,255,255);
    
    const Pixel BROWN = Pixel(165,42,42);
    const Pixel SKY_BLUE = Pixel(87, 232, 229);
}

#endif
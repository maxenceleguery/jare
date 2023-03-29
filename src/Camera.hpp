#pragma once
#include "Vector.hpp"
#include "Pixel.hpp"
#include <vector>
#include <iostream>
#include <fstream>

class Camera {
    private:
        Vector<double> position;
        uint width;
        uint height;
        double fov;
        std::vector<Pixel> pixels;

    public:
        Camera();
        Camera(Vector<double> pos, uint width0, uint height0) : position(pos), width(width0), height(height0), pixels(width0*height0) {};
        ~Camera();

        uint getWidth() const {
            return width;
        }

        uint getHeight() const {
            return height;
        }

        Pixel getPixel(uint index) const {
            return pixels[index];
        }

        void setPixel(uint index, Pixel& color) {
            pixels[index] = color;
        }

        void renderImage() const {
            std::ofstream imageFlux("./image.ppm");
            if (imageFlux) {
                imageFlux << "P3\n" << width << " " << height << "\n255\n"; 
        
                for(uint l = 0; l < height; ++l) 
                    for(uint c = 0; c < width; ++c)
                        pixels[l*width+c].renderPixel(imageFlux);
            }
            imageFlux.close();
        }
};

Camera::~Camera() {

}
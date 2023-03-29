#pragma once
#include "Vector.hpp"
#include "Pixel.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <png.h>

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

        void write_png_file(const char* filename, uint8_t* image_data) {
            FILE *fp = fopen(filename, "wb");
            png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
            png_infop info_ptr = png_create_info_struct(png_ptr);

            png_init_io(png_ptr, fp);

            png_set_IHDR(png_ptr, info_ptr, width, height,
                        8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

            png_bytep row_pointers[height];
            for(uint y = 0; y < height; y++)
                row_pointers[y] = &image_data[y * width * 3];

            png_set_rows(png_ptr, info_ptr, row_pointers);
            png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

            fclose(fp);
        }

        void renderImage() {
            uint8_t* image_data = new uint8_t[width * height * 3];
            
            for(uint i = 0; i < width * height; ++i) {
                uint8_t r = pixels[i].getR();
                uint8_t g = pixels[i].getG();
                uint8_t b = pixels[i].getB();

                image_data[i * 3] = r;
                image_data[i * 3 + 1] = g;
                image_data[i * 3 + 2] = b;
            }

            write_png_file("./image.png", image_data);

            delete[] image_data;
        }
};

Camera::~Camera() {

}
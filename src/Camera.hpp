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
        Vector<double> orientation;
        uint width;
        uint height;
        double capteurWidth;
        double capteurHeight;
        double fov = 0.01;
        std::vector<Pixel> pixels;

    public:
        Camera();
        Camera(Vector<double> pos, uint width0, uint height0) : position(pos), orientation(Vector<double>()), width(width0), height(height0), pixels(width0*height0) {
            capteurWidth = (0.005*width0)/(1.*height0);
            capteurHeight = 0.005;
        };
        Camera(Vector<double> pos, Vector<double> ori, uint width0, uint height0) : position(pos), orientation(ori), width(width0), height(height0), pixels(width0*height0) {
            capteurWidth = (0.005*width0)/(1.*height0);
            capteurHeight = 0.005;
        };
        ~Camera();

        uint getWidth() const {
            return width;
        }

        uint getHeight() const {
            return height;
        }

        Vector<double> getPixelCoordOnCapt(uint w, uint h) const {
            double W = (1.*w - width/2.)*(1.*capteurWidth/width);
            double H = (-1.*h + height/2.)*(1.*capteurHeight/height);
            return Vector<double>(0.,0.,1.)*H + (Vector<double>(0.,0.,1.).crossProduct(orientation).normalize())*W;
        }

        Pixel getPixel(uint index) const {
            return pixels[index];
        }

        void setPixel(uint index, const Pixel& color) {
            pixels[index] = color;
        }

        Vector<double> getPosition() const {
            return position;
        }

        void setPosition(const Vector<double>& pos) {
            position=pos;
        }

        Vector<double> getOrientation() const {
            return orientation;
        }

        void setOrientation(Vector<double>& ori) {
            orientation=ori;
        }

        double getFov() const {
            return fov;
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

        void renderImage(const char* filename) {
            uint8_t* image_data = new uint8_t[width * height * 3];
            
            for(uint i = 0; i < width * height; ++i) {
                uint8_t r = pixels[i].getR();
                uint8_t g = pixels[i].getG();
                uint8_t b = pixels[i].getB();

                image_data[i * 3] = r;
                image_data[i * 3 + 1] = g;
                image_data[i * 3 + 2] = b;
            }

            write_png_file(filename, image_data);

            delete[] image_data;
        }
};

Camera::~Camera() {

}
#pragma once
#include "Vector.hpp"
#include "Pixel.hpp"
#include "Matrix.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <png.h>

#define ROT_RIGHT 2000
#define ROT_FRONT 2001
#define ROT_UP 2002

class Camera {
    private:
        Vector<double> position;

        Vector<double> vectRight;
        Vector<double> vectFront;
        Vector<double> vectUp;

        uint width;
        uint height;
        double capteurWidth;
        double capteurHeight;
        double fov = 0.01;
        double gamma = 2.0;
        std::vector<Pixel> pixels;

    public:
        Camera(){};
        Camera(Vector<double> pos, uint width0, uint height0) : position(pos), vectFront(Vector<double>(0,1,0)), vectUp(Vector<double>(0,0,1)), vectRight(Vector<double>(1,0,0).crossProduct(Vector<double>(0,0,1)).normalize()), width(width0), height(height0), pixels(width0*height0) {
            capteurWidth = (0.005*width0)/(1.*height0);
            capteurHeight = 0.005;
        };
        Camera(Vector<double> pos, Vector<double> front, uint width0, uint height0) : position(pos), vectFront(front.normalize()), vectUp(Vector<double>(0,0,1)), vectRight(front.crossProduct(Vector<double>(0,0,1)).normalize()), width(width0), height(height0), pixels(width0*height0) {
            capteurWidth = (0.005*width0)/(1.*height0);
            capteurHeight = 0.005;
        };
        ~Camera() {};

        inline uint getWidth() const {
            return width;
        }

        inline uint getHeight() const {
            return height;
        }

        double getGamma() const {
            return gamma;
        }

        void setGamma(double g) {
            gamma = g;
        }

        Vector<double> getPixelCoordOnCapt(double w, double h) const {
            double W = (1.*w - width/2.)*(1.*capteurWidth/width);
            double H = (-1.*h + height/2.)*(1.*capteurHeight/height);
            return vectUp*H + vectRight*W;
        }

        inline Pixel getPixel(uint index) const {
            return pixels[index];
        }

        void setPixel(uint index, const Pixel& color) {
            pixels[index] = color;
        }

        inline Vector<double> getPosition() const {
            return position;
        }

        void setPosition(const Vector<double>& pos) {
            position=pos;
        }

        void move(const Vector<double>& offset) {
            position += offset;
        }

        inline Vector<double> getVectFront() const {
            return vectFront;
        }

        void setVectFront(Vector<double>& ori) {
            vectFront=ori;
        }

        inline double getFov() const {
            return fov;
        }

        void rotate(double angle, uint axis) {
            Vector<double> direction;
            switch (axis) {
            case ROT_FRONT:
                direction=vectFront.normalize();
                break;
            case ROT_RIGHT:
                direction=vectRight.normalize();
                break;
            case ROT_UP:
                direction=vectUp.normalize();
                break;
            
            default:
                std::cout << "Wrong axis provided" << std::endl;
                return;
            }
            double ux = direction.getX();
            double uy = direction.getY();
            double uz = direction.getZ();
            Matrix<double> P = Matrix<double>(ux*ux,ux*uy,ux*uz,ux*uy,uy*uy,uy*uz,ux*uz,uy*uz,uz*uz);
            Matrix<double> I = Matrix<double>(1.,MATRIX_EYE);
            Matrix<double> Q = Matrix<double>(0,-uz,uy,uz,0,-ux,-uy,ux,0);

            Matrix<double> R = P + (I-P)*std::cos(angle) + Q*std::sin(angle);
            vectFront=(R*vectFront).normalize();
            vectRight=(R*vectRight).normalize();
            vectUp=(R*vectUp).normalize();
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
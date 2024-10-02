#pragma once

#include "Vector.hpp"
#include "Pixel.hpp"
#include "Matrix.hpp"
#include "Ray.hpp"
#include "utils/Array.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <png.h>

#include <cuda_runtime.h>

#define ROT_RIGHT 2000
#define ROT_FRONT 2001
#define ROT_UP 2002

class Camera : public CudaReady {
    private:
        Vector<float> position;

        Vector<float> vectRight;
        Vector<float> vectFront;
        Vector<float> vectUp;

        uint width;
        uint height;
        float capteurWidth;
        float capteurHeight;
        float fov = 0.01;
        float gamma = 2.0;

        float current_fps = 0.;
        uint num_images_rendered = 0;

        Array<Pixel> pixels;

    public:
        uint threadsByRay = 1;

        __host__ Camera() {};
        __host__ Camera(Vector<float> pos, uint width0, uint height0) : position(pos), vectFront(Vector<float>(0,1,0)), vectUp(Vector<float>(0,0,1)), vectRight(Vector<float>(1,0,0).crossProduct(Vector<float>(0,0,1)).normalize()), width(width0), height(height0) {
            pixels =  Array<Pixel>(width0*height0*threadsByRay);
            capteurWidth = (0.005*width0)/(1.*height0);
            capteurHeight = 0.005;
        };
        __host__ Camera(Vector<float> pos, Vector<float> front, uint width0, uint height0) : position(pos), vectFront(front.normalize()), vectUp(Vector<float>(0,0,1)), vectRight(front.crossProduct(Vector<float>(0,0,1)).normalize()), width(width0), height(height0), pixels(width0*height0*threadsByRay) {
            pixels =  Array<Pixel>(width0*height0*threadsByRay);
            capteurWidth = (0.005*width0)/(1.*height0);
            capteurHeight = 0.005;
        };

        __host__ void cuda() override {
            pixels.cuda();
        }

        __host__ void cpu() override {
            pixels.cpu();
        }

        __host__ void sync_to_cpu() override {
            pixels.sync_to_cpu();
        }

        __host__ void free() override {
            pixels.free();
        }

        __host__ __device__ uint getWidth() const {
            return width;
        }

        __host__ __device__ uint getHeight() const {
            return height;
        }

        __host__ __device__ float getGamma() const {
            return gamma;
        }

        __host__ __device__ void setGamma(const float g) {
            gamma = g;
        }

        __host__ float getCurrentFPS() const {
            return current_fps;
        }

        __host__ void setCurrentFPS(const float fps) {
            current_fps = fps;
        }

        __host__ __device__ Vector<float> getPixelCoordOnCapt(const float w, const float h) const {
            const float W = (1.*w - width/2.)*(1.*capteurWidth/width);
            const float H = (-1.*h + height/2.)*(1.*capteurHeight/height);
            return vectUp*H + vectRight*W;
        }

        __host__ Pixel getPixelCPU(const uint index) const {
            return pixels.getValueFromCPU(index);
        }

        __host__ __device__ Pixel getPixel(const uint index) const {
            return pixels[index];
        }

        __host__ __device__ void setPixel(const uint index, const Pixel& color) {
            pixels[index] = color;
        }

        __host__ __device__ void updatePixel(const uint index, const Pixel& color) {
            if (index == 0) num_images_rendered++;
            if ((pixels[index].toVector() - color.toVector()).normSquared() > 0.5) {
                num_images_rendered = 1;
            }
            float weight = 1.f / (num_images_rendered);
            pixels[index] = pixels[index]*(1-weight) + color*weight;
        }

        __host__ __device__ Vector<float> getPosition() const {
            return position;
        }

        __host__ __device__ void setPosition(const Vector<float>& pos) {
            position=pos;
        }

        __host__ __device__ void move(const Vector<float>& offset) {
            position += offset;
        }

        __host__ __device__ Vector<float> getVectFront() const {
            return vectFront;
        }

        __host__ __device__ void setVectFront(Vector<float>& ori) {
            vectFront=ori;
        }

        __host__ __device__ float getFov() const {
            return fov;
        }

        __host__ __device__ void rotate(const float angle, const uint axis) {
            Vector<float> direction;
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
                //std::cout << "Wrong axis provided" << std::endl;
                return;
            }
            float ux = direction.getX();
            float uy = direction.getY();
            float uz = direction.getZ();
            Matrix<float> P = Matrix<float>(ux*ux,ux*uy,ux*uz,ux*uy,uy*uy,uy*uz,ux*uz,uy*uz,uz*uz);
            Matrix<float> I = Matrix<float>(1.,MATRIX_EYE);
            Matrix<float> Q = Matrix<float>(0,-uz,uy,uz,0,-ux,-uy,ux,0);

            Matrix<float> R = P + (I-P)*std::cos(angle) + Q*std::sin(angle);
            vectFront=(R*vectFront).normalize();
            vectRight=(R*vectRight).normalize();
            vectUp=(R*vectUp).normalize();
        }

        __host__ __device__ Ray generate_ray(const uint w, const uint h) const {
            return Ray(position, (vectFront*fov+getPixelCoordOnCapt(w,h)).normalize());
        }

        __host__ void write_png_file(const char* filename, uint8_t* image_data) {
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

        __host__ void renderImage(const char* filename) {
            uint8_t* image_data = new uint8_t[width * height * 3];
            
            for(uint i = 0; i < pixels.size(); ++i) {
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
#pragma once
#include "Vector.hpp"
#include "Pixel.hpp"
#include "Camera.hpp"

class Image {
    private:
        Pixel** pixels;
        uint width;
        uint height;       

    public:
        float gaussianKernel[3][3] = {
                {1. , 2. , 1.},
                {2. , 4. , 2.},
                {1. , 2. , 1.}
            };
            
        __host__ Image(uint _width, uint _height) : width(_width), height(_height) {
            pixels = new Pixel*[width];
            for(uint w=0;w<width;w++) {
                pixels[w] = new Pixel[height];
            }
        };
        __host__ Image(Camera* cam) {
            width = cam->getWidth();
            height = cam->getHeight();
            pixels = new Pixel*[width];
            for(uint w=0;w<width;w++) {
                pixels[w] = new Pixel[height];
            }
            for(uint h = 0; h < height; ++h) {
                for(uint w = 0; w < width; ++w) {
                    pixels[w][h] = cam->getPixel(h*width+w);
                }
            }
        }
        __host__ ~Image() {
            for(uint w=0;w<width;w++) {
                delete[] pixels[w];
            }
            delete[] pixels;
        };

        __host__ __device__ void setPixel(const Pixel& color, uint w, uint h) {
            pixels[w][h] = color;
        }
        __host__ __device__ Pixel getPixel(uint w, uint h) const {
            return pixels[w][h];
        }

        __host__ __device__ uint getNbPixels() const {
            return width*height;
        }

        __host__ __device__ uint getWidth() const {
            return width;
        }

        __host__ __device__ uint getHeight() const {
            return height;
        }

        // Kernel should be square with odd size
        __host__ Image convolve(const float kernel[3][3], uint sizeKernel) {
            Image img = Image(width,height);
            int offset = (sizeKernel-1)/2;
            for(uint w=0;w<width;w++) {
                for(uint h=0;h<height;h++) {
                    Vector<float> center;
                    for (int x=-offset;x<offset+1;x++) {
                        for (int y=-offset;y<offset+1;y++) {
                            //std::cout << w << " " << h << " " << x << " " << y << std::endl;
                            if (w+x < width && h+y < height)
                                center+=pixels[w+x][h+y].toVector()*kernel[x+offset][y+offset]/16;
                        }
                    }
                    img.setPixel(Pixel(center),w,h);
                }
            }
            return img;
        }


};

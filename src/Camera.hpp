#pragma once
#include "Vector.hpp"
#include "Pixel.hpp"
#include "Matrix.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <png.h>
#include <thread>

#include <SDL2/SDL.h>

#define ROT_RIGHT 2000
#define ROT_FRONT 2001
#define ROT_UP 2002

class Camera {
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
        uint FPS = 30;

        std::vector<Pixel> pixels;

        SDL_Window* window;
        SDL_Renderer* renderer;
        SDL_Texture* texture;

        std::thread renderThread;
        bool isOnBool = true;

    public:
        Camera(){};
        Camera(Vector<float> pos, uint width0, uint height0) : position(pos), vectFront(Vector<float>(0,1,0)), vectUp(Vector<float>(0,0,1)), vectRight(Vector<float>(1,0,0).crossProduct(Vector<float>(0,0,1)).normalize()), width(width0), height(height0), pixels(width0*height0) {
            capteurWidth = (0.005*width0)/(1.*height0);
            capteurHeight = 0.005;
            showImage();
        };
        Camera(Vector<float> pos, Vector<float> front, uint width0, uint height0) : position(pos), vectFront(front.normalize()), vectUp(Vector<float>(0,0,1)), vectRight(front.crossProduct(Vector<float>(0,0,1)).normalize()), width(width0), height(height0), pixels(width0*height0) {
            capteurWidth = (0.005*width0)/(1.*height0);
            capteurHeight = 0.005;
            showImage();
        };
        ~Camera() {};

        bool isOn() const {
            return isOnBool;
        }

        inline uint getWidth() const {
            return width;
        }

        inline uint getHeight() const {
            return height;
        }

        float getGamma() const {
            return gamma;
        }

        void setGamma(float g) {
            gamma = g;
        }

        Vector<float> getPixelCoordOnCapt(float w, float h) const {
            float W = (1.*w - width/2.)*(1.*capteurWidth/width);
            float H = (-1.*h + height/2.)*(1.*capteurHeight/height);
            return vectUp*H + vectRight*W;
        }

        inline Pixel getPixel(uint index) const {
            return pixels[index];
        }

        void setPixel(uint index, const Pixel& color) {
            pixels[index] = color;
        }

        inline Vector<float> getPosition() const {
            return position;
        }

        void setPosition(const Vector<float>& pos) {
            position=pos;
        }

        void move(const Vector<float>& offset) {
            position += offset;
        }

        inline Vector<float> getVectFront() const {
            return vectFront;
        }

        void setVectFront(Vector<float>& ori) {
            vectFront=ori;
        }

        inline float getFov() const {
            return fov;
        }

        void rotate(float angle, uint axis) {
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
                std::cout << "Wrong axis provided" << std::endl;
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

        void initWindow() {
            SDL_Init(SDL_INIT_EVERYTHING);
            window = SDL_CreateWindow("Raytracer", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0);
            renderer = SDL_CreateRenderer(window, -1, 0);
        }

        void closeWindow() {
            SDL_DestroyRenderer(renderer);
            SDL_DestroyWindow(window);
        }

        void updateTexture() {
            SDL_Surface* surface = SDL_CreateRGBSurface(0, width, height, 24, 0, 0, 0, 0);
            unsigned char* surface_pixels = (unsigned char*)surface -> pixels;
            for(uint h = 0; h < height; ++h) {
                for(uint w = 0; w < width; ++w) {
                    surface_pixels[3 * (h * surface->w + w) + 0] = pixels[h*width+w].getB();
                    surface_pixels[3 * (h * surface->w + w) + 1] = pixels[h*width+w].getG();
                    surface_pixels[3 * (h * surface->w + w) + 2] = pixels[h*width+w].getR();
                    if ((int) pixels[h*width+w].getB() != 0) {
                        std::cout << pixels[h*width+w].getB() << std::endl;
                    }
                }
            }
            SDL_DestroyTexture(texture);
            texture = SDL_CreateTextureFromSurface(renderer, surface);
            SDL_FreeSurface(surface);
        }

        static void threadLoop(Camera* cam) {
            std::cout << "Thread launched" << std::endl;
            cam->initWindow();
            SDL_Event e;
            bool running = true;
            while(running) {
                while (SDL_PollEvent(&e)) {
                    if (e.type == SDL_QUIT) running = false;
                    if (e.key.keysym.sym == SDLK_UP) {
                        cam->move(Vector<float>(0.1, 0., 0.));
                    }
                    if (e.key.keysym.sym == SDLK_DOWN) {
                        cam->move(Vector<float>(-0.1, 0., 0.));
                    }
                    if (e.key.keysym.sym == SDLK_LEFT) {
                        cam->move(Vector<float>(0., 0.1, 0.));
                    }
                    if (e.key.keysym.sym == SDLK_RIGHT) {
                        cam->move(Vector<float>(0., -0.1, 0.));
                    }
                    if (e.key.keysym.sym == SDLK_SPACE) {
                        cam->move(Vector<float>(0., 0., 0.1));
                    }
                    if (e.key.keysym.sym == SDLK_LSHIFT) {
                        cam->move(Vector<float>(0., 0., -0.1));
                    }
                }
                cam->updateTexture();
                SDL_RenderCopy(cam->renderer, cam->texture, nullptr, nullptr);
                SDL_RenderPresent(cam->renderer);
                SDL_Delay(1000*(Uint32)1./cam->FPS);
            }
            cam->isOnBool = false;
            cam->closeWindow();
        }

        void showImage() {
            renderThread = std::thread(threadLoop, this);
        }

        void stop() {
            renderThread.join();
        }
};
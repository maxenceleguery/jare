#pragma once
#include "Vector.hpp"
#include "Camera.hpp"
#include "Face.hpp"
#include "FaceCuda.hpp"
#include "Line.hpp"
#include "Ray.hpp"
#include "RayTrace.cuh"

#include <stdlib.h>
#include <time.h>
#include "omp.h"

#include <cuda_runtime.h>

#define SIMPLE_RENDER 1000
#define RAYTRACING 1001

#define cudaErrorCheck(call){cudaAssert(call,__FILE__,__LINE__);}

class Environment {
    public:
        using Faces = std::vector<Face>;
    private:
        Camera* cam;
        Faces faces;
        uint samples = 5;
        Pixel backgroundColor = Pixel(0,0,0);
        uint mode = RAYTRACING;

    public:
        Environment(): faces(1) {};
        Environment(Camera* cam0) : cam(cam0), faces(1) {};
        ~Environment() {};

        void addFace(Face& face) {
            faces.push_back(face);
        }

        void addObj(const std::string name) {
            std::cout << "Loading " << name.c_str() << std::endl;
            std::string path = std::string("./models/");
            std::ifstream monFlux((path+name).c_str());
            std::string ligne;
            getline(monFlux, ligne);
            std::cout << ligne << std::endl;
            
        }        

        void render() {
            uint H = cam->getHeight();
            uint W = cam->getWidth();
            #pragma omp parallel for num_threads(omp_get_num_devices())
            for(uint h = 0; h < H; ++h) {
                #pragma omp parallel for num_threads(omp_get_num_devices())
                for(uint w = 0; w < W; ++w) {
                    Pixel color;
                    Vector<double> colorVec;
                    if (mode==SIMPLE_RENDER) {
                        Vector<double> direction = (cam->getOrientation()*cam->getFov()+cam->getPixelCoordOnCapt(w,h)).normalize();
                        Ray ray = Ray(cam->getPosition(),direction);

                        color = ray.rayTrace1(faces, backgroundColor);
                    }
                    else if (mode==RAYTRACING) {
                        Vector<double> vectTmp;
                        samples = 4; // has to be a perfect square
                        int samplesSqrt=(int)std::sqrt(samples);
                        
                        double dy=-(samplesSqrt-1)/2.;
                        do {
                            double dx=-(samplesSqrt-1)/2.;
                            do {
                                Vector<double> direction = (cam->getOrientation()*cam->getFov()+cam->getPixelCoordOnCapt(w+dx/(1.*samplesSqrt),h+dy/(1.*samplesSqrt))).normalize();
                                Ray ray = Ray(cam->getPosition(),direction);

                                vectTmp = (ray.rayTrace2(faces, h*w)).toVector();

                                colorVec += vectTmp;
                                dx++;
                            } while (dx<(samplesSqrt-1)/2);
                            dy++;
                        } while (dy<(samplesSqrt-1)/2.);
                        colorVec/=(samples/2);
                        color=Pixel(colorVec);
                    }
                    cam->setPixel(h*W+w, color);
                }
            }
        }

        void renderCuda() {
            uint H = cam->getHeight();
            uint W = cam->getWidth();

            uint nbFaces = faces.size();
            FaceCuda* faces_ptr = new FaceCuda[nbFaces];
            for (uint i=0;i<nbFaces;i++) {
                std::vector<Vector<double>> vertices = faces[i].getVertices();

                uint nbVertices = vertices.size();

                Vector<double>* d_vertices;
                cudaErrorCheck(cudaMalloc(&d_vertices,nbVertices*sizeof(Vector<double>)));
                cudaErrorCheck(cudaMemcpy(d_vertices, vertices.data(), nbVertices*sizeof(Vector<double>), cudaMemcpyHostToDevice));
                faces_ptr[i].setvertices(d_vertices);
                faces_ptr[i].setNbVertices(nbVertices);
                faces_ptr[i].setMaterial(faces[i].getMaterial());
            }

            FaceCuda* d_faces;
            cudaErrorCheck(cudaMalloc(&d_faces,nbFaces*sizeof(FaceCuda)));
            cudaErrorCheck(cudaMemcpy(d_faces, faces_ptr, nbFaces*sizeof(FaceCuda), cudaMemcpyHostToDevice));

            Pixel* colors = new Pixel[W * H];
            Pixel* d_colors;
            cudaErrorCheck(cudaMalloc(&d_colors, H*W*sizeof(Pixel)));
            cudaErrorCheck(cudaMemcpy(d_colors, colors, H*W*sizeof(Pixel), cudaMemcpyHostToDevice));

            Ray* rays = new Ray[W * H];
            Ray* d_rays;
            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w) {
                    Vector<double> direction = (cam->getOrientation()*cam->getFov()+cam->getPixelCoordOnCapt(w,h)).normalize();
                    rays[h*W+w] = Ray(cam->getPosition(),direction);
                }
            }
            cudaErrorCheck(cudaMalloc(&d_rays, H*W*sizeof(Ray)));
            cudaErrorCheck(cudaMemcpy(d_rays, rays, H*W*sizeof(Ray), cudaMemcpyHostToDevice));
            delete[] rays;

            int blocksize = 256;
            int nblocks = H*W / blocksize;

            rayTrace3(d_rays,d_faces,d_colors,nbFaces,nblocks,blocksize,W,H);

            cudaErrorCheck(cudaMemcpy(colors, d_colors, H*W*sizeof(Pixel), cudaMemcpyDeviceToHost));
            cudaErrorCheck(cudaFree(d_colors));
            for (uint i=0;i<nbFaces;i++) {
                cudaErrorCheck(cudaFree(faces_ptr[i].getVertices()));
            }
            cudaErrorCheck(cudaFree(d_faces));
            cudaErrorCheck(cudaFree(d_rays));

            delete[] faces_ptr;

            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w) {
                    cam->setPixel(h*W+w, colors[h*W+w]);
                }
            }
            delete[] colors;
        }

        void addBackground(const Pixel& color) {
            backgroundColor = color;
            for(uint h = 0; h < cam->getHeight(); ++h) 
                for(uint w = 0; w < cam->getWidth(); ++w)
                    cam->setPixel(h*cam->getWidth()+w, color);
        }
};
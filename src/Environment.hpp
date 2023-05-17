#pragma once
#include "Vector.hpp"
#include "Camera.hpp"
#include "Face.hpp"
#include "FaceCuda.hpp"
#include "Line.hpp"
#include "Ray.hpp"
#include "RayTrace.cuh"
#include "Image.hpp"
#include "Obj.hpp"

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

        uint samplesByThread = 8; //8
        uint threadsByRay = 4; //4

        Pixel backgroundColor = Pixel(0,0,0);
        uint mode = RAYTRACING;

    public:
        Environment(): faces(1) {};
        Environment(Camera* cam0) : cam(cam0), faces(1) {};
        ~Environment() {};

        void addFace(Face& face) {
            faces.push_back(face);
        }

        void addSquare(Vector<double> v1, Vector<double> v2, Vector<double> v3, Vector<double> v4, Pixel color) {
            Face face = Face(v1,color);
            face.addVectex(v2);
            face.addVectex(v3);
            face.addVectex(v4);
            addFace(face);
        }

        void addSquare(Vector<double> v1, Vector<double> v2, Vector<double> v3, Vector<double> v4, Material mat) {
            Face face = Face(v1,mat);
            face.addVectex(v2);
            face.addVectex(v3);
            face.addVectex(v4);
            addFace(face);
        }

        void addObj(const std::string name, Vector<double> offset, double scale, Material mat) {
            Obj obj = Obj(name);
            //obj.print();

            std::vector<Vector<double>> vertices = obj.getVertices();
            std::vector<std::vector<Vector<int>>> indexes = obj.getIndexes();

            double angle = 3.14159/2.0;
            double ux = 1;
            double uy = 0;
            double uz = 0;
            Matrix<double> P = Matrix<double>(ux*ux,ux*uy,ux*uz,ux*uy,uy*uy,uy*uz,ux*uz,uy*uz,uz*uz);
            Matrix<double> I = Matrix<double>(1.,MATRIX_EYE);
            Matrix<double> Q = Matrix<double>(0,-uz,uy,uz,0,-ux,-uy,ux,0);

            Matrix<double> R = P + (I-P)*std::cos(angle) + Q*std::sin(angle);

            for (uint i=0;i<indexes.size();i++) {
                std::vector<Vector<int>> fi = indexes[i];
                Vector<double> vec1 = R*vertices[fi[0].getX()]*scale + offset;
                Face face(vec1,mat);
                for (uint j=1;j<fi.size();j++) {
                    face.addVectex( R*vertices[fi[j].getX()]*scale + offset );
                }
                addFace(face);
            } 
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
                        Vector<double> direction = (cam->getVectFront()*cam->getFov()+cam->getPixelCoordOnCapt(w,h)).normalize();
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
                                Vector<double> direction = (cam->getVectFront()*cam->getFov()+cam->getPixelCoordOnCapt(w+dx/(1.*samplesSqrt),h+dy/(1.*samplesSqrt))).normalize();
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

            Pixel* colors = new Pixel[threadsByRay * W * H];
            Pixel* d_colors;
            cudaErrorCheck(cudaMalloc(&d_colors, threadsByRay*H*W*sizeof(Pixel)));
            cudaErrorCheck(cudaMemcpy(d_colors, colors, threadsByRay*H*W*sizeof(Pixel), cudaMemcpyHostToDevice));

            Ray* rays = new Ray[W * H];
            Ray* d_rays;
            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w) {
                    Vector<double> direction = (cam->getVectFront()*cam->getFov()+cam->getPixelCoordOnCapt(w,h)).normalize();
                    rays[h*W+w] = Ray(cam->getPosition(),direction);
                }
            }
            cudaErrorCheck(cudaMalloc(&d_rays, H*W*sizeof(Ray)));
            cudaErrorCheck(cudaMemcpy(d_rays, rays, H*W*sizeof(Ray), cudaMemcpyHostToDevice));
            delete[] rays;

            int blocksize = 256; // 1024 at most
            int nblocks = threadsByRay*H*W / blocksize;

            srand(time(NULL));
            int state  = rand() % 50 + 1;

            rayTrace(d_rays,d_faces,d_colors,nbFaces,nblocks,blocksize,W,H,samplesByThread,threadsByRay,state);

            cudaErrorCheck(cudaMemcpy(colors, d_colors, threadsByRay*H*W*sizeof(Pixel), cudaMemcpyDeviceToHost));
            cudaErrorCheck(cudaFree(d_colors));
            for (uint i=0;i<nbFaces;i++) {
                cudaErrorCheck(cudaFree(faces_ptr[i].getVertices()));
            }
            cudaErrorCheck(cudaFree(d_faces));
            cudaErrorCheck(cudaFree(d_rays));

            delete[] faces_ptr;

            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w) {
                    Vector<double> partialColor;
                    for(uint i=0;i<threadsByRay;i++)
                        partialColor+=(colors[W*H*i + h*W + w].toVector());
                    partialColor/=threadsByRay;
                    cam->setPixel(h*W+w, Pixel(partialColor));
                }
            }
            delete[] colors;

            Image img = Image(cam);
            //Image img2 = img.convolve(img.gaussianKernel,3);
            
            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w) {
                    cam->setPixel(h*W+w, img.getPixel(w,h));
                }
            }
        }

        void addBackground(const Pixel& color) {
            backgroundColor = color;
            for(uint h = 0; h < cam->getHeight(); ++h) 
                for(uint w = 0; w < cam->getWidth(); ++w)
                    cam->setPixel(h*cam->getWidth()+w, color);
        }
};
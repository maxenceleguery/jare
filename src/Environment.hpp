#pragma once
#include "Vector.hpp"
#include "Camera.hpp"
#include "Triangle.hpp"
#include "Triangle.hpp"
#include "Line.hpp"
#include "Ray.hpp"
#include "RayTrace.cuh"
#include "Image.hpp"
#include "Obj.hpp"
#include "Mesh.hpp"
#include "utils/ProgressBar.hpp"

#include <stdlib.h>
#include <time.h>
#include "omp.h"
#include <chrono>

#include <cuda_runtime.h>

#define cudaErrorCheck(call){cudaAssert(call,__FILE__,__LINE__);}

enum Mode {
    SIMPLE_RENDER,
    RAYTRACING,
    BVH_RAYTRACING
};

class Environment {
    private:
        Camera* cam;
        Meshes meshes;
        uint samples = 5;

        uint samplesByThread = 4; //8
        uint threadsByRay = 1; //1

        Pixel backgroundColor = Pixel(0,0,0);
        Mode mode = BVH_RAYTRACING;

    public:
        Environment() {};
        Environment(Camera* cam0) : cam(cam0) {};
        ~Environment() {};

        void setMode(const Mode m) {
            mode = m;
        }

        void addTriangle(Triangle& triangle) {
            meshes.push_back(Mesh(triangle));
        }

        void addSquare(Vector<double> v1, Vector<double> v2, Vector<double> v3, Vector<double> v4, Material mat) {
            Triangle triangle = Triangle(v1,mat);
            triangle.setvertex(1, v2);
            triangle.setvertex(2, v4);

            Triangle triangleBis = Triangle(v2,mat);
            triangleBis.setvertex(1, v3);
            triangleBis.setvertex(2, v4);

            Mesh mesh = Mesh(triangle);
            mesh.push_back(triangleBis);
            meshes.push_back(mesh);
        }

        void addSquare(Vector<double> v1, Vector<double> v2, Vector<double> v3, Vector<double> v4, Pixel color) {
            addSquare(v1, v2, v3, v4, Material(color));
        }

        void addObj(const std::string name, Vector<double> offset, double scale, Material mat) {
            std::cout << "Loading " << name.c_str() << std::endl;
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

            /*
            The OBJ format can provide multiple vertices for one triangle. We have to convert it in triangles as follow.
            */
            Mesh mesh = Mesh();
            for (uint i=0;i<indexes.size();i++) {
                std::vector<Vector<int>> fi = indexes[i];

                for (uint v=2;v<fi.size();v++) {
                    Triangle triangle = Triangle(mat);
                    if (v==2) {
                        triangle.setvertex(0, R*vertices[fi[0].getX()]*scale + offset );
                        triangle.setvertex(1, R*vertices[fi[1].getX()]*scale + offset );
                        triangle.setvertex(2, R*vertices[fi[2].getX()]*scale + offset );
                    } else {
                        triangle.setvertex(0, R*vertices[fi[v-3].getX()]*scale + offset );
                        triangle.setvertex(1, R*vertices[fi[v-1].getX()]*scale + offset );
                        triangle.setvertex(2, R*vertices[fi[v].getX()]*scale + offset );
                    }
                    if (triangle.isPlaneValid()) {
                        mesh.push_back(triangle);
                        obj.nbTriangles += 1;
                    } else {
                        //std::cout << "Triangle not valid" << std::endl;
                        obj.failedTriangles += 1;
                    }
                }
            }
            meshes.push_back(mesh);
            std::cout << name.c_str() << " loaded with " << obj.nbTriangles << " triangles and " << obj.failedTriangles << " wrong ones." << std::endl;
        }        

        void render() {
            const uint H = cam->getHeight();
            const uint W = cam->getWidth();

            Array<BVH> BVHs = Array<BVH>();
            if (mode==BVH_RAYTRACING) {
                for (uint i=0; i<meshes.size(); i++) {
                    std::cout << "BVH " << i << std::endl;
                    BVHs.push_back(BVH(meshes[i]));
                }
                std::cout << "BVHs done" << std::endl;
            }

            //#pragma omp parallel for num_threads(omp_get_num_devices())
            for(uint h = 0; h < H; ++h) {
                //#pragma omp parallel for num_threads(omp_get_num_devices())
                for(uint w = 0; w < W; ++w) {
                    printProgress((h*W+(w+1))/(1.*H*W));

                    Pixel color;
                    Vector<double> colorVec;
                    
                    if (mode==SIMPLE_RENDER) {
                        Vector<double> direction = (cam->getVectFront()*cam->getFov()+cam->getPixelCoordOnCapt(w,h)).normalize();
                        Ray ray = Ray(cam->getPosition(),direction);

                        color = ray.rayTrace1(meshes, backgroundColor);
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

                                vectTmp = (ray.rayTrace2(meshes, h*w)).toVector();

                                colorVec += vectTmp;
                                dx++;
                            } while (dx<(samplesSqrt-1)/2);
                            dy++;
                        } while (dy<(samplesSqrt-1)/2.);
                        colorVec/=(samples/2);
                        color=Pixel(colorVec);
                    }

                    else if (mode==BVH_RAYTRACING) {
                        Vector<double> vectTmp;
                        samples = 1; // has to be a perfect square
                        int samplesSqrt=(int)std::sqrt(samples);
                        
                        double dy=-(samplesSqrt-1)/2.;
                        do {
                            double dx=-(samplesSqrt-1)/2.;
                            do {
                                Vector<double> direction = (cam->getVectFront()*cam->getFov()+cam->getPixelCoordOnCapt(w+dx/(1.*samplesSqrt),h+dy/(1.*samplesSqrt))).normalize();
                                Ray ray = Ray(cam->getPosition(),direction);

                                vectTmp = (ray.rayTraceBVHHost(BVHs, h*w)).toVector();

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
            //uint nbMesh = meshes.size();
            Mesh triangles = meshes[0];

            uint nbTriangles = 1;

            auto start = std::chrono::steady_clock::now();

            Triangle* triangles_cpu = new Triangle[nbTriangles];
            for (uint i=0;i<nbTriangles;i++) {
                triangles_cpu[i] = triangles[i];

                //Vector<double>* vertices = triangles[i].getVertices();

                //uint nbVertices = 3;

                //Vector<double> vertices_gpu[3];
                //cudaErrorCheck(cudaMalloc(&vertices_gpu, nbVertices*sizeof(Vector<double>)));
                //cudaErrorCheck(cudaMemcpy(vertices_gpu, vertices, nbVertices*sizeof(Vector<double>), cudaMemcpyHostToDevice));
                //triangles_cpu[i].setvertices(vertices_gpu);
                //triangles_cpu[i].setvertex(0, triangles[i].getVertex(0));
                //triangles_cpu[i].setMaterial(triangles[i].getMaterial());
            }

            Triangle* triangles_gpu;
            cudaErrorCheck(cudaMalloc(&triangles_gpu, nbTriangles*sizeof(Triangle)));
            cudaErrorCheck(cudaMemcpy(triangles_gpu, triangles_cpu, nbTriangles*sizeof(Triangle), cudaMemcpyHostToDevice));

            Pixel* colors = new Pixel[threadsByRay * W * H];
            Pixel* colors_gpu;
            cudaErrorCheck(cudaMalloc(&colors_gpu, threadsByRay*H*W*sizeof(Pixel)));
            cudaErrorCheck(cudaMemcpy(colors_gpu, colors, threadsByRay*H*W*sizeof(Pixel), cudaMemcpyHostToDevice));

            Ray* rays = new Ray[W * H];
            Ray* rays_gpu;
            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w) {
                    Vector<double> direction = (cam->getVectFront()*cam->getFov()+cam->getPixelCoordOnCapt(w,h)).normalize();
                    rays[h*W+w] = Ray(cam->getPosition(),direction);
                }
            }
            cudaErrorCheck(cudaMalloc(&rays_gpu, H*W*sizeof(Ray)));
            cudaErrorCheck(cudaMemcpy(rays_gpu, rays, H*W*sizeof(Ray), cudaMemcpyHostToDevice));
            delete[] rays;

            auto end = std::chrono::steady_clock::now();

            std::chrono::duration<double> elapsed_seconds = end-start;
            std::cout << "Copy on device:\t\t" << elapsed_seconds.count() << "s\n";

            int blocksize = 256; // 1024 at most
            int nblocks = threadsByRay*H*W / blocksize;

            srand(time(NULL));
            int state  = rand() % 50 + 1;

            start = std::chrono::steady_clock::now();
            rayTrace(rays_gpu,triangles_gpu,colors_gpu,nbTriangles,nblocks,blocksize,W,H,samplesByThread,threadsByRay,state);
            end = std::chrono::steady_clock::now();
            elapsed_seconds = end-start;
            std::cout << "Raytracing time:\t" << elapsed_seconds.count() << "s\n";

            start = std::chrono::steady_clock::now();
            cudaErrorCheck(cudaMemcpy(colors, colors_gpu, threadsByRay*H*W*sizeof(Pixel), cudaMemcpyDeviceToHost));
            end = std::chrono::steady_clock::now();
            elapsed_seconds = end-start;
            std::cout << "Copy back to host:\t" << elapsed_seconds.count() << "s\n";

            start = std::chrono::steady_clock::now();
            cudaErrorCheck(cudaFree(colors_gpu));
            /*
            for (uint i=0;i<nbTriangles;i++) {
                cudaErrorCheck(cudaFree(triangles_cpu[i].getVertices()));
            }*/
            cudaErrorCheck(cudaFree(triangles_gpu));
            cudaErrorCheck(cudaFree(rays_gpu));
            delete[] triangles_cpu;
            end = std::chrono::steady_clock::now();
            elapsed_seconds = end-start;
            std::cout << "Free memory:\t\t" << elapsed_seconds.count() << "s\n";

            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w)
                    cam->setPixel(h*W+w, Pixel(colors[h*W + w].toVector().pow(1/cam->getGamma())));
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
        
        /*
        void renderCudaBVH() {
            uint H = cam->getHeight();
            uint W = cam->getWidth();

            Array<BVH> BVHs = Array<BVH>();
            if (mode==BVH_RAYTRACING) {
                for (uint i=0; i<meshes.size(); i++) {
                    BVHs.push_back(BVH(meshes[i]));
                }
                std::cout << "BVHs done" << std::endl;
            }

            auto start = std::chrono::steady_clock::now();

            Array<BVH>* bvhs_gpu;
            cudaErrorCheck(cudaMalloc(&bvhs_gpu, sizeof(Array<BVH>)));
            cudaErrorCheck(cudaMemcpy(bvhs_gpu, &BVHs, sizeof(Array<BVH>), cudaMemcpyHostToDevice));

            Triangle* triangles_cpu = new Triangle[nbTriangles];
            for (uint i=0;i<nbTriangles;i++) {
                Vector<double>* vertices = triangles[i].getVertices();

                uint nbVertices = 3;

                Vector<double>* vertices_gpu;
                cudaErrorCheck(cudaMalloc(&vertices_gpu, nbVertices*sizeof(Vector<double>)));
                cudaErrorCheck(cudaMemcpy(vertices_gpu, vertices, nbVertices*sizeof(Vector<double>), cudaMemcpyHostToDevice));
                triangles_cpu[i].setvertices(vertices_gpu);
                triangles_cpu[i].setMaterial(triangles[i].getMaterial());
            }

            Triangle* triangles_gpu;
            cudaErrorCheck(cudaMalloc(&triangles_gpu, nbTriangles*sizeof(Triangle)));
            cudaErrorCheck(cudaMemcpy(triangles_gpu, triangles_cpu, nbTriangles*sizeof(Triangle), cudaMemcpyHostToDevice));

            Pixel* colors = new Pixel[threadsByRay * W * H];
            Pixel* colors_gpu;
            cudaErrorCheck(cudaMalloc(&colors_gpu, threadsByRay*H*W*sizeof(Pixel)));
            cudaErrorCheck(cudaMemcpy(colors_gpu, colors, threadsByRay*H*W*sizeof(Pixel), cudaMemcpyHostToDevice));

            Ray* rays = new Ray[W * H];
            Ray* rays_gpu;
            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w) {
                    Vector<double> direction = (cam->getVectFront()*cam->getFov()+cam->getPixelCoordOnCapt(w,h)).normalize();
                    rays[h*W+w] = Ray(cam->getPosition(),direction);
                }
            }
            cudaErrorCheck(cudaMalloc(&rays_gpu, H*W*sizeof(Ray)));
            cudaErrorCheck(cudaMemcpy(rays_gpu, rays, H*W*sizeof(Ray), cudaMemcpyHostToDevice));
            delete[] rays;

            auto end = std::chrono::steady_clock::now();

            std::chrono::duration<double> elapsed_seconds = end-start;
            std::cout << "Copy on device:\t\t" << elapsed_seconds.count() << "s\n";

            int blocksize = 256; // 1024 at most
            int nblocks = threadsByRay*H*W / blocksize;

            srand(time(NULL));
            int state  = rand() % 50 + 1;

            start = std::chrono::steady_clock::now();
            rayTraceBVH(rays_gpu,triangles_gpu,colors_gpu,nbTriangles,nblocks,blocksize,W,H,samplesByThread,threadsByRay,state);
            end = std::chrono::steady_clock::now();
            elapsed_seconds = end-start;
            std::cout << "Raytracing time:\t" << elapsed_seconds.count() << "s\n";

            start = std::chrono::steady_clock::now();
            cudaErrorCheck(cudaMemcpy(colors, colors_gpu, threadsByRay*H*W*sizeof(Pixel), cudaMemcpyDeviceToHost));
            end = std::chrono::steady_clock::now();
            elapsed_seconds = end-start;
            std::cout << "Copy back to host:\t" << elapsed_seconds.count() << "s\n";

            start = std::chrono::steady_clock::now();
            cudaErrorCheck(cudaFree(colors_gpu));
            for (uint i=0;i<nbTriangles;i++) {
                cudaErrorCheck(cudaFree(triangles_cpu[i].getVertices()));
            }
            cudaErrorCheck(cudaFree(triangles_gpu));
            cudaErrorCheck(cudaFree(rays_gpu));
            delete[] triangles_cpu;
            end = std::chrono::steady_clock::now();
            elapsed_seconds = end-start;
            std::cout << "Free memory:\t\t" << elapsed_seconds.count() << "s\n";

            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w)
                    cam->setPixel(h*W+w, Pixel(colors[h*W + w].toVector().pow(1/cam->getGamma())));
            }
            delete[] colors;

            /*
            Image img = Image(cam);
            //Image img2 = img.convolve(img.gaussianKernel,3);
            
            for(uint h = 0; h < H; ++h) {
                for(uint w = 0; w < W; ++w) {
                    cam->setPixel(h*W+w, img.getPixel(w,h));
                }
            }
            */
        //}

        void addBackground(const Pixel& color) {
            backgroundColor = color;
            for(uint h = 0; h < cam->getHeight(); ++h) 
                for(uint w = 0; w < cam->getWidth(); ++w)
                    cam->setPixel(h*cam->getWidth()+w, color);
        }
};
#pragma once

#include "Vector.hpp"
#include "Camera.hpp"
#include "Triangle.hpp"
#include "Line.hpp"
#include "Ray.hpp"

#include "shaders/Rasterize.hpp"
#include "shaders/RayTrace.hpp"
#include "shaders/Aggreg.hpp"
#include "shaders/Convolve.hpp"
#include "shaders/Operation.hpp"

#include "Tracing.hpp"
#include "TRSMatrix.hpp"

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
        Meshes meshes;
        uint samples = 5;

        uint samplesByThread = 2;

        Pixel backgroundColor = Pixel(0,0,0);
        Mode mode = BVH_RAYTRACING;

    public:
        Camera* cam;
        Array<BVH> BVHs = Array<BVH>();

        Environment() {
            std::chrono::milliseconds ms = duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            );
            srand(ms.count());
        };
        Environment(Camera* cam0) : Environment() {
            cam = cam0;
        };
        ~Environment() {
            if (mode==BVH_RAYTRACING) {
                BVHs.cpu();
                BVHs.free();
            }
        };

        void addBackground(const Pixel& color) {
            backgroundColor = color;
            for(uint h = 0; h < cam->getHeight(); ++h) 
                for(uint w = 0; w < cam->getWidth(); ++w)
                    cam->setPixel(h*cam->getWidth()+w, color);
        }

        void compute_bvhs() {
            auto start = std::chrono::steady_clock::now();
            for (uint i=0; i<meshes.size(); i++) {
                BVHs.push_back(BVH(meshes[i]));
            }
            BVHs.cuda();
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<float> elapsed_seconds = end-start;
            std::cout << "BVHs constructed and on device:\t\t" << elapsed_seconds.count() << "s\n";
        }

        void setMode(const Mode m) {
            mode = m;
        }

        void addTriangle(Triangle& triangle) {
            meshes.push_back(Mesh(triangle));
        }

        void addSquare(Vector<float> v1, Vector<float> v2, Vector<float> v3, Vector<float> v4, Material mat) {
            Triangle triangle = Triangle(v1, v2, v4);
            triangle.setMaterial(mat);

            Triangle triangleBis = Triangle(v2, v3, v4);
            triangleBis.setMaterial(mat);

            Mesh mesh = Mesh(triangle);
            mesh.push_back(triangleBis);
            meshes.push_back(mesh);
        }

        void addSquare(Vector<float> v1, Vector<float> v2, Vector<float> v3, Vector<float> v4, Pixel color) {
            addSquare(v1, v2, v3, v4, Material(color));
        }

        void addObj(const std::string name, const Vector<float>& offset, const float scale, const Vector<float>& rot_angle_deg, const Material mat) {
            std::cout << "Loading " << name.c_str() << std::endl;
            Obj obj = Obj(name);
            //obj.print();

            std::vector<Vector<float>> vertices = obj.getVertices();
            std::vector<Vector<float>> normal_vertices = obj.getNormalVertices();
            std::vector<std::vector<Vector<int>>> indexes = obj.getIndexes();

            /*
            The OBJ format can provide multiple vertices for one triangle. We have to convert it in triangles as follow.
            */
            Mesh mesh = Mesh();
            for (uint i=0;i<indexes.size();i++) {
                std::vector<Vector<int>> fi = indexes[i];

                for (uint v=2;v<fi.size();v++) {
                    Triangle triangle = Triangle(mat);
                    if (v==2) {
                        for (uint j = 0; j<3; j++) { 
                            triangle.setvertex(j, vertices[fi[j].getX()]);
                            triangle.setNormal(j, normal_vertices[fi[j].getZ()]);
                        }
                    } else {
                        triangle.setvertex(0, vertices[fi[v-3].getX()]);
                        triangle.setvertex(1, vertices[fi[v-1].getX()]);
                        triangle.setvertex(2, vertices[fi[v  ].getX()]);

                        triangle.setNormal(0, normal_vertices[fi[v-3].getZ()]);
                        triangle.setNormal(1, normal_vertices[fi[v-1].getZ()]);
                        triangle.setNormal(2, normal_vertices[fi[v  ].getZ()]);
                    }
                    mesh.push_back(triangle);
                    obj.nbTriangles += 1;
                }
            }
            mesh.setTransformMatrix(offset, Vector<float>(1., 1., 1.)*scale, rot_angle_deg);
            mesh.setDefaultsOrientations();
            meshes.push_back(mesh);
            std::cout << name.c_str() << " loaded with " << obj.nbTriangles << " triangles" << std::endl;
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
                    printProgress((h*W+(w+1))/(1.f*H*W));

                    uint idx = h*W+w;

                    Pixel color;
                    Vector<float> colorVec;
                    
                    if (mode==SIMPLE_RENDER) {
                        Ray ray = cam->generate_ray(w,h);

                        color = Tracing::simpleRayTraceHost(ray, meshes, backgroundColor);
                    }

                    else if (mode==RAYTRACING) {
                        Vector<float> vectTmp;
                        samples = 4; // has to be a perfect square
                        int samplesSqrt=(int)std::sqrt(samples);
                        
                        float dy=-(samplesSqrt-1)/2.f;
                        do {
                            float dx=-(samplesSqrt-1)/2.f;
                            do {
                                Ray ray = cam->generate_ray(w+dx/(1.f*samplesSqrt),h+dy/(1.f*samplesSqrt));
                                
                                vectTmp = (Tracing::rayTraceHost(ray, meshes, idx)).toVector();

                                colorVec += vectTmp;
                                dx++;
                            } while (dx<(samplesSqrt-1)/2.f);
                            dy++;
                        } while (dy<(samplesSqrt-1)/2.f);
                        colorVec/=(samples/2);
                        color=Pixel(colorVec);
                    }

                    else if (mode==BVH_RAYTRACING) {
                        Vector<float> vectTmp;
                        samples = 1; // has to be a perfect square
                        int samplesSqrt=(int)std::sqrt(samples);
                        
                        float dy=-(samplesSqrt-1)/2.f;
                        do {
                            float dx=-(samplesSqrt-1)/2.;
                            do {
                                Ray ray = cam->generate_ray(w+dx/(1.f*samplesSqrt),h+dy/(1.f*samplesSqrt));

                                vectTmp = (Tracing::rayTraceBVHHost(ray, BVHs, idx)).toVector();

                                colorVec += vectTmp;
                                dx++;
                            } while (dx<(samplesSqrt-1)/2);
                            dy++;
                        } while (dy<(samplesSqrt-1)/2.f);
                        colorVec/=(samples/2);
                        color=Pixel(colorVec);
                    }
                    cam->setPixel(h*W+w, color);
                }
            }
            if (mode==BVH_RAYTRACING) {
                BVHs.free();
            }
        }
        
        void renderCudaBVH() {
            auto start = std::chrono::steady_clock::now();
            int state  = rand() % 1000000000 + 1;
            //std::cout << state << std::endl;

            if (cam->is_raytrace_enable) {
                RayTraceShader raytrace = RayTraceShader({BVHs, *cam, samplesByThread}, state);
                compute_shader(raytrace);
                //ConvolutionShader denoise = ConvolutionShader({ {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}, *cam});
                //compute_shader(denoise);

                /*
                float Y = 0;
                for (int i = 0; i<cam->getWidth()*cam->getHeight(); i++) {
                    Y += cam->getPixel(i).getLuminance();
                }
                Y /= cam->getWidth()*cam->getHeight();
                OperationShader normalize = OperationShader({DIVISION, Y, *cam});
                compute_shader(normalize);
                */

            } else {
                RasterizeShader raster = RasterizeShader({BVHs, *cam}, state);
                compute_shader(raster);
            }

            //AggregShader shader2 = AggregShader({*cam});
            //compute_shader(shader2, state);

            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<float> elapsed_seconds = end-start;
            cam->setCurrentFPS(1.f/(elapsed_seconds.count()));
        }
};
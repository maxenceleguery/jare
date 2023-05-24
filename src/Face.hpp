#pragma once
#include <iostream>
#include "Vector.hpp"
#include "Pixel.hpp"
#include "Line.hpp"
#include "Material.hpp"

#include <cuda_runtime.h>

#include <vector>
#include <cmath>

#define PI 3.14159

class Face {
    private:
        std::vector<Vector<double>> vertices;
        Material material;

    public:
        __host__ __device__ Face() : vertices(1) {};
        __host__ Face(std::vector<Vector<double>>& vertices0, Pixel color0) : vertices(vertices0) {
            material = Material(color0);
        };
        __host__ Face(Vector<double>& vec0, Pixel color0) : vertices(1), material(color0) {
            vertices[0]=vec0;
            material = Material(color0);
        };
        __host__ Face(Vector<double>& vec0, Material mat0) : vertices(1), material(mat0) {
            vertices[0]=vec0;
        };
        __host__ ~Face() {};

        __host__ std::vector<Vector<double>> getVertices() {
            return vertices;
        }

        __host__ __device__ Material getMaterial() const {
            return material;
        }

        __host__ __device__ void setMaterial(const Material mat) {
            material=mat;
        }

        __host__ __device__ void addVectex(const Vector<double>& vec) {
            vertices.push_back(vec);
        }

        __host__ __device__ Vector<double> getNormalVector() const {
            return (vertices[1]-vertices[0]).crossProduct(vertices[2]-vertices[0]).normalize();
        }

        __host__ __device__ Vector<double> getBarycenter() const {
            Vector<double> center = vertices[0];
                for (uint i=1;i<vertices.size();i++) {
                    center+=vertices[i];
                }
            center/=vertices.size();
            return center;
        }

        __host__ bool isPlaneValid() const {
            if (vertices.size() < 3) {
                std::cout << "Not enough vertices to define a plan" << std::endl;
                return false;
            } else if (vertices.size() == 3) {
                return true;
            } else {
                Vector<double> normalVector = getNormalVector();
                if (normalVector == Vector<double>())
                    return false;
                for (uint i=0; i<vertices.size();i++) {
                    if (std::abs(normalVector*(vertices[i]-vertices[0]).normalize()) > 1E-3)
                        return false;
                }
                return true;
            }
        }

        __host__ __device__ bool isOnPlane(const Vector<double>& vec) const {
            Vector<double> normalVector = getNormalVector();
            return !(std::abs( (vec-vertices[0])*normalVector ) > 1E-5);
        }

        __host__ __device__ bool isInPolygoneOld(const Vector<double>& vec) {
            if (isOnPlane(vec)) {
                Vector<double> normalVector = getNormalVector();
                for (uint i=0;i<vertices.size()-1;i++) {
                    Vector<double> vec2 = (vec-vertices[i]).crossProduct(vertices[i+1]-vertices[i]);
                    if (std::abs(normalVector*vec2) > 1E-3)
                        return false;
                }
                return true;
            }
            return false;
        }

        __host__ __device__ bool isInPolygoneOld2(const Vector<double>& vec) {
            if (isOnPlane(vec)) {
                double sumAngles = 0.;
                for (uint i=0;i<vertices.size()-1;i++) {
                    //std::cout << sumAngles << std::endl;
                    sumAngles+=(vertices[i]-vec).getAngle(vertices[i+1]-vec);
                }
                sumAngles+=(vertices[vertices.size()-1]-vec).getAngle(vertices[0]-vec);
                //if ( !std::isnan(vertices) )
                    //std::cout << sumAngles << std::endl;
                if ( sumAngles > PI)
                    return false;
                else {
                    //std::cout << sumAngles << std::endl;
                    return true;
                }
            }
            return false;
        }

        __host__ __device__ bool isInPolygone(const Vector<double>& vec) {
            Vector<double> center = getBarycenter();
            if (isOnPlane(vec) && isOnPlane(center) ) {
                Line line0 = Line(center,vec-center);
                uint counter = 0;
                for (uint i=0;i<vertices.size();i++) {
                    Line line = Line(vertices[i],vertices[(i+1)%vertices.size()]-vertices[i]);
                    if (line0.IsIntersected(line))
                        counter++;
                }
                return (counter%2 == 0);
            }
            return false;
        }

        // Plane equation ax + by + cz + d = 0
        /*__host__ __device__ std::vector<double> getPlaneEquation() {
            Vector<double> normalVector = getNormalVector();
            std::vector<double> planeEq;
            planeEq.reserve(4);
            planeEq[0] = normalVector.getX(); // a
            planeEq[1] = normalVector.getY(); // b
            planeEq[2] = normalVector.getZ(); // c
            planeEq[3] = -(normalVector*(vertices_ptr[0])); // d
            return planeEq;
        }*/

        __host__ __device__ Vector<double> getIntersection(const Line line) {
            Vector<double> startingPoint = line.getPoint();
            Vector<double> direction = line.getDirection();
            Vector<double> normalVector = getNormalVector();
            if (std::abs(direction*normalVector) > 1E-5) {
                double k = (normalVector*(vertices[0]) - normalVector*startingPoint)/(direction*normalVector);
                Vector<double> intersectionPoint = startingPoint + direction*k;
                if (isInPolygone(intersectionPoint) && k>1E-7) {
                    for (uint i=0;i<vertices.size();i++) {
                        if (intersectionPoint==vertices[i])
                            return Vector<double>();
                    }
                    return intersectionPoint;
                }
            }
            return Vector<double>();
        }

        __host__ void print() const {
            for (uint i=0;i<vertices.size();i++) {
                std::cout << "Vector " << i+1 << " : ";
                vertices[i].printCoord();
            }
        }

        __host__ __device__ void move(const Vector<double>& vec) {
            for (uint i=0;i<vertices.size();i++) {
                vertices[i] += vec;
            }
        }
};

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

class Triangle {

    private:
        Vector<double> vertex0;
        Vector<double> vertex1;
        Vector<double> vertex2;
        const uint nbVertices = 3;
        Material material;

    public:
        __host__ __device__ Triangle() {};

        __host__ __device__ Triangle(const Triangle& tri) {
            vertex0 = tri.getVertex(0);
            vertex1 = tri.getVertex(1);
            vertex2 = tri.getVertex(2);

            material = tri.material;
        };

        __host__ __device__ Triangle(Material mat0) : Triangle() {
            material = mat0;
        };

        __host__ __device__ Triangle(Vector<double>& vec0, Material mat0) : Triangle() {
            vertex0 = vec0;
            material = mat0;
        };

        __host__ __device__ Triangle(Vector<double>& vec0, Pixel color0) : Triangle(vec0, Material(color0)) {};

        __host__ __device__ Vector<double> getVertex(const uint i) const {
            if (i == 0)
                return vertex0;
            else if (i == 1)
                return vertex1;
            else if (i == 2)
                return vertex2;
            else
                return Vector<double>();
        }

        __host__ __device__ void setvertex(const uint i, const Vector<double>& vec) {
            if (i == 0)
                vertex0 = vec;
            else if (i == 1)
                vertex1 = vec;
            else if (i == 2)
                vertex2 = vec;
        }

        __host__ __device__ uint getNbVertices() const {
            return nbVertices;
        }

        __host__ __device__ Material getMaterial() const {
            return material;
        }

        __host__ __device__ void setMaterial(const Material mat) {
            material=mat;
        }

        __host__ __device__ Vector<double> getNormalVector() const {
            return (vertex1-vertex0).crossProduct(vertex2-vertex0).normalize();
        }

        __host__ __device__ Vector<double> getBarycenter() const {
            return (vertex0 + vertex1 + vertex2)/3;
        }

        __host__ __device__ bool isPlaneValid() const {
            if (nbVertices < 3) {
                return false;
            } else {
                Vector<double> normalVector = getNormalVector();
                if (normalVector == Vector<double>())
                    return false;
                if (std::abs(normalVector*(vertex1-vertex0).normalize()) > 1E-3 || std::abs(normalVector*(vertex2-vertex0).normalize()) > 1E-3)
                    return false;
                return true;
            }
        }

        __host__ __device__ bool isOnPlane(const Vector<double>& vec) const {
            Vector<double> normalVector = getNormalVector();
            return std::abs( (vec-vertex0).normalize()*normalVector ) < 1E-6;
        }

        __host__ __device__ bool isInInterval(const double value, const double lhs, const double rhs) const {
            return lhs <= value && value <= rhs;
        }

        __host__ __device__ double triangleArea(const Vector<double>& v1, const Vector<double>& v2, const Vector<double>& v3) const {
            return (v2-v1).crossProduct(v3-v1).norm()/2;
        }

        __host__ __device__ bool isInPolygone(const Vector<double>& vec) const {
            if (isOnPlane(vec)) {
                if (nbVertices == 3) {
                    double area = triangleArea(vertex0, vertex1, vertex2);
                    double alpha = triangleArea(vec, vertex1, vertex2)/area;
                    double beta = triangleArea(vec, vertex2, vertex0)/area;
                    double gamma = triangleArea(vec, vertex0, vertex1)/area;
                    return isInInterval(alpha, 0., 1.) && isInInterval(beta, 0., 1.) && isInInterval(gamma, 0., 1.) && std::abs(alpha + beta + gamma - 1) < 1E-6;
                } else {
                    Vector<double> center = getBarycenter();
                    if (isOnPlane(center)) {
                        Line line0 = Line(center,vec-center);
                        uint counter = 0;

                        Line line = Line(vertex0,vertex1-vertex0);
                        if (line0.IsIntersected(line))
                            counter++;

                        line = Line(vertex1,vertex2-vertex1);
                        if (line0.IsIntersected(line))
                            counter++;

                        line = Line(vertex2,vertex0-vertex1);
                        if (line0.IsIntersected(line))
                            counter++;
                        return (counter%2 == 0);
                    }
                }
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

        __host__ __device__ Vector<double> getIntersection(const Line& line) const {
            Vector<double> startingPoint = line.getPoint();
            Vector<double> direction = line.getDirection();
            Vector<double> normalVector = getNormalVector();
            if (std::abs(direction*normalVector) > 1E-6) {
                double k = (normalVector*(vertex0) - normalVector*startingPoint)/(direction*normalVector);
                Vector<double> intersectionPoint = startingPoint + direction*k;
                if (isInPolygone(intersectionPoint) && k>1E-6) {
                    if (intersectionPoint==vertex0 || intersectionPoint==vertex1 || intersectionPoint==vertex2)
                        return Vector<double>();
                    return intersectionPoint;
                }
            }
            return Vector<double>();
        }

        __host__ void print() const {
            std::cout << "Vector 1 : ";
            vertex0.printCoord();
            std::cout << "Vector 2 : ";
            vertex1.printCoord();
            std::cout << "Vector 3 : ";
            vertex2.printCoord();
        }

        __host__ __device__ void move(const Vector<double>& vec) {
            vertex0 += vec;
            vertex1 += vec;
            vertex2 += vec;
        }

        __host__ __device__ Triangle operator=(const Triangle& tri) {
            if (this != &tri) {
                vertex0 = tri.getVertex(0);
                vertex1 = tri.getVertex(1);
                vertex2 = tri.getVertex(2);

                material = tri.material;
            }
            return *this;
        }
};

#pragma once
#include <iostream>
#include "Vector.hpp"
#include "Pixel.hpp"
#include "Line.hpp"
#include "Material.hpp"

#include <cuda_runtime.h>

#include <vector>
#include <cmath>

class Triangle {

    private:
        Vector<float> vertex0;
        Vector<float> vertex1;
        Vector<float> vertex2;
        Material material;

        Vector<float> mini;
        Vector<float> maxi;

    public:
        __host__ __device__ Triangle() {};

        __host__ __device__ Triangle(const Triangle& tri) {
            vertex0 = tri.getVertex(0);
            vertex1 = tri.getVertex(1);
            vertex2 = tri.getVertex(2);

            mini = min();
            maxi = max();

            material = tri.material;
        };

        __host__ __device__ Triangle(Material mat0) : Triangle() {
            material = mat0;
        };

        __host__ __device__ Triangle(Vector<float>& vec0, Material mat0) : Triangle() {
            vertex0 = vec0;
            material = mat0;
        };

        __host__ __device__ Triangle(Vector<float>& vec0, Pixel color0) : Triangle(vec0, Material(color0)) {};

        __host__ __device__ Vector<float> min() const {
            return vertex0.min(vertex1.min(vertex2));
        }

        __host__ __device__ Vector<float> max() const {
            return vertex0.max(vertex1.max(vertex2));
        }

        __host__ __device__ Vector<float> getMin() const {
            return mini;
        }

        __host__ __device__ Vector<float> getMax() const {
            return maxi;
        }

        __host__ __device__ Vector<float> getVertex(const uint i) const {
            if (i == 0)
                return vertex0;
            else if (i == 1)
                return vertex1;
            else if (i == 2)
                return vertex2;
            else
                return Vector<float>();
        }

        __host__ __device__ void setvertex(const uint i, const Vector<float>& vec) {
            if (i == 0)
                vertex0 = vec;
            else if (i == 1)
                vertex1 = vec;
            else if (i == 2)
                vertex2 = vec;

            mini = min();
            maxi = max();
        }

        __host__ __device__ Material getMaterial() const {
            return material;
        }

        __host__ __device__ void setMaterial(const Material mat) {
            material=mat;
        }

        __host__ __device__ Vector<float> getNormalVector() const {
            return (vertex1-vertex0).crossProduct(vertex2-vertex0).normalize();
        }

        __host__ __device__ Vector<float> getBarycenter() const {
            return (vertex0 + vertex1 + vertex2)/3;
        }

        __host__ __device__ bool isOnPlane(const Vector<float>& vec) const {
            Vector<float> normalVector = getNormalVector();
            return std::abs( (vec-vertex0).normalize()*normalVector ) < 1E-8;
        }

        __host__ __device__ float triangleArea(const Vector<float>& v1, const Vector<float>& v2, const Vector<float>& v3) const {
            return (v2-v1).crossProduct(v3-v1).norm()/2;
        }

        __host__ void print() const {
            std::cout << "Vector 1 : ";
            vertex0.printCoord();
            std::cout << "Vector 2 : ";
            vertex1.printCoord();
            std::cout << "Vector 3 : ";
            vertex2.printCoord();
        }

        __host__ __device__ void move(const Vector<float>& vec) {
            vertex0 += vec;
            vertex1 += vec;
            vertex2 += vec;

            mini = min();
            maxi = max();
        }

        __host__ __device__ Triangle operator=(const Triangle& tri) {
            if (this != &tri) {
                vertex0 = tri.getVertex(0);
                vertex1 = tri.getVertex(1);
                vertex2 = tri.getVertex(2);

                material = tri.material;

                mini = min();
                maxi = max();
            }
            return *this;
        }
};

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

        Vector<float> normal0;
        Vector<float> normal1;
        Vector<float> normal2;
        Material material;

        Vector<float> mini;
        Vector<float> maxi;

        __host__ __device__ Vector<float> min() const {
            return vertex0.min(vertex1.min(vertex2));
        }

        __host__ __device__ Vector<float> max() const {
            return vertex0.max(vertex1.max(vertex2));
        }

    public:
        __host__ __device__ Triangle() {};

        __host__ __device__ Triangle(const Triangle& tri) {
            vertex0 = tri.vertex0;
            vertex1 = tri.vertex1;
            vertex2 = tri.vertex2;

            normal0 = tri.normal0;
            normal1 = tri.normal1;
            normal2 = tri.normal2;

            mini = min();
            maxi = max();

            material = tri.material;
        };

        __host__ __device__ Triangle(Material mat0) {
            material = mat0;
        };

        __host__ __device__ Triangle(Vector<float>& vec0, Material mat0) {
            vertex0 = vec0;
            material = mat0;
        };

        __host__ __device__ Triangle(Vector<float>& vec0, Pixel color0) : Triangle(vec0, Material(color0)) {};

        __host__ __device__ Vector<float> getMin() const {
            return mini;
        }

        __host__ __device__ Vector<float> getMax() const {
            return maxi;
        }

        __host__ __device__ Vector<float> getVertex(const uint i) const {
            if (i == 0) return vertex0;
            else if (i == 1) return vertex1;
            else if (i == 2) return vertex2;
            else return Vector<float>();
        }

        __host__ __device__ void setvertex(const uint i, const Vector<float> vec) {
            if (i == 0) vertex0 = vec;
            else if (i == 1) vertex1 = vec;
            else if (i == 2) vertex2 = vec;

            mini = min();
            maxi = max();
        }

        __host__ __device__ Vector<float> getNormal(const uint i) const {
            if (i == 0) return normal0;
            else if (i == 1) return normal1;
            else if (i == 2) return normal2;
            else return Vector<float>();
        }

        __host__ __device__ void setNormal(const uint i, const Vector<float> vec) {
            if (i == 0) normal0 = vec.normalize();
            else if (i == 1) normal1 = vec.normalize();
            else if (i == 2) normal2 = vec.normalize();
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

        __host__ __device__ Vector<float> getNormalVector(const float u, const float v, const float w) const {
            if (normal0 == Vector<float>() || normal1 == Vector<float>() || normal2 == Vector<float>())
                return getNormalVector();
            return (normal0*w + normal1*u + normal2*v).normalize();
        }

        __host__ __device__ Vector<float> getNormalVector(const Vector<float> intersection) const {
            //if (normal0 == Vector<float>() || normal1 == Vector<float>() || normal2 == Vector<float>())
            //    return getNormalVector();
            Vector<float> dists = Vector<float>((vertex0-intersection).norm(), (vertex1-intersection).norm(), (vertex2-intersection).norm());
            dists /= dists.sum();
            dists = -dists + 1;
            return (normal0*dists[0] + normal1*dists[1] + normal2*dists[2]).normalize();
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
            std::cout << " - Normal 1 : ";
            normal0.printCoord();
            std::cout << "Vector 2 : ";
            vertex1.printCoord();
            std::cout << " - Normal 2 : ";
            normal1.printCoord();
            std::cout << "Vector 3 : ";
            vertex2.printCoord();
            std::cout << " - Normal 3 : ";
            normal2.printCoord();
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

                normal0 = tri.getNormal(0);
                normal1 = tri.getNormal(1);
                normal2 = tri.getNormal(2);

                material = tri.material;

                mini = min();
                maxi = max();
            }
            return *this;
        }
};

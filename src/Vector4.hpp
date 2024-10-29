#pragma once
#include <iostream>
#include <cstring>
#include <cmath>

#include "Vector.hpp"
#include "utils/MinMax.hpp"

#include <cuda_runtime.h>

template<typename T>
class Vector4 {
    private:
        __host__ __device__ float fast_inverse_square_root(float number) const {
            uint32_t i;
            float x2, y0;
            const float threehalfs = 1.5F;

            x2 = number*0.5F;
            y0 = number;
            memcpy(&i, &y0, 4); // i=*(long*) &y0; //evil floating point bit hack
            i = 0x5f3759df - (i >> 1); //what the fuck ?
            memcpy(&y0, &i, 4); // y0=*(float*) &i;
            y0 = y0*(threehalfs - (x2*y0*y0)); // Newton's method
            y0 = y0*(threehalfs - (x2*y0*y0)); // Newton's method again
            return y0;
        }
        
    public:
        T v1;
        T v2;
        T v3;
        T v4;

        __host__ __device__ Vector4() : v1((T)0), v2((T)0), v3((T)0), v4((T)0) {};
        __host__ __device__ Vector4(T _v1, T _v2, T _v3, T _v4) : v1(_v1), v2(_v2), v3(_v3), v4(_v4) {};
        template<typename U>
        __host__ __device__ Vector4(const Vector4<U>& vec) : v1((T)vec.v1), v2((T)vec.v2), v3((T)vec.v3), v4((T)vec.v4) {};
        template<typename U, typename W>
        __host__ __device__ Vector4(const Vector<U>& vec, W val) : v1((T)vec.getX()), v2((T)vec.getY()), v3((T)vec.getZ()), v4((T)val) {};
        __host__ __device__ ~Vector4(){};

        __host__ __device__ Vector<T> toVector() const {
            return Vector<T>(v1, v2, v3);
        }

        __host__ __device__ int size() const {
            return 4;
        }

        __host__ __device__ T operator[](const uint i) {
            switch (i) {
                case 0:
                    return v1;
                case 1:
                    return v2;
                case 2:
                    return v3;
                case 3:
                    return v4;
                default:
                    return -1;
            }
        }

        __host__ void printCoord() const {
            std::cout << std::fixed;
            std::cout.precision(2);
            std::cout << "("
                        << v1
                        << ";"
                        << v2
                        << ";"
                        << v3
                        << ";"
                        << v4
                        << ")"
                        << std::endl;
        }

        __host__ __device__ void printCoordDevice() const {
            printf("(%f, %f, %f, %f)\n", (double)v1, (double)v2, (double)v3, (double)v4);
        }

        __host__ __device__ Vector4<T> invCoords() const {
            return Vector4<T>(1.f/v1, 1.f/v2, 1.f/v3, 1.f/v4);
        }

        __host__ __device__ T normSquared() const {
            return v1*v1 + v2*v2 + v3*v3 + v4*v4;
        }

        __host__ __device__ T norm() const {
            return std::sqrt(v1*v1 + v2*v2 + v3*v3 + v4*v4);
        }

        __host__ __device__ Vector4<T> normalize() {
            const float invNorm = fast_inverse_square_root(normSquared());
            v1*=invNorm;
            v2*=invNorm;
            v3*=invNorm;
            v4*=invNorm;
            return *this;
        }

        __host__ __device__ Vector4<T> normalize() const {
            const float invNorm = fast_inverse_square_root(normSquared());
            return Vector4<T>(v1*invNorm, v2*invNorm, v3*invNorm, v4*invNorm);
        }

        template<typename U>
        __host__ __device__ Vector4& operator= (const Vector4<U>& vec) {
            if (this != &vec) {
                v1 = static_cast<T>(vec.v1);
                v2 = static_cast<T>(vec.v2);
                v3 = static_cast<T> (vec.v3);
                v4 = static_cast<T> (vec.v4);
            }
            return *this;
        }

        __host__ __device__ Vector4<T> operator+ (const Vector4<T>& vec) const {
            return Vector4<T>(v1+vec.v1, v2+vec.v2, v3+vec.v3, v4+vec.v4);
        }

        __host__ __device__ Vector4<T> operator+ (const T number) const {
            return Vector4<T>(v1+number, v2+number, v3+number, v4+number);
        }

        __host__ __device__ Vector4<T> operator- (const Vector4<T>& vec) const {
            return Vector4<T>(v1-vec.v1, v2-vec.v2, v3-vec.v3, v4-vec.v4);
        }

        __host__ __device__ Vector4<T> operator- (const T number) const {
            return Vector4<T>(v1-number, v2-number, v3-number, v4-number);
        }

        __host__ __device__ Vector4<T> operator- () const {
            return Vector4<T>(-v1,-v2,-v3,-v4);
        }

        __host__ __device__ Vector4<T> operator* (const T number) const {
            return Vector4<T>(v1*number, v2*number, v3*number, v4*number);
        }

        __host__ __device__ Vector4<T> operator/ (const T number) const {
            return Vector4<T>(v1/number, v2/number, v3/number, v4/number);
        }

        __host__ __device__ Vector4<T>& operator+= (const T nb) {
            v1 += nb;
            v2 += nb;
            v3 += nb;
            v4 += nb;
            return *this;
        }

        __host__ __device__ Vector4<T>& operator+= (const Vector4<T>& vec) {
            v1 += vec.v1;
            v2 += vec.v2;
            v3 += vec.v3;
            v4 += vec.v4;
            return *this;
        }

        __host__ __device__ Vector4<T>& operator-= (const T nb) {
            v1 -= nb;
            v2 -= nb;
            v3 -= nb;
            v4 -= nb;
            return *this;
        }

        __host__ __device__ Vector4<T>& operator-= (const Vector4<T>& vec) {
            v1 -= vec.v1;
            v2 -= vec.v2;
            v3 -= vec.v3;
            v4 -= vec.v4;
            return *this;
        }

        __host__ __device__ Vector4<T>& operator*= (const T nb) {
            v1 *= nb;
            v2 *= nb;
            v3 *= nb;
            v4 *= nb;
            return *this;
        }

        __host__ __device__ Vector4<T>& operator/= (const T nb) {
            v1 /= nb;
            v2 /= nb;
            v3 /= nb;
            v4 /= nb;
            return *this;
        }

        __host__ __device__ T operator * (const Vector4<T>& vec) const {
            return v1*vec.v1 + v2*vec.v2 + v3*vec.v3 + v4*vec.v4;
        }

        __host__ __device__ Vector4<T>& pow(const T nb) {
            v1 = std::pow(v1,nb);
            v2 = std::pow(v2,nb);
            v3 = std::pow(v3,nb);
            v4 = std::pow(v4,nb);
            return *this;
        }

        template <typename U>
        __host__ __device__ bool operator == (const Vector4<U>& vec) const {
            if (std::is_same<T,U>::value) {
                return std::abs(v1-vec.v1) < 1E-3f && std::abs(v2-vec.v2) < 1E-3f && std::abs(v3-vec.v3) < 1E-3f && std::abs(v4-vec.v4) < 1E-3f;
            }
            return false;
        }

        template <typename U>
        __host__ __device__ bool operator != (const Vector4<U>& vec) const {
            return !(*this==vec);
        }

        template <typename U>
        __host__ __device__ Vector4<U> productTermByTerm(const Vector4<U>& vec2) const {
            if (std::is_same<T,U>::value) {
                return Vector4(v1*vec2.v1, v2*vec2.v2, v3*vec2.v3, v4*vec2.v4);
            }
        }

        /*
        template <typename U>
        __host__ __device__ Vector4<U> crossProduct(const Vector4<U>& vec2) const {
            if (std::is_same<T,U>::value) {
                return Vector4(v2*vec2.v3 - v3*vec2.v2, v3*vec2.v1 - v1*vec2.v3, v1*vec2.v2 - v2*vec2.v1);
            }
        }
        */

        /*
        template <typename U>
        __host__ __device__ float getAngle(const Vector4<U>& vec2) {
            if (std::is_same<T,U>::value) {
                if (std::abs((*this).normalize()*vec2.normalize())>1) {
                    return 0.;
                }
                return std::acos( (*this).normalize()*vec2.normalize() );
            }
        }*/

        __host__ __device__ Vector4<T> max(const Vector4<T>& vec2) const {
            return Vector4<T>(Utils::max(v1, vec2.v1), Utils::max(v2, vec2.v2), Utils::max(v3, vec2.v3), Utils::max(v4, vec2.v4));
        }

        __host__ __device__ T max() const {
            return Utils::max(v1, Utils::max(v2, Utils::max(v3, v4)));
        }

        __host__ __device__ Vector4<T> min(const Vector4<T>& vec2) const {
            return Vector4<T>(Utils::min(v1, vec2.v1), Utils::min(v2, vec2.v2), Utils::min(v3, vec2.v3), Utils::min(v4, vec2.v4));
        }

        __host__ __device__ T min() const {
            return Utils::min(v1, Utils::min(v2, Utils::min(v3, v4)));
        }

        __host__ __device__ T sum() const {
            return v1+v2+v3+v4;
        }

        __host__ __device__ T mean() const {
            return sum()/size();
        }

        __host__ __device__ Vector4<T> lerp(const Vector4<T>& vec2, const float percentage) const {
            return ((*this)*(1-percentage) + vec2*percentage);
        }

        __host__ __device__ Vector4<T>& clamp(const T min, const T max) {
            *this = (*this).min(Vector4<T>(max, max, max, max));
            *this = (*this).max(Vector4<T>(min, min, min, min));
            return *this;
        }
};

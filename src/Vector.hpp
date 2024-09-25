#pragma once
#include <iostream>
#include <cstring>
#include <cmath>

#include "utils/MinMax.hpp"

#include <cuda_runtime.h>

template<typename T>
class Vector {

    private:
        T x;
        T y;
        T z;

        __host__ __device__ float fast_inverse_square_root(float number) const {
            uint32_t i;
            float x2, y0;
            const float threehalfs = 1.5F;

            x2 = number*0.5F;
            y0=number;
            memcpy(&i, &y0, 4); // i=*(long*) &y0; //evil floating point bit hack
            i = 0x5f3759df - (i >> 1); //what the fuck ?
            memcpy(&y0, &i, 4); // y0=*(float*) &i;
            y0=y0*(threehalfs - (x2*y0*y0)); // Newton's method
            y0=y0*(threehalfs - (x2*y0*y0)); // Newton's method again
            return y0;
        }
        
    public:
        __host__ __device__ Vector() : x((T)0), y((T)0), z((T)0) {};
        __host__ __device__ Vector(T x0, T y0, T z0) : x(x0), y(y0), z(z0) {};
        template<typename U>
        __host__ __device__ Vector(const Vector<U>& vec) : x((T)vec.getX()), y((T)vec.getY()), z((T)vec.getZ()) {};
        __host__ __device__ ~Vector(){};

        __host__ __device__ T getX() const {
            return x;
        }
        __host__ __device__ T getY() const {
            return y;
        }
        __host__ __device__ T getZ() const {
            return z;
        }
        __host__ __device__ int size() const {
            return 3;
        }

        __host__ __device__ T operator[](const uint i) {
            switch (i) {
                case 0:
                    return x;
                case 1:
                    return y;
                case 2:
                    return z;
                default:
                    return -1;
            }
        }

        __host__ void printCoord() const {
            std::cout << "("
                        << x
                        << ";"
                        << y
                        << ";"
                        << z
                        << ")"
                        << std::endl;
        }

        __host__ __device__ Vector<T> invCoords() const {
            return Vector<T>(1./x, 1./y, 1./z);
        }

        __host__ __device__ T normSquared() const {
            return x*x + y*y + z*z;
        }

        __host__ __device__ T norm() const {
            return std::sqrt(x*x + y*y + z*z);
        }

        __host__ __device__ Vector normalize() {
            const float invNorm = fast_inverse_square_root(normSquared());
            x*=invNorm;
            y*=invNorm;
            z*=invNorm;
            return *this;
        }

        __host__ __device__ Vector normalize() const {
            const float invNorm = fast_inverse_square_root(normSquared());
            return Vector<T>(x*invNorm, y*invNorm, z*invNorm);
        }

        template<typename U>
        __host__ __device__ Vector& operator= (const Vector<U>& vec) {
            if (this != &vec) {
                x = static_cast<T>(vec.x);
                y = static_cast<T>(vec.y);
                z = static_cast<T> (vec.z);
            }
            return *this;
        }

        __host__ __device__ Vector<T> operator+ (const Vector<T>& vec) const {
            return Vector<T>(x+vec.x,y+vec.y,z+vec.z);
        }

        __host__ __device__ Vector<T> operator+ (const T number) const {
            return Vector<T>(x+number,y+number,z+number);
        }

        __host__ __device__ Vector<T> operator- (const Vector<T>& vec) const {
            return Vector<T>(x-vec.x,y-vec.y,z-vec.z);
        }

        __host__ __device__ Vector<T> operator- (const T number) const {
            return Vector<T>(x-number,y-number,z-number);
        }

        __host__ __device__ Vector<T> operator- () const {
            return Vector<T>(-x,-y,-z);
        }

        __host__ __device__ Vector<T> operator* (const T number) const {
            return Vector<T>(x*number,y*number,z*number);
        }

        __host__ __device__ Vector<T> operator/ (const T number) const {
            return Vector<T>(x/number,y/number,z/number);
        }

        __host__ __device__ Vector<T>& operator+= (const T nb) {
            x += nb;
            y += nb;
            z += nb;
            return *this;
        }

        __host__ __device__ Vector<T>& operator+= (const Vector<T>& vec) {
            x += vec.x;
            y += vec.y;
            z += vec.z;
            return *this;
        }

        __host__ __device__ Vector<T>& operator-= (const T nb) {
            x -= nb;
            y -= nb;
            z -= nb;
            return *this;
        }

        __host__ __device__ Vector<T>& operator-= (const Vector<T>& vec) {
            x -= vec.x;
            y -= vec.y;
            z -= vec.z;
            return *this;
        }

        __host__ __device__ Vector<T>& operator*= (const T nb) {
            x *= nb;
            y *= nb;
            z *= nb;
            return *this;
        }

        __host__ __device__ Vector<T>& operator/= (const T nb) {
            x /= nb;
            y /= nb;
            z /= nb;
            return *this;
        }

        __host__ __device__ T operator * (const Vector<T>& vec) const {
            return x*vec.x + y*vec.y + z*vec.z;
        }

        __host__ __device__ Vector<T>& pow(const T nb) {
            x = std::pow(x,nb);
            y = std::pow(y,nb);
            z = std::pow(z,nb);
            return *this;
        }

        template <typename U>
        __host__ __device__ bool operator == (const Vector<U>& vec) const {
            if (std::is_same<T,U>::value) {
                return std::abs(x-vec.getX()) < 1E-3 && std::abs(y-vec.getY()) < 1E-3 && std::abs(z-vec.getZ()) < 1E-3;
            }
            return false;
        }

        template <typename U>
        __host__ __device__ bool operator != (const Vector<U>& vec) const {
            return !(*this==vec);
        }

        template <typename U>
        __host__ __device__ Vector<U> productTermByTerm(const Vector<U>& vec2) const {
            if (std::is_same<T,U>::value) {
                return Vector(x*vec2.x, y*vec2.y, z*vec2.z);
            }
        }

        template <typename U>
        __host__ __device__ Vector<U> crossProduct(const Vector<U>& vec2) const {
            if (std::is_same<T,U>::value) {
                return Vector(y*vec2.z - z*vec2.y,z*vec2.x - x*vec2.z,x*vec2.y - y*vec2.x);
            }
        }

        template <typename U>
        __host__ __device__ float getAngle(const Vector<U>& vec2) {
            if (std::is_same<T,U>::value) {
                if (std::abs((*this).normalize()*vec2.normalize())>1) {
                    return 0.;
                }
                return std::acos( (*this).normalize()*vec2.normalize() );
            }
        }

        __host__ __device__ Vector<T> max(const Vector<T>& vec2) const {
            return Vector<T>(Utils::max(x, vec2.x), Utils::max(y, vec2.y), Utils::max(z, vec2.z));
        }

        __host__ __device__ T max() const {
            return Utils::max(x, Utils::max(y, z));
        }

        __host__ __device__ Vector<T> min(const Vector<T>& vec2) const {
            return Vector<T>(Utils::min(x, vec2.x), Utils::min(y, vec2.y), Utils::min(z, vec2.z));
        }

        __host__ __device__ T min() const {
            return Utils::min(x, Utils::min(y, z));
        }

        __host__ __device__ Vector<T> lerp(const Vector<T>& vec2, const float percentage) const {
            return ((*this)*(1-percentage) + vec2*percentage);
        }

        __host__ __device__ void clamp(const T min, const T max) {
            *this = (*this).min(Vector<T>(max, max, max));
            *this = (*this).max(Vector<T>(min, min, min));
        }
};

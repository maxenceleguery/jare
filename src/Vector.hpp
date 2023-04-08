#pragma once
#include <iostream>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>

template<typename T>
class Vector {

    private:
        T x;
        T y;
        T z;

        __host__ __device__ double fast_inverse_square_root(float number) const {
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
            return (double)y0;
        }
        
    public:
        __host__ __device__ Vector() : x((T)0), y((T)0), z((T)0) {};
        __host__ __device__ Vector(T x0, T y0, T z0) : x(x0), y(y0), z(z0) {};
        //explicit Vector(Pixel pixel) : x(pixel.getR()), y(pixel.getG()), z(pixel.getB()) {};
        //Vector(Vector<T>& vec) : x(vec.x), y(vec.y), z(vec.z) {};
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

        __host__ __device__ T normSquared() const {
            return x*x + y*y + z*z;
        }

        __host__ __device__ Vector<T> normalize() {
            double invNorm = fast_inverse_square_root(normSquared());
            this->x*=invNorm;
            this->y*=invNorm;
            this->z*=invNorm;
            return *this;
        }

        __host__ __device__ Vector<T> normalize(const Vector<T>& vec) {
            Vector<T> result = Vector(vec);
            double invNorm = fast_inverse_square_root(normSquared());
            result.x*=invNorm;
            result.y*=invNorm;
            result.z*=invNorm;
            return result;
        }

        template<typename U>
        __host__ __device__ Vector& operator = (const Vector<U>& vec) {
            this->x = static_cast<T>(vec.x);
            this->y = static_cast<T>(vec.y);
            this->z = static_cast<T> (vec.z);
            return *this;
        }

        __host__ __device__ Vector<T> operator + (const Vector<T>& vec) const {
            return Vector<T>(x+vec.x,y+vec.y,z+vec.z);
        }

        __host__ __device__ Vector<T> operator- (const Vector<T>& vec) const {
            return Vector<T>(x-vec.x,y-vec.y,z-vec.z);
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
            this->x += nb;
            this->y += nb;
            this->z += nb;
            return *this;
        }

        __host__ __device__ Vector<T>& operator+= (const Vector<T>& vec) {
            this->x += vec.x;
            this->y += vec.y;
            this->z += vec.z;
            return *this;
        }

        __host__ __device__ Vector<T>& operator-= (const T nb) {
            this->x -= nb;
            this->y -= nb;
            this->z -= nb;
            return *this;
        }

        __host__ __device__ Vector<T>& operator-= (const Vector<T>& vec) {
            this->x -= vec.x;
            this->y -= vec.y;
            this->z -= vec.z;
            return *this;
        }

        __host__ __device__ Vector<T>& operator*= (const T nb) {
            this->x *= nb;
            this->y *= nb;
            this->z *= nb;
            return *this;
        }

        __host__ __device__ Vector<T>& operator/= (const T nb) {
            this->x /= nb;
            this->y /= nb;
            this->z /= nb;
            return *this;
        }

        __host__ __device__ T operator * (const Vector<T>& vec) const {
            return x*vec.x + y*vec.y + z*vec.z;
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
                return Vector(x*vec2.getX(), y*vec2.getY(), z*vec2.getZ());
            }
        }

        template <typename U>
        __host__ __device__ Vector<U> crossProduct(const Vector<U>& vec2) const {
            if (std::is_same<T,U>::value) {
                return Vector(y*vec2.getZ() - z*vec2.getY(),z*vec2.getX() - x*vec2.getZ(),x*vec2.getY() - y*vec2.getX());
            }
        }

        template <typename U>
        __host__ __device__ double getAngle(const Vector<U>& vec2) {
            if (std::is_same<T,U>::value) {
                if (std::abs(normalize(*this)*normalize(vec2))>1) {
                    return 0.;
                }
                return std::acos( normalize(*this)*normalize(vec2) );
            }
        }
};

#pragma once

#include <cmath>
#include <cuda_runtime.h>

class Quaternion {
    private:

    public:
        float a, b, c, d;

        __host__ __device__ Quaternion() {};
        __host__ __device__ Quaternion(const float a, const float b, const float c, const float d) : a(a), b(b), c(c), d(d) {};
        __host__ __device__ ~Quaternion() {};

        __host__ __device__ Quaternion operator+ (const Quaternion& q) const {
            return Quaternion(a+q.a, b+q.b, c+q.c, d+q.d);
        }

        __host__ __device__ Quaternion operator+ (const float number) const {
            return Quaternion(a+number, b, c, d);
        }

        __host__ __device__ Quaternion operator- (const Quaternion& q) const {
            return Quaternion(a-q.a, b-q.b, c-q.c, d-q.d);
        }

        __host__ __device__ Quaternion operator- (const float number) const {
            return Quaternion(a-number, b, c, d);
        }

        __host__ __device__ Quaternion operator* (const Quaternion& q) const {
            return Quaternion(
                a*q.a - b*q.b - c*q.c - d*q.d,
                a*q.b + b*q.a + c*q.d - d*q.c,
                a*q.c + c*q.a + d*q.b - b*q.d,
                a*q.d + d*q.a + b*q.c - c*q.b
            );
        }

        __host__ __device__ Quaternion operator/ (const Quaternion& q) const {
            return (*this)*q.inverse();
        }

        __host__ __device__ Quaternion operator* (const float number) const {
            return Quaternion(a*number, b*number, c*number, d*number);
        }

        __host__ __device__ Quaternion operator/ (const float number) const {
            return Quaternion(a/number, b/number, c/number, d/number);
        }

        __host__ __device__ bool operator == (const Quaternion& q) const {
            return std::abs(a-q.a) < 1E-3f && std::abs(b-q.b) < 1E-3f && std::abs(c-q.c) < 1E-3f && std::abs(d-q.d) < 1E-3f;
        }

        __host__ __device__ bool operator != (const Quaternion& q) const {
            return !(*this==q);
        }

        __host__ __device__ float normSquared() const {
            return a*a + b*b + c*c + d*d;
        }

        __host__ __device__ Quaternion conj() const {
            return Quaternion(a, -b, -c, -d);
        }

        __host__ __device__ Quaternion inverse() const {
            return conj()/normSquared();
        }
};


#pragma once

#include <cuda_runtime.h>

#include "Vector.hpp"
#include "Vector4.hpp"

class Matrix4x4 {
    private:
        Vector4<float> line1;
        Vector4<float> line2;
        Vector4<float> line3;
        Vector4<float> line4;

    public:
        __host__ __device__ Matrix4x4() : line1(1, 0, 0, 0), line2(0, 1, 0, 0), line3(0, 0, 1, 0), line4(0, 0, 0, 1) {};
        __host__ __device__ Matrix4x4(const Vector4<float>& vec1, const Vector4<float>& vec2, const Vector4<float>& vec3, const Vector4<float>& vec4)
        : line1(vec1), line2(vec2), line3(vec3), line4(vec4) {};
        __host__ __device__ ~Matrix4x4() {};

        __host__ __device__ Matrix4x4 operator+ (const Matrix4x4& mat) const {
            return Matrix4x4(line1+mat.line1, line2+mat.line2, line3+mat.line3, line4+mat.line4);
        }

        __host__ __device__ Matrix4x4 operator+ (const float number) const {
            return Matrix4x4(line1+number, line2+number, line3+number, line4+number);
        }

        __host__ __device__ Matrix4x4 operator- (const Matrix4x4& mat) const {
            return Matrix4x4(line1-mat.line1, line2-mat.line2, line3-mat.line3, line4-mat.line4);
        }

        __host__ __device__ Matrix4x4 operator- (const float number) const {
            return Matrix4x4(line1-number, line2-number, line3-number, line4-number);
        }

        __host__ __device__ Vector4<float> operator* (const Vector4<float>& vec) const {
            return Vector4<float>(
                line1*vec,
                line2*vec,
                line3*vec,
                line4*vec
            );
        }

        __host__ __device__ Vector<float> operator* (const Vector<float>& vec) const {
            return Vector<float>(
                line1.v1*vec.getX() + line1.v2*vec.getY() + line1.v3*vec.getZ() + line1.v4,
                line2.v1*vec.getX() + line2.v2*vec.getY() + line2.v3*vec.getZ() + line2.v4,
                line3.v1*vec.getX() + line3.v2*vec.getY() + line3.v3*vec.getZ() + line3.v4
            );
        }

        __host__ __device__ Matrix4x4 operator* (const Matrix4x4& mat) const {
            Matrix4x4 matT = mat.transpose();
            return Matrix4x4(
                Vector4<float>(line1*matT.line1, line1*matT.line2, line1*matT.line3, line1*matT.line4),
                Vector4<float>(line2*matT.line1, line2*matT.line2, line2*matT.line3, line2*matT.line4),
                Vector4<float>(line3*matT.line1, line3*matT.line2, line3*matT.line3, line3*matT.line4),
                Vector4<float>(line4*matT.line1, line4*matT.line2, line4*matT.line3, line4*matT.line4)
            );
        }

        /*
        __host__ __device__ Matrix4x4 operator/ (const Matrix4x4& mat) const {
            return (*this)*mat.inverse();
        }*/

        __host__ __device__ Matrix4x4 operator* (const float number) const {
            return Matrix4x4(line1*number, line2*number, line3*number, line4*number);
        }

        __host__ __device__ Matrix4x4 operator/ (const float number) const {
            return Matrix4x4(line1/number, line2/number, line3/number, line4/number);
        }

        __host__ __device__ bool operator == (const Matrix4x4& mat) const {
            return line1 == mat.line1 && line2 == mat.line2 && line3 == mat.line3 && line4 == mat.line4;
        }

        __host__ __device__ bool operator != (const Matrix4x4& mat) const {
            return !(*this==mat);
        }

        __host__ __device__ Matrix4x4 transpose() const {
            return Matrix4x4(
                Vector4<float>(line1.v1, line2.v1, line3.v1, line4.v1),
                Vector4<float>(line1.v2, line2.v2, line3.v2, line4.v2),
                Vector4<float>(line1.v3, line2.v3, line3.v3, line4.v3),
                Vector4<float>(line1.v4, line2.v4, line3.v4, line4.v4)
            );
        }

        __host__ __device__ float det() const {
            return line1.v1 * (line2.v2 * (line3.v3 * line4.v4 - line4.v3 * line3.v4) -
                            line2.v3 * (line3.v2 * line4.v4 - line4.v2 * line3.v4) +
                            line2.v4 * (line3.v2 * line4.v3 - line4.v2 * line3.v3)) -
                line1.v2 * (line2.v1 * (line3.v3 * line4.v4 - line4.v3 * line3.v4) -
                            line2.v3 * (line3.v1 * line4.v4 - line4.v1 * line3.v4) +
                            line2.v4 * (line3.v1 * line4.v3 - line4.v1 * line3.v3)) +
                line1.v3 * (line2.v1 * (line3.v2 * line4.v4 - line4.v2 * line3.v4) -
                            line2.v2 * (line3.v1 * line4.v4 - line4.v1 * line3.v4) +
                            line2.v4 * (line3.v1 * line4.v2 - line4.v1 * line3.v2)) -
                line1.v4 * (line2.v1 * (line3.v2 * line4.v3 - line4.v2 * line3.v3) -
                            line2.v2 * (line3.v1 * line4.v3 - line4.v1 * line3.v3) +
                            line2.v3 * (line3.v1 * line4.v2 - line4.v1 * line3.v2));
        }

        __host__ __device__ Matrix4x4 inverse() const {
            // Calculate the determinant
            float deter = det();

            // Check if the determinant is zero (matrix is not invertible)
            if (deter == 0) {
                // Handle the non-invertible matrix case (e.g., throw an exception or return an identity matrix)
                return Matrix4x4(); // Returning a default (identity) matrix for simplicity
            }

            float invDet = 1.0f / deter;

            // Calculate the inverse matrix using the adjugate method
            Matrix4x4 inverseMat(
                Vector4<float>(
                    (line2.v2 * (line3.v3 * line4.v4 - line4.v3 * line3.v4) -
                    line2.v3 * (line3.v2 * line4.v4 - line4.v2 * line3.v4) +
                    line2.v4 * (line3.v2 * line4.v3 - line4.v2 * line3.v3)) * invDet,
                    -(line1.v2 * (line3.v3 * line4.v4 - line4.v3 * line3.v4) -
                    line1.v3 * (line3.v2 * line4.v4 - line4.v2 * line3.v4) +
                    line1.v4 * (line3.v2 * line4.v3 - line4.v2 * line3.v3)) * invDet,
                    (line1.v2 * (line2.v3 * line4.v4 - line4.v3 * line2.v4) -
                    line1.v3 * (line2.v2 * line4.v4 - line4.v2 * line2.v4) +
                    line1.v4 * (line2.v2 * line4.v3 - line4.v2 * line2.v3)) * invDet,
                    -(line1.v2 * (line2.v3 * line3.v4 - line3.v3 * line2.v4) -
                    line1.v3 * (line2.v2 * line3.v4 - line3.v2 * line2.v4) +
                    line1.v4 * (line2.v2 * line3.v3 - line3.v2 * line2.v3)) * invDet),
                Vector4<float>(
                    -(line2.v1 * (line3.v3 * line4.v4 - line4.v3 * line3.v4) -
                    line2.v3 * (line3.v1 * line4.v4 - line4.v1 * line3.v4) +
                    line2.v4 * (line3.v1 * line4.v3 - line4.v1 * line3.v3)) * invDet,
                    (line1.v1 * (line3.v3 * line4.v4 - line4.v3 * line3.v4) -
                    line1.v3 * (line3.v1 * line4.v4 - line4.v1 * line3.v4) +
                    line1.v4 * (line3.v1 * line4.v3 - line4.v1 * line3.v3)) * invDet,
                    -(line1.v1 * (line2.v3 * line4.v4 - line4.v3 * line2.v4) -
                    line1.v3 * (line2.v1 * line4.v4 - line4.v1 * line2.v4) +
                    line1.v4 * (line2.v1 * line4.v3 - line4.v1 * line2.v3)) * invDet,
                    (line1.v1 * (line2.v3 * line3.v4 - line3.v3 * line2.v4) -
                    line1.v3 * (line2.v1 * line3.v4 - line3.v1 * line2.v4) +
                    line1.v4 * (line2.v1 * line3.v3 - line3.v1 * line2.v3)) * invDet),
                Vector4<float>(
                    (line2.v1 * (line3.v2 * line4.v4 - line4.v2 * line3.v4) -
                    line2.v2 * (line3.v1 * line4.v4 - line4.v1 * line3.v4) +
                    line2.v4 * (line3.v1 * line4.v2 - line4.v1 * line3.v2)) * invDet,
                    -(line1.v1 * (line3.v2 * line4.v4 - line4.v2 * line3.v4) -
                    line1.v2 * (line3.v1 * line4.v4 - line4.v1 * line3.v4) +
                    line1.v4 * (line3.v1 * line4.v2 - line4.v1 * line3.v2)) * invDet,
                    (line1.v1 * (line2.v2 * line4.v4 - line4.v2 * line2.v4) -
                    line1.v2 * (line2.v1 * line4.v4 - line4.v1 * line2.v4) +
                    line1.v4 * (line2.v1 * line4.v2 - line4.v1 * line2.v2)) * invDet,
                    -(line1.v1 * (line2.v2 * line3.v4 - line3.v2 * line2.v4) -
                    line1.v2 * (line2.v1 * line3.v4 - line3.v1 * line2.v4) +
                    line1.v4 * (line2.v1 * line3.v2 - line3.v1 * line2.v2)) * invDet),
                Vector4<float>(
                    -(line2.v1 * (line3.v2 * line4.v3 - line4.v2 * line3.v3) -
                    line2.v2 * (line3.v1 * line4.v3 - line4.v1 * line3.v3) +
                    line2.v3 * (line3.v1 * line4.v2 - line4.v1 * line3.v2)) * invDet,
                    (line1.v1 * (line3.v2 * line4.v3 - line4.v2 * line3.v3) -
                    line1.v2 * (line3.v1 * line4.v3 - line4.v1 * line3.v3) +
                    line1.v3 * (line3.v1 * line4.v2 - line4.v1 * line3.v2)) * invDet,
                    -(line1.v1 * (line2.v2 * line4.v3 - line4.v2 * line2.v3) -
                    line1.v2 * (line2.v1 * line4.v3 - line4.v1 * line2.v3) +
                    line1.v3 * (line2.v1 * line4.v2 - line4.v1 * line2.v2)) * invDet,
                    (line1.v1 * (line2.v2 * line3.v3 - line3.v2 * line2.v3) -
                    line1.v2 * (line2.v1 * line3.v3 - line3.v1 * line2.v3) +
                    line1.v3 * (line2.v1 * line3.v2 - line3.v1 * line2.v2)) * invDet)
            );

            return inverseMat;
        }

        __host__ void print() const {
            line1.printCoord();
            line2.printCoord();
            line3.printCoord();
            line4.printCoord();
        }
};


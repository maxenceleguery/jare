#pragma once
#include <iostream>
#include <cstring>
#include <cmath>

#include <cuda_runtime.h>

#define MATRIX_ZERO 0
#define MATRIX_EYE 1

template<typename T>
class Matrix {
    private:
        T a11; T a12; T a13;
        T a21; T a22; T a23;
        T a31; T a32; T a33;
        
    public:
        __host__ __device__ Matrix() : a11(0), a12(0), a13(0),
            a21(0), a22(0), a23(0),
            a31(0), a32(0), a33(0) {};
        __host__ __device__ Matrix(T coef11, T coef12, T coef13,
                T coef21, T coef22, T coef23,
                T coef31, T coef32, T coef33) :
            a11(coef11), a12(coef12), a13(coef13),
            a21(coef21), a22(coef22), a23(coef23),
            a31(coef31), a32(coef32), a33(coef33) {};

        __host__ __device__ Matrix(T diag,int type) {
            if (type == MATRIX_EYE) {
                a11=diag; a12=0.; a13=0.;
                a21=0.; a22=diag; a23=0.;
                a31=0.; a32=0.; a33=diag;
            } else if (type == MATRIX_ZERO) {
                a11=0.; a12=0.; a13=0.;
                a21=0.; a22=0.; a23=0.;
                a31=0.; a32=0.; a33=0.;
            }
        };
        __host__ __device__ Matrix(Vector<double> vec1, Vector<double> vec2, Vector<double> vec3) {
            a11=vec1.getX(); a12=vec2.getX(); a13=vec3.getX();
            a21=vec1.getY(); a22=vec2.getY(); a23=vec3.getY();
            a31=vec1.getZ(); a32=vec2.getZ(); a33=vec3.getZ();
        }
        __host__ __device__ ~Matrix(){};

        __host__ __device__ Matrix<T> operator + (const Matrix<T>& mat) const {
            Matrix<T> result;
            result.a11 = a11 + mat.a11;
            result.a12 = a12 + mat.a12;
            result.a13 = a13 + mat.a13;
            result.a21 = a21 + mat.a21;
            result.a22 = a22 + mat.a22;
            result.a23 = a23 + mat.a23;
            result.a31 = a31 + mat.a31;
            result.a32 = a32 + mat.a32;
            result.a33 = a33 + mat.a33;
            return result;
        }

        __host__ __device__ Matrix<T> operator - (const Matrix<T>& mat) const {
            Matrix<T> result;
            result.a11 = a11 - mat.a11;
            result.a12 = a12 - mat.a12;
            result.a13 = a13 - mat.a13;
            result.a21 = a21 - mat.a21;
            result.a22 = a22 - mat.a22;
            result.a23 = a23 - mat.a23;
            result.a31 = a31 - mat.a31;
            result.a32 = a32 - mat.a32;
            result.a33 = a33 - mat.a33;
            return result;
        }

        __host__ __device__ Matrix<T> operator * (const Matrix<T>& mat) const {
            Matrix<T> result;
            result.a11 = a11*mat.a11 + a12*mat.a21 + a13*mat.a31;
            result.a12 = a11*mat.a12 + a12*mat.a22 + a13*mat.a32;
            result.a13 = a11*mat.a13 + a12*mat.a23 + a13*mat.a33;
            result.a21 = a21*mat.a11 + a22*mat.a21 + a23*mat.a31;
            result.a22 = a21*mat.a12 + a22*mat.a22 + a23*mat.a32;
            result.a23 = a21*mat.a13 + a22*mat.a23 + a23*mat.a33;
            result.a31 = a31*mat.a11 + a32*mat.a21 + a33*mat.a31;
            result.a32 = a31*mat.a12 + a32*mat.a22 + a33*mat.a32;
            result.a33 = a31*mat.a13 + a32*mat.a23 + a33*mat.a33;
            return result;
        }

        __host__ __device__ Vector<T> operator * (const Vector<T>& vec) const {
            Vector<T> ligne1 = Vector(a11,a12,a13);
            Vector<T> ligne2 = Vector(a21,a22,a23);
            Vector<T> ligne3 = Vector(a31,a32,a33);
            return Vector<T>(ligne1*vec,ligne2*vec,ligne3*vec);
        }

        template<typename U>
        __host__ __device__ Matrix<T> operator * (U number) const {
            Matrix<T> result = Matrix(*this);
            result.a11 *= number;
            result.a12 *= number;
            result.a13 *= number;
            result.a21 *= number;
            result.a22 *= number;
            result.a23 *= number;
            result.a31 *= number;
            result.a32 *= number;
            result.a33 *= number;
            return result;
        }

        template<typename U>
        __host__ __device__ Matrix<T> operator / (U number) const {
            Matrix<T> result = Matrix(*this);
            result.a11 /= number;
            result.a12 /= number;
            result.a13 /= number;
            result.a21 /= number;
            result.a22 /= number;
            result.a23 /= number;
            result.a31 /= number;
            result.a32 /= number;
            result.a33 /= number;
            return result;
        }

        __host__ __device__ T operator [](int index) const {
            switch (index) {
            case 0:
                return a11;
            case 1:
                return a12;
            case 2:
                return a13;
            case 3:
                return a21;
            case 4:
                return a22;
            case 5:
                return a23;
            case 6:
                return a31;
            case 7:
                return a32;
            case 8:
                return a33;
            
            default:
                std::cout << "Index out of range" << std::endl;
                return -1;
            }
        }

        __host__ __device__ T trace() const {
            return a11+a22+a33;
        }

        __host__ __device__ Matrix<T> transpose() const {
            return Matrix<T>(a11,a21,a31,a12,a22,a32,a13,a23,a33);
        }

        __host__ __device__ T det() const {
            T det = 0;
            det += a11*(a22*a33 - a23*a32);
            det -= a21*(a12*a33 - a32*a13);
            det += a31*(a12*a23 - a22*a13);
            return det;
        }

        __host__ __device__ bool isInversible() const {
            return (std::abs(det()) > 1E-5);
        }

        __host__ __device__ Matrix<T> inverse() const {
            if ( isInversible() ) {
                Matrix<T> transp = transpose();
                Matrix<T> inv;
                inv.a11 = transp.a22*transp.a33 - transp.a23*transp.a32;
                inv.a12 = -(transp.a21*transp.a33 - transp.a31*transp.a23);
                inv.a13 = transp.a21*transp.a32 - transp.a31*transp.a22;
                inv.a21 = -(transp.a12*transp.a33 - transp.a32*transp.a13);
                inv.a22 = (transp.a11*transp.a33 - transp.a13*transp.a31);
                inv.a23 = -(transp.a11*transp.a32 - transp.a31*transp.a12);
                inv.a31 = (transp.a12*transp.a23 - transp.a22*transp.a13);
                inv.a32 = -(transp.a11*transp.a23 - transp.a21*transp.a13);
                inv.a33 = (transp.a11*transp.a22 - transp.a21*transp.a12);
                return inv/det();
            }
            return Matrix<T>();
        }

        __host__ void print() {
            std::cout << "|" << a11 << " " << a12 << " " << a13 << "|" << std::endl;
            std::cout << "|" << a21 << " " << a22 << " " << a23 << "|" << std::endl;
            std::cout << "|" << a31 << " " << a32 << " " << a33 << "|" << std::endl;
        }
};

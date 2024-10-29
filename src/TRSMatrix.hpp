#pragma once

#include "Matrix4x4.hpp"
#include "Transformations.hpp"

class TRSMatrix {
    private:
        Matrix4x4 TMatrix;
        Matrix4x4 RMatrix;
        Matrix4x4 SMatrix;
        Matrix4x4 TRSMatrix_precomputed;
        Matrix4x4 TRSMatrix_inv_precomputed;

        __host__ __device__ void update() {
            TRSMatrix_precomputed = TMatrix*RMatrix*SMatrix;
            TRSMatrix_inv_precomputed = TRSMatrix_precomputed.inverse();
        }

    public:
        __host__ __device__ TRSMatrix() {};
        __host__ __device__ TRSMatrix(const Vector<float>& offset, const Vector<float>& scale, const Vector<float>& rotation) {
            TMatrix = Transformations::GetTranslationMatrix(offset);
            RMatrix = Transformations::GetRotationMatrix(rotation);
            SMatrix = Transformations::GetScaleMatrix(scale);
            update();
        };
        __host__ __device__ ~TRSMatrix() {};

        __host__ __device__ Matrix4x4 getLocalToWorldMatrix() const {
            return TRSMatrix_precomputed;
        }

        __host__ __device__ Matrix4x4 getWorldToLocalMatrix() const {
            return TRSMatrix_inv_precomputed;
        }

        __host__ __device__ void setOffset(const Vector<float>& offset) {
            TMatrix = Transformations::GetTranslationMatrix(offset);
            update();
        }

        __host__ __device__ void addOffset(const Vector<float>& offset) {
            TMatrix = TMatrix*Transformations::GetTranslationMatrix(offset);
            update();
        }

        __host__ __device__ void setRotation(const Vector<float>& rotation) {
            RMatrix = Transformations::GetRotationMatrix(rotation);
            update();
        }

        __host__ __device__ void addRotation(const Vector<float>& rotation) {
            RMatrix = RMatrix*Transformations::GetRotationMatrix(rotation);
            update();
        }

        __host__ __device__ void setScale(const Vector<float>& scale) {
            SMatrix = Transformations::GetScaleMatrix(scale);
            update();
        }

        __host__ __device__ void addScale(const Vector<float>& scale) {
            SMatrix = SMatrix*Transformations::GetScaleMatrix(scale);
            update();
        }
};


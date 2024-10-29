#pragma once

#include "Triangle.hpp"
#include "utils/Array.hpp"
#include "utils/SceneObject.hpp"
#include "TRSMatrix.hpp"
#include "Transformations.hpp"

class Mesh : public Array<Triangle>, public SceneObject {
    private:
        //Array<TRSMatrix> transform_matrix;
    public:
        Mesh() {
            setDefaultsTransforms();
            setDefaultsOrientations();
        };
        Mesh(const uint num_triangle) : Array<Triangle>(num_triangle) {
            setDefaultsTransforms();
            setDefaultsOrientations();
        };
        Mesh(const Triangle& triangle) : Array<Triangle>(triangle) {
            setDefaultsTransforms();
            setDefaultsOrientations();
        };

        /*
        __host__ void setTransformMatrix(const TRSMatrix& TRS_matrix) {
            if (transform_matrix.size() == 1) {
                throw std::runtime_error("Transformation matrix has already been setted. Use SceneObject::offset, SceneObject::scale or SceneObject::rotate instead.");
            }
            transform_matrix.push_back(TRS_matrix);
        }

        __host__ __device__ Matrix4x4 getLocalToWorldMatrix() const {
            if (transform_matrix.size() == 1) {
                return transform_matrix[0].getLocalToWorldMatrix();
            } else {
                return Matrix4x4();
            }
        }

        __host__ __device__ Matrix4x4 getWorldToLocalMatrix() const {
            if (transform_matrix.size() == 1) {
                return transform_matrix[0].getWorldToLocalMatrix();
            } else {
                return Matrix4x4();
            }
        }*/

        /*
        __host__ void offset(const Vector<float>& offset) override {
            if (transform_matrix.size() == 1) {
                transform_matrix[0].addOffset(offset);
                addOffset(offset);
            } else {
                transform_matrix.push_back(TRSMatrix(offset, Vector<float>(1., 1., 1.), Vector<float>()));
            }
        }

        __host__ void scale(const Vector<float>& scale) override {
            if (transform_matrix.size() == 1) {
                transform_matrix[0].addScale(scale);
            } else {
                transform_matrix.push_back(TRSMatrix(Vector<float>(), scale, Vector<float>()));
            }
        }

        __host__ void rotate(const Vector<float>& angleDeg) override {
            if (transform_matrix.size() == 1) {
                transform_matrix[0].addRotation(angleDeg);
            } else {
                transform_matrix.push_back(TRSMatrix(Vector<float>(), Vector<float>(1., 1., 1.), angleDeg));
            }
        }*/

        __host__ void cuda() override {
            Array<Triangle>::cuda();
            SceneObject::cuda();
        }

        __host__ void cpu() override {
            Array<Triangle>::cpu();
            SceneObject::cpu();
        }

        __host__ void free() override {
            Array<Triangle>::free();
            SceneObject::free();
        }

};

using Meshes = Array<Mesh>;
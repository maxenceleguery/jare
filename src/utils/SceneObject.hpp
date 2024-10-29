#pragma once

#include <vector>
#include "../Vector.hpp"
#include "../TRSMatrix.hpp"
#include "./CudaReady.hpp"
#include "./Array.hpp"

enum Axis {
    X,
    Y,
    Z,
};


static Vector<float> rotateVector(const Vector<float>& vec, const float angle, const Axis axis) {
    float ux = 0.f;
    float uy = 0.f;
    float uz = 0.f;
    switch (axis) {
        case X:
            ux = 1.f;
            break;
        case Y:
            uy = 1.f;
            break;
        case Z:
            uz = 1.f;
            break;
    }
    
    Matrix<float> P = Matrix<float>(ux*ux,ux*uy,ux*uz,
                                    ux*uy,uy*uy,uy*uz,
                                    ux*uz,uy*uz,uz*uz);
    Matrix<float> I = Matrix<float>(1.f, MATRIX_EYE);
    Matrix<float> Q = Matrix<float>(0,-uz,uy,
                                    uz,0,-ux,
                                    -uy,ux,0);

    Matrix<float> R = P + (I-P)*std::cos(angle) + Q*std::sin(angle);
    return R*vec;
}

static std::vector<Vector<float>> rotateVectors(std::vector<Vector<float>> vecs, const float angle, const Axis axis) {
    float ux = 0.f;
    float uy = 0.f;
    float uz = 0.f;
    switch (axis) {
        case X:
            ux = 1.f;
            break;
        case Y:
            uy = 1.f;
            break;
        case Z:
            uz = 1.f;
            break;
    }
    
    Matrix<float> P = Matrix<float>(ux*ux,ux*uy,ux*uz,
                                    ux*uy,uy*uy,uy*uz,
                                    ux*uz,uy*uz,uz*uz);
    Matrix<float> I = Matrix<float>(1.f, MATRIX_EYE);
    Matrix<float> Q = Matrix<float>(0,-uz,uy,
                                    uz,0,-ux,
                                    -uy,ux,0);

    Matrix<float> R = P + (I-P)*std::cos(angle) + Q*std::sin(angle);
    for (Vector<float>& vec : vecs) {
        vec = R*vec;
    }
    return vecs;
}

class SceneObject : public CudaReady {
    private:
        Array<TRSMatrix> transform_matrix;

        void updateTRS() {
            if (transform_matrix.size() == 1) {
                transform_matrix[0] = TRSMatrix(transforms[0], transforms[1], transforms[2]);
            } else {
                transform_matrix.push_back(TRSMatrix(transforms[0], transforms[1], transforms[2]));
            }
        }
    public:
        Array<Vector<float>> orientations;
        Array<Vector<float>> transforms;

        void setDefaultsTransforms() {
            if (transforms.size() == 3) {
                transforms[0] = Vector<float>(0, 0, 0); // Total offset
                transforms[1] = Vector<float>(1, 1, 1); // Total scale
                transforms[2] = Vector<float>(0, 0, 0); // Total rotation
            } else {
                transforms.push_back(Vector<float>(0, 0, 0)); // Total offset
                transforms.push_back(Vector<float>(1, 1, 1)); // Total scale
                transforms.push_back(Vector<float>(0, 0, 0)); // Total rotation
            }
        }

        void setDefaultsTransforms(const Vector<float>& offset, const Vector<float>& scale, const Vector<float>& rotation) {
            if (transforms.size() == 3) {
                transforms[0] = offset; // Total offset
                transforms[1] = scale; // Total scale
                transforms[2] = rotation; // Total rotation
            } else {
                transforms.push_back(offset); // Total offset
                transforms.push_back(scale); // Total scale
                transforms.push_back(rotation); // Total rotation
            }
        }

        void setDefaultsOrientations() {
            if (orientations.size() == 3) {
                orientations[0] = Vector<float>(1, 0, 0); // Front
                orientations[1] = Vector<float>(0, 1, 0); // Left
                orientations[2] = Vector<float>(0, 0, 1); // Up
            } else {
                orientations.push_back(Vector<float>(1, 0, 0)); // Front
                orientations.push_back(Vector<float>(0, 1, 0)); // Left
                orientations.push_back(Vector<float>(0, 0, 1)); // Up
            }
        }

        void setDefaultsOrientations(const Vector<float>& front, const Vector<float>& left, const Vector<float>& up) {
            if (orientations.size() == 3) {
                orientations[0] = front; // Front
                orientations[1] = left; // Left
                orientations[2] = up; // Up
            } else {
                orientations.push_back(front); // Front
                orientations.push_back(left); // Left
                orientations.push_back(up); // Up
            }
        }


        __host__ void setTransformMatrix(const Vector<float>& offset, const Vector<float>& scale, const Vector<float>& rotation) {
            if (transform_matrix.size() == 1) {
                transform_matrix[0] = TRSMatrix(offset, rotation, scale);
            } else {
                transform_matrix.push_back(TRSMatrix(offset, rotation, scale));
            }
            
            if (transforms.size() == 3) {
                transforms[0] = offset;
                transforms[1] = scale;
                transforms[2] = rotation;
            } else {
                transforms.push_back(offset);
                transforms.push_back(scale);
                transforms.push_back(rotation);
            }
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
        }

        // Offset
        __host__ void setAbsoluteOffset(const Vector<float>& offset) {
            transforms[0] = offset;
            updateTRS();
        }

        __host__ void setRelativeOffset(const Vector<float>& offset) {
            transforms[0] = orientations[0]*offset.getX() + orientations[1]*offset.getY() + orientations[2]*offset.getZ();
            updateTRS();
        }
        
        __host__ void addAbsoluteOffset(const Vector<float>& offset) {
            transforms[0] += offset;
            updateTRS();
        }

        __host__ void addRelativeOffset(const Vector<float>& offset) {
            transforms[0] += orientations[0]*offset.getX() + orientations[1]*offset.getY() + orientations[2]*offset.getZ();
            updateTRS();
        }

        // Scale
        __host__ void setAbsoluteScale(const Vector<float>& scale) {
            transforms[1] = scale;
            updateTRS();
        }

        __host__ void setRelativeScale(const Vector<float>& scale) {
            transforms[1] = orientations[0]*scale.getX() + orientations[1]*scale.getY() + orientations[2]*scale.getZ();
            updateTRS();
        }
        
        __host__ void addAbsoluteScale(const Vector<float>& scale) {
            transforms[1] += scale;
            updateTRS();
        }

        __host__ void addRelativeScale(const Vector<float>& scale) {
            transforms[1] += orientations[0]*scale.getX() + orientations[1]*scale.getY() + orientations[2]*scale.getZ();
            updateTRS();
        }

        // Rotation
        __host__ void addAbsoluteRotation(const Vector<float>& angleDeg) {
            transform_matrix[0].addRotation(angleDeg);
            const Matrix4x4 mat = Transformations::GetRotationMatrix(angleDeg);
            orientations[0] = (mat*orientations[0]).normalize();
            orientations[1] = (mat*orientations[1]).normalize();
            orientations[2] = (mat*orientations[2]).normalize();

            transforms[2] += angleDeg;
            updateTRS();
        }

        __host__ void addRelativeRotation(const Vector<float>& angleDeg) {
            const Matrix4x4 mat_basis_change = Matrix4x4(
                Vector4<float>(orientations[0], 0),
                Vector4<float>(orientations[1], 0),
                Vector4<float>(orientations[2], 0),
                Vector4<float>()
            ).transpose();
            const Matrix4x4 mat = Transformations::GetRotationMatrix(mat_basis_change*angleDeg);
            orientations[0] = (mat*orientations[0]).normalize();
            orientations[1] = (mat*orientations[1]).normalize();
            orientations[2] = (mat*orientations[2]).normalize();

            transforms[2] += mat_basis_change*angleDeg;
            updateTRS();
        }

        __host__ void cuda() override {
            transform_matrix.cuda();
            orientations.cuda();
            transforms.cuda();
        }

        __host__ void cpu() override {
            transform_matrix.cpu();
            orientations.cpu();
            transforms.cpu();
        }

        __host__ void free() override {
            transform_matrix.free();
            orientations.free();
            transforms.free();
        }
};

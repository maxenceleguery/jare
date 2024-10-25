#pragma once

#include <vector>
#include "../Vector.hpp"
#include "../Matrix.hpp"

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

class SceneObject {
    public:
        virtual void offset(const Vector<float>& offset) = 0;
        virtual void scale(const Vector<float>& scale) = 0;
        virtual void rotate(const Vector<float>& angleDeg) = 0;
};

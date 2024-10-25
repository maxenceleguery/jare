#pragma once

#include "Vector.hpp"
#include "Quaternion.hpp"
#include "Matrix4x4.hpp"

namespace Transformations {

    static float ConvertDegToRad(const float degrees) {
        return ((float)3.141592653589f / (float) 180) * degrees;
    }

    static Matrix4x4 GetTranslationMatrix(const Vector<float>& position) {
        return Matrix4x4(
            Vector4<float>(1, 0, 0, position.getX()),
            Vector4<float>(0, 1, 0, position.getY()),
            Vector4<float>(0, 0, 1, position.getZ()),
            Vector4<float>(0, 0, 0, 1));

        /*
        return Matrix4x4(
            Vector4<float>(1, 0, 0, 0),
            Vector4<float>(0, 1, 0, 0),
            Vector4<float>(0, 0, 1, 0),
            Vector4<float>(position.getX(), position.getY(), position.getZ(), 1));
        */
    }

    static Matrix4x4 GetRotationMatrix(const Vector<float>& anglesDeg) {
        Vector<float> anglesRad = Vector<float>(ConvertDegToRad(anglesDeg.getX()), ConvertDegToRad(anglesDeg.getY()), ConvertDegToRad(anglesDeg.getZ()));

        Matrix4x4 rotationX = Matrix4x4(
            Vector4<float>(1, 0, 0, 0), 
            Vector4<float>(0, std::cos(anglesRad[0]), std::sin(anglesRad[0]), 0), 
            Vector4<float>(0, -std::sin(anglesRad[0]), std::cos(anglesRad[0]), 0),
            Vector4<float>(0, 0, 0, 1)
        );

        Matrix4x4 rotationY = Matrix4x4(
            Vector4<float>(std::cos(anglesRad[1]), 0, -std::sin(anglesRad[1]), 0),
            Vector4<float>(0, 1, 0, 0),
            Vector4<float>(std::sin(anglesRad[1]), 0, std::cos(anglesRad[1]), 0),
            Vector4<float>(0, 0, 0, 1)
        );

        Matrix4x4 rotationZ = Matrix4x4(
            Vector4<float>(std::cos(anglesRad[2]), std::sin(anglesRad[2]), 0, 0),
            Vector4<float>(-std::sin(anglesRad[2]), std::cos(anglesRad[2]), 0, 0),
            Vector4<float>(0, 0, 1, 0),
            Vector4<float>(0, 0, 0, 1)
        );

        return rotationX * rotationY * rotationZ;
    }

    static Matrix4x4 GetRotationMatrix(const Quaternion& q) {
        return Matrix4x4(
            Vector4<float>(1 - 2*q.c*q.c - 2*q.d*q.d, 2*q.b*q.c - 2*q.a*q.d, 2*q.b*q.d + 2*q.a*q.c, 0),
            Vector4<float>(2*q.b*q.c + 2*q.a*q.d, 1 - 2*q.b*q.b - 2*q.d*q.d, 2*q.c*q.d - 2*q.a*q.b, 0),
            Vector4<float>(2*q.b*q.d - 2*q.a*q.c, 2*q.c*q.d + 2*q.a*q.b, 1 - 2*q.b*q.b - 2*q.c*q.c, 0),
            Vector4<float>(0, 0, 0, 1)
        );
    }

    static Matrix4x4 GetScaleMatrix(const Vector<float>& scale) {
        return Matrix4x4(
            Vector4<float>(scale.getX(), 0, 0, 0),
            Vector4<float>(0, scale.getY(), 0, 0),
            Vector4<float>(0, 0, scale.getZ(), 0),
            Vector4<float>(0, 0, 0, 1)
        );
    }

    static Matrix4x4 Get_TRS_Matrix(const Vector<float>& position, const Vector<float>& rotationAngles, const Vector<float>& scale)  {
        return GetTranslationMatrix(position) * GetRotationMatrix(rotationAngles) * GetScaleMatrix(scale);
    }
    
} // namespace Transformations

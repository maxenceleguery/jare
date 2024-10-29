#pragma once
#include <iostream>
#include "Vector.hpp"
#include "Matrix.hpp"

#include <cuda_runtime.h>

class Line {
    protected:
        Vector<float> point;
        Vector<float> direction;
        Vector<float> invDir;

        __host__ __device__ bool isInInterval(const float value, const float x, const float y, const float eps=0) {
            return  x+eps < value && value < y-eps;
        }
    public:
        __host__ __device__ Line(){};
        __host__ __device__ Line(Vector<float> point0, Vector<float> direction0) : point(point0), direction(direction0.normalize()), invDir(direction0.normalize().invCoords()) {};
        __host__ __device__ ~Line(){};

        __host__ __device__ Vector<float> getPoint() const {
            return point;
        } 

        __host__ __device__ Vector<float> getDirection() const {
            return direction;
        } 

        __host__ __device__ void setPoint(const Vector<float>& p) {
            point=p;
        } 

        __host__ __device__ void setDirection(const Vector<float>& d) {
            direction=d;
            direction.normalize();
            invDir=direction.invCoords();
        } 

        __host__ __device__ bool IsIntersected(Line& line) {
            Vector<float> M = Vector<float>(point);
            Vector<float> N = Vector<float>(line.point);
            Vector<float> u = Vector<float>(direction);
            Vector<float> v = Vector<float>(line.direction);
            Matrix<float> mat = Matrix<float>(v.normSquared(),-v*u,0.,-v*u,u.normSquared(),0.,0.,0.,1.);
            if (u.crossProduct(v) == Vector<float>() || !mat.isInversible())
                return false;

            //Unecessary computation have been removed (Matrix inversions and multiplications are now "developed" by hand)
            // Speedup : 1.37

            //mat=mat.inverse();
            float det = mat.det();
            float fraqU = u.normSquared()/det;
            float fraqV = v.normSquared()/det;
            float vu = v*u/det;
            //mat=Matrix<float>(u.normSquared()/det,v*u/det,0,v*u/det,v.normSquared()/det,0,0,0,1);
            //mat=mat*(Matrix<float>(v,-u,Vector<float>()).transpose());
            /*mat = Matrix<float>(v.getX()*fraqU-vu*u.getX(),
                                v.getY()*fraqU-vu*u.getY(),
                                v.getZ()*fraqU-vu*u.getZ(),
                                v.getX()*vu-fraqV*u.getX(),
                                v.getY()*vu-fraqV*u.getY(),
                                v.getZ()*vu-fraqV*u.getZ(),
                                0,0,0); */
            //Vector<float> K = mat*(M-N);
            //float k1=K.getY(); float k2=K.getX();
            Vector<float> MN = M-N;
            float k1 = (v.getX()*vu-fraqV*u.getX())*MN.getX() + (v.getY()*vu-fraqV*u.getY())*MN.getY() + (v.getZ()*vu-fraqV*u.getZ())*MN.getZ();
            float k2 = (v.getX()*fraqU-vu*u.getX())*MN.getX() + (v.getY()*fraqU-vu*u.getY())*MN.getY() + (v.getZ()*fraqU-vu*u.getZ())*MN.getZ();
            return (M+u*k1)==(N+v*k2) && isInInterval(k1,0,1,1E-5) && isInInterval(k2,0,1,1E-5);
        }
};
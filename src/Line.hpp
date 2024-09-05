#pragma once
#include <iostream>
#include "Vector.hpp"
#include "Matrix.hpp"

#include <cuda_runtime.h>

class Line {
    protected:
        Vector<double> point;
        Vector<double> direction;
        Vector<double> invDir;

        __host__ __device__ bool isInInterval(const double value, const double x, const double y, const double eps=0) {
            return  x+eps < value && value < y-eps;
        }
    public:
        __host__ __device__ Line(){};
        __host__ __device__ Line(Vector<double> point0, Vector<double> direction0) : point(point0), direction(direction0.normalize()), invDir(direction.invCoords()) {};
        __host__ __device__ ~Line(){};

        __host__ __device__ Vector<double> getPoint() const {
            return point;
        } 

        __host__ __device__ Vector<double> getDirection() const {
            return direction;
        } 

        __host__ __device__ void setPoint(const Vector<double>& p) {
            point=p;
        } 

        __host__ __device__ void setDirection(const Vector<double>& d) {
            direction=d;
            direction.normalize();
            invDir=d.invCoords();
        } 

        __host__ __device__ bool IsIntersected(Line& line) {
            Vector<double> M = Vector<double>(point);
            Vector<double> N = Vector<double>(line.point);
            Vector<double> u = Vector<double>(direction);
            Vector<double> v = Vector<double>(line.direction);
            Matrix<double> mat = Matrix<double>(v.normSquared(),-v*u,0.,-v*u,u.normSquared(),0.,0.,0.,1.);
            if (u.crossProduct(v) == Vector<double>() || !mat.isInversible())
                return false;

            //Unecessary computation have been removed (Matrix inversions and multiplications are now "developed" by hand)
            // Speedup : 1.37

            //mat=mat.inverse();
            double det = mat.det();
            double fraqU = u.normSquared()/det;
            double fraqV = v.normSquared()/det;
            double vu = v*u/det;
            //mat=Matrix<double>(u.normSquared()/det,v*u/det,0,v*u/det,v.normSquared()/det,0,0,0,1);
            //mat=mat*(Matrix<double>(v,-u,Vector<double>()).transpose());
            /*mat = Matrix<double>(v.getX()*fraqU-vu*u.getX(),
                                v.getY()*fraqU-vu*u.getY(),
                                v.getZ()*fraqU-vu*u.getZ(),
                                v.getX()*vu-fraqV*u.getX(),
                                v.getY()*vu-fraqV*u.getY(),
                                v.getZ()*vu-fraqV*u.getZ(),
                                0,0,0); */
            //Vector<double> K = mat*(M-N);
            //double k1=K.getY(); double k2=K.getX();
            Vector<double> MN = M-N;
            double k1 = (v.getX()*vu-fraqV*u.getX())*MN.getX() + (v.getY()*vu-fraqV*u.getY())*MN.getY() + (v.getZ()*vu-fraqV*u.getZ())*MN.getZ();
            double k2 = (v.getX()*fraqU-vu*u.getX())*MN.getX() + (v.getY()*fraqU-vu*u.getY())*MN.getY() + (v.getZ()*fraqU-vu*u.getZ())*MN.getZ();
            return (M+u*k1)==(N+v*k2) && isInInterval(k1,0,1,1E-5) && isInInterval(k2,0,1,1E-5);
        }
};
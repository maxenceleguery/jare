#pragma once
#include <iostream>
#include "Vector.hpp"
#include "Matrix.hpp"

class Line {
    private:
        Vector<double> point;
        Vector<double> direction;
    public:
        Line(){};
        Line(Vector<double> point0, Vector<double> direction0) : point(point0), direction(direction0) {};
        ~Line(){};

        Vector<double> getPoint() const {
            return point;
        } 

        Vector<double> getDirection() const {
            return direction;
        } 

        bool IsIntersected(Line& line) {
            Vector<double> M = Vector<double>(point);
            Vector<double> N = Vector<double>(line.point);
            Vector<double> u = Vector<double>(direction);
            Vector<double> v = Vector<double>(line.direction);
            Matrix<double> mat = Matrix<double>(v.normSquared(),-v*u,0.,-v*u,u.normSquared(),0.,0.,0.,1.);
            if (!mat.isInversible() || u.crossProduct(v) == Vector<double>())
                return false; 
            mat=mat.inverse();
            mat=mat*(Matrix<double>(v,-u,Vector<double>()).transpose());
            Vector<double> K = mat*(M-N);
            //std::cout << "K : ";
            //K.printCoord();
            double k1=K.getY();
            double k2=K.getX();
            /*std::cout << "Intersection : " << std::endl;
            (M+u*k1).printCoord();
            (N+v*k2).printCoord();*/
            return (M+u*k1)==(N+v*k2) && 1E-7<k1 && k1<1-1E-7 && 1E-7<k2 && k2<1-1E-7;
        }
};



#pragma once
#include <iostream>
#include "Vector.hpp"
#include <vector>
#include <cmath>

class Face {
    private:
        std::vector<Vector<double>> vertices;
    public:
        Face();
        //Face(std::vector<Vector<double>>& vertices0) : vertices(vertices0) {};
        Face(Vector<double>& vec0) : vertices(1) {
            vertices[0]=vec0;
        };
        ~Face();

    void addVectex(const Vector<double>& vec) {
        vertices.push_back(vec);
    }

    Vector<double> getNormalVector() const {
        return (vertices[1]-vertices[0]).crossProduct(vertices[2]-vertices[0]).normalize();
    }

    bool isPlaneValid() const {
        if (vertices.size() < 3) {
            std::cout << "Not enough vertices to define a plan" << std::endl;
            return false;
        } else {
            Vector<double> normalVector = getNormalVector();
            if (normalVector == Vector(0,0,0))
                return false;
            for (uint i=0; i<vertices.size();i++) {
                if (std::abs(normalVector*(vertices[i]-vertices[0])) > 1E-5)
                    return false;
            }
            return true;
        }
    }

    void getPlaneEquation() {

    }

    void print() const {
        for (uint i=0;i<vertices.size();i++) {
            std::cout << "Vector " << i+1 << " : ";
            vertices[i].printCoord();
        }
    }
};


Face::~Face()
{
}

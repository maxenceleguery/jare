#pragma once
#include <iostream>
#include "Vector.hpp"
#include "Pixel.hpp"
#include "Line.hpp"
#include "Material.hpp"

#include <vector>
#include <cmath>
#include <numbers>

class Face {
    private:
        std::vector<Vector<double>> vertices;
        Material material;

    public:
        Face() : vertices(1) {};
        Face(std::vector<Vector<double>>& vertices0, Pixel color0) : vertices(vertices0) {
            material = Material(color0);
        };
        Face(Vector<double>& vec0, Pixel color0) : vertices(1), material(color0) {
            vertices[0]=vec0;
            material = Material(color0);
        };
        Face(Vector<double>& vec0, Material mat0) : vertices(1), material(mat0) {
            vertices[0]=vec0;
        };
        ~Face();

        Material getMaterial() const {
            return material;
        }

        void setMaterial(const Material mat) {
            material=mat;
        }

        void addVectex(const Vector<double>& vec) {
            vertices.push_back(vec);
        }

        Vector<double> getNormalVector() const {
            return (vertices[1]-vertices[0]).crossProduct(vertices[2]-vertices[0]).normalize();
        }

        Vector<double> getBarycenter() const {
            Vector<double> center = vertices[0];
                for (uint i=1;i<vertices.size();i++) {
                    center+=vertices[i];
                }
            center/=vertices.size();
            return center;
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

        bool isOnPlane(const Vector<double>& vec) const {
            Vector<double> normalVector = getNormalVector();
            return !(std::abs( (vec-vertices[0])*normalVector ) > 1E-5);
        }

        bool isInPolygoneOld(const Vector<double>& vec) {
            if (isOnPlane(vec)) {
                Vector<double> normalVector = getNormalVector();
                for (uint i=0;i<vertices.size()-1;i++) {
                    Vector<double> vec2 = (vec-vertices[i]).crossProduct(vertices[i+1]-vertices[i]);
                    if (std::abs(normalVector*vec2) > 1E-3)
                        return false;
                }
                return true;
            }
            return false;
        }

        bool isInPolygoneOld2(const Vector<double>& vec) {
            if (isOnPlane(vec)) {
                double sumAngles = 0.;
                for (uint i=0;i<vertices.size()-1;i++) {
                    //std::cout << sumAngles << std::endl;
                    sumAngles+=(vertices[i]-vec).getAngle(vertices[i+1]-vec);
                }
                sumAngles+=(vertices[vertices.size()-1]-vec).getAngle(vertices[0]-vec);
                //if ( !std::isnan(sumAngles) )
                    //std::cout << sumAngles << std::endl;
                if ( sumAngles > std::numbers::pi)
                    return false;
                else {
                    //std::cout << sumAngles << std::endl;
                    return true;
                }
            }
            return false;
        }

        // Only works if face is convex :(
        bool isInPolygone(const Vector<double>& vec) {
            Vector<double> center = getBarycenter();
            if (isOnPlane(vec) && isOnPlane(center) ) {
                Line line0 = Line(center,vec-center);
                uint counter = 0;
                for (uint i=0;i<vertices.size()-1;i++) {
                    Line line = Line(vertices[i],vertices[i+1]-vertices[i]);
                    if (line0.IsIntersected(line))
                        counter++;
                }
                Line line = Line(vertices[vertices.size()-1],vertices[0]-vertices[vertices.size()-1]);
                if (line0.IsIntersected(line))
                    counter++;
                //std::cout << counter << std::endl;
                return (counter%2 == 0);
            }
            return false;
        }

        // Plane equation ax + by + cz + d = 0
        std::vector<double> getPlaneEquation() {
            Vector<double> normalVector = getNormalVector();
            std::vector<double> planeEq;
            planeEq.reserve(4);
            planeEq[0] = normalVector.getX(); // a
            planeEq[1] = normalVector.getY(); // b
            planeEq[2] = normalVector.getZ(); // c
            planeEq[3] = -(normalVector*(vertices[0])); // d
            return planeEq;
        }

        Vector<double> getIntersection(const Line& line) {
            Vector<double> startingPoint = line.getPoint();
            Vector<double> direction = line.getDirection();
            Vector<double> normalVector = getNormalVector();
            double d = -normalVector*(vertices[0]);
            if (direction*normalVector != 0) {
                double k = (-d - normalVector*startingPoint)/(direction*normalVector);
                Vector<double> intersectionPoint = startingPoint + direction*k;
                if (isInPolygone(intersectionPoint) && k>1E-7) {
                    for (uint i=0;i<vertices.size();i++) {
                        if (intersectionPoint==vertices[i])
                            return Vector<double>();
                    }
                    return intersectionPoint;
                }
            }
            return Vector<double>();
        }

        void print() const {
            for (uint i=0;i<vertices.size();i++) {
                std::cout << "Vector " << i+1 << " : ";
                vertices[i].printCoord();
            }
        }

        void move(const Vector<double>& vec) {
            for (uint i=0;i<vertices.size();i++) {
                vertices[i] += vec;
            }
        }
};


Face::~Face()
{
}

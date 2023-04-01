#pragma once
#include "Vector.hpp"
#include "Camera.hpp"
#include "Face.hpp"
#include "Line.hpp"
#include "omp.h"

class Environment {
    private:
        Camera* cam;
        std::vector<Face> faces;
    public:
        Environment(): faces(1) {};
        Environment(Camera* cam0) : cam(cam0), faces(1) {};
        ~Environment();

        void addFace(Face& face) {
            faces.push_back(face);
        }

        void rayTrace() {
            #pragma omp parallel for num_threads(8)
            for(uint h = 0; h < cam->getHeight(); ++h) {
                #pragma omp parallel for num_threads(8)
                for(uint w = 0; w < cam->getWidth(); ++w) {
                    for (uint i = 0;i<faces.size();i++) {
                        Vector<double> direction = (cam->getOrientation()*cam->getFov()+cam->getPixelCoordOnCapt(w,h));
                        Line ray = Line(cam->getPosition(),direction);
                        Vector<double> intersectionPoint = faces[i].getIntersection(ray);
                        //intersectionPoint.printCoord();
                        if (intersectionPoint != Vector(0.,0.,0.)) {
                            //double distanceToCam = std::sqrt((intersectionPoint-cam->getPosition()).normSquared());
                            Pixel color = faces[i].getColor();
                            //std::cout << distanceToCam << std::endl;
                            //color.setR( (cam->getPosition()-intersectionPoint).normSquared()*(255-50)/(9.061-9.039) - 9.039*(255-50)/(9.061-9.039) + 50);
                            //color.setR(255 - color.getR());
                            cam->setPixel(h*cam->getWidth()+w, color);
                            //std::cout << w << " " << h << std::endl;
                        }
                    }
                }
            }
        }

        void addBackground(const Pixel& color) {
            //std::cout << cam.getWidth() << " " << cam.getHeight() << std::endl;
            for(uint h = 0; h < cam->getHeight(); ++h) 
                for(uint w = 0; w < cam->getWidth(); ++w)
                    cam->setPixel(h*cam->getWidth()+w, color);
        }
};


Environment::~Environment() {
}

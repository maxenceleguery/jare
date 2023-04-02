#pragma once
#include "Vector.hpp"
#include "Camera.hpp"
#include "Face.hpp"
#include "Line.hpp"

#include <stdlib.h>
#include <time.h>
#include "omp.h"

#define SIMPLE_RENDER 1000
#define RAYTRACING 1001

struct Hit {
    Material mat;
    Vector<double> point;
    Vector<double> normal;
    double distance = 1E50;
    bool hasHit = false;
};


class Environment {
    private:
        Camera* cam;
        std::vector<Face> faces;
        uint maxBounce = 3;
        uint samples = 1;
        Pixel backgroundColor = Pixel(0,0,0);
        uint mode = RAYTRACING;

        double randomValue(uint state) const {
            srand(time(NULL)*state);
            double result = (rand() % 10)/10.;
            return result;
        }

        double randomValueNormalDistribution(uint state) const { 
            double theta = 2 * std::numbers::pi * randomValue(state);
            double rho = std::sqrt(-2*std::log(randomValue(state*state)));
            return rho*std::cos(theta);
        }

        Vector<double> randomDirection(uint state) const {
            double x;  double y;  double z;
            do {
                x = randomValueNormalDistribution(state);
                y = randomValueNormalDistribution(state*42);
                z = randomValueNormalDistribution(state*77);
            } while ( std::abs(x)<1E-5 && std::abs(y)<1E-5 && std::abs(z)<1E-5);            
            return Vector<double>(x,y,z).normalize();
        }

        int sign(double number) const {
            if (number<0.)
                return -1;
            else if (number>0.)
                return 1;
            else
                return 0;
        }

    public:
        Environment(): faces(1) {};
        Environment(Camera* cam0) : cam(cam0), faces(1) {};
        ~Environment();

        void addFace(Face& face) {
            faces.push_back(face);
        }

        Vector<double> getDiffusionDirection(const Vector<double>& normal, uint state) const {
            Vector<double> dir = randomDirection(state);
            return dir*sign(dir*normal);
        }

        Hit simpleTrace(Line& ray) {
            Hit hit;
            for (uint i = 0;i<faces.size();i++) {
                Vector<double> intersectionPoint = faces[i].getIntersection(ray);
                double distance = std::sqrt((intersectionPoint-ray.getPoint()).normSquared());
                if (intersectionPoint != Vector(0.,0.,0.) && distance<hit.distance) {
                    hit.distance = distance;
                    hit.mat = faces[i].getMaterial();
                    hit.point=intersectionPoint;
                    hit.normal=faces[i].getNormalVector();
                    hit.hasHit=true;
                    //double distanceToCam = std::sqrt((intersectionPoint-cam->getPosition()).normSquared());
                    //color.setR( (cam->getPosition()-intersectionPoint).normSquared()*(255-50)/(9.061-9.039) - 9.039*(255-50)/(9.061-9.039) + 50);
                    //color.setR(255 - color.getR());
                }
            }
            return hit;
        }

        Pixel rayTrace1(Line& ray) {
            Hit hit = simpleTrace(ray);
            if (hit.hasHit)
                return hit.mat.getColor();
            else
                return backgroundColor;
        }

        Pixel rayTrace2(Line& ray, uint state) {
            Vector<double> incomingLight = Vector<double>();
            Vector<double> rayColor = Vector<double>(1.,1.,1.);
            for (uint bounce=0;bounce<maxBounce;bounce++) {
                Hit hit = simpleTrace(ray);
                if (hit.hasHit) {
                    ray.setPoint(hit.point);
                    ray.setDirection(getDiffusionDirection(hit.normal,state));
                    Vector<double> emittedLight = hit.mat.getColor().toVector() * hit.mat.getEmissionStrengh();
                    incomingLight += emittedLight.productTermByTerm(rayColor);
                    rayColor = rayColor.productTermByTerm(hit.mat.getColor().toVector())*(hit.normal*ray.getDirection());

                } else {
                    break;
                }
            }
            Pixel finalColor = Pixel(incomingLight);
            return finalColor;
        }

        void render() {
            #pragma omp parallel for num_threads(omp_get_num_devices())
            for(uint h = 0; h < cam->getHeight(); ++h) {
                #pragma omp parallel for num_threads(omp_get_num_devices())
                for(uint w = 0; w < cam->getWidth(); ++w) {
                    Vector<double> direction = (cam->getOrientation()*cam->getFov()+cam->getPixelCoordOnCapt(w,h)).normalize();
                    Line ray = Line(cam->getPosition(),direction);
                    Pixel color;
                    Vector<double> colorVec;
                    if (mode==SIMPLE_RENDER) 
                        color = rayTrace1(ray);
                    else if (mode==RAYTRACING) {
                        for (uint i=0;i<samples;i++)
                            colorVec += rayTrace2(ray,h*w).toVector();
                        colorVec/=samples;
                        color=Pixel(colorVec);
                    }
                    cam->setPixel(h*cam->getWidth()+w, color);
                }
            }
        }

        void addBackground(const Pixel& color) {
            backgroundColor = color;
            for(uint h = 0; h < cam->getHeight(); ++h) 
                for(uint w = 0; w < cam->getWidth(); ++w)
                    cam->setPixel(h*cam->getWidth()+w, color);
        }
};


Environment::~Environment() {
}

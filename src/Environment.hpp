#pragma once
#include "Vector.hpp"
#include "Camera.hpp"

class Environment {
    private:
        Camera* cam;
    public:
        Environment();
        Environment(Camera* cam0) : cam(cam0) {};
        ~Environment();

        void addBackground(Pixel& color) {
            //std::cout << cam.getWidth() << " " << cam.getHeight() << std::endl;
            for(uint l = 0; l < cam->getHeight(); ++l) 
                for(uint c = 0; c < cam->getWidth(); ++c)
                    cam->setPixel(l*cam->getWidth()+c, color);
        }
};


Environment::~Environment() {
}

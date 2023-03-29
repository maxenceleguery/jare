#include <iostream> 
#include "Vector.hpp"
#include "Camera.hpp"
#include "Pixel.hpp"
#include "Environment.hpp"
#include "Face.hpp"
#include <omp.h>

#include <string>

void renderFadeBlackToWhite(Environment& env, Camera& cam) {
	for (uint8_t i = 0; i<255; i++) {
		Pixel color = Pixel(i,i,i);
		env.addBackground(color);
		std::string name = "./render/image";
		std::string format = ".png";
		name.append(std::to_string(i));
		name.append(format);
		cam.renderImage(name.c_str());
	}
}

void testFace() {
	Vector<double> ex = Vector(0.,0.,0.);
	Vector<double> ey = Vector(0.,0.,1.);
	Vector<double> ez = Vector(1.,1.,1.);
	Vector<double> vec = Vector(1.,1.,0.);
	Face face = Face(ex);
	face.addVectex(ey);
	face.addVectex(ez);
	face.addVectex(vec);
	face.print();
	if (face.isPlaneValid())
		std::cout << "Plane valid" << std::endl;
	else
		std::cout << "Plane not valid" << std::endl;
	std::cout << "Normal vector : ";
	face.getNormalVector().printCoord();
}
 
int main() { 

	Vector<double> origine = Vector<double>(0,0,0);
	Camera cam = Camera(origine,800,600);
	Environment env = Environment(&cam);
	//std::cout << "Adding background" << std::endl;


	Pixel red = Pixel(255,0,0);
	Pixel green = Pixel(0,255,0);
	Pixel blue = Pixel(0,0,255);
	Pixel black = Pixel(0,0,0);
	Pixel white = Pixel(255,255,255);

	env.addBackground(red);
	cam.renderImage("./image.png");

	testFace();
	//renderFadeBlackToWhite(env,cam);

	return EXIT_SUCCESS; 
}
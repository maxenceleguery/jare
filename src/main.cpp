#include <iostream> 
#include "Vector.hpp"
#include "Camera.hpp"
#include "Pixel.hpp"
#include "Environment.hpp"
 
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
	env.addBackground(green);
	cam.renderImage();

	return EXIT_SUCCESS; 
}
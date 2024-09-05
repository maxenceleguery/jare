#include <iostream> 
#include "Vector.hpp"
#include "Camera.hpp"
#include "Pixel.hpp"
#include "Environment.hpp"
#include "Triangle.hpp"
#include "Matrix.hpp"
#include "Line.hpp"

#include <cuda_runtime.h>
#include "RayTrace.cuh"

#include <omp.h>
#include <string>
#include <chrono>

void testCam() {
	Vector<double> origine = Vector<double>(-2.,0.5,0.5);
	Camera cam = Camera(origine,Vector<double>(1.,0.,0.),800,600);
	cam.getPixelCoordOnCapt(0,0).printCoord();
	cam.getPixelCoordOnCapt(0,cam.getHeight()).printCoord();
	cam.getPixelCoordOnCapt(cam.getWidth(),0).printCoord();
	cam.getPixelCoordOnCapt(cam.getWidth(),cam.getHeight()).printCoord();
}

void renderFadeBlackToWhite() {
	Vector<double> origine = Vector<double>(0,0,0);
	Camera cam = Camera(origine,800,600);
	Environment env = Environment(&cam);

	for (uint8_t i = 0; i<255; i++) {
		Pixel color = Pixel(i,i,i);
		env.addBackground(color);
		std::string path = "./render/image";
		std::string format = ".png";
		path.append(std::to_string(i));
		path.append(format);
		cam.renderImage(path.c_str());
	}
}

void testTriangle() {
	Vector<double> ex = Vector(0.,0.,0.);
	Vector<double> ey = Vector(0.,0.,1.);
	Vector<double> ez = Vector(1.,1.,1.);
	Vector<double> vec = Vector(1.,1.,0.);
	Triangle triangle = Triangle(ex,Pixel(255,0,0));
	triangle.setvertex(1, ey);
	triangle.setvertex(2, ez);
	triangle.print();
	std::cout << "Normal vector : ";
	triangle.getNormalVector().printCoord();

	Vector<double> vec2 = Vector(0.6,0.5,0.5);
	if (triangle.isOnPlane(vec2))
		std::cout << "Point on plane" << std::endl;
	else
		std::cout << "Point not on plane" << std::endl;

	Vector<double> vec3 = Vector(0.5,0.5,10.);

}

void testMatrix() {
	Matrix<double> mat1 = Matrix(1.,2.,3.,1.,2.,3.,1.,2.,3.);
	Matrix<double> mat2 = Matrix(1.,MATRIX_EYE);
	Matrix<double> mat3 = Matrix(1.,2.,3.,0.,1.,4.,5.,6.,0.);
	Vector<double> vec1 = Vector(1.,2.,1.);
	(mat1+mat1).print();
	(mat1-mat1).print();
	(mat1*mat1).print();
	(mat1*10).print();
	(mat1.transpose()).print();
	(mat1*vec1).printCoord();
	std::cout << mat3.det() << std::endl;
	(mat3.inverse()).print();
}

void testLine() {
	Vector<double> vec1 = Vector(0.,1.,0.);
	Vector<double> vec2 = Vector(1.,-1.,0.);
	Vector<double> vec3 = Vector(0.,-2.,0.);
	Vector<double> vec4 = Vector(1.,1.,0.);
	Line l1 = Line(vec1,vec2);
	Line l2 = Line(vec3,vec4);
	std::cout << l1.IsIntersected(l2) << std::endl;
}

void testVector() {
	Vector<double> vec1 = Vector(1.,1.,0.);
	Vector<double> vec2 = Vector(1.,0.,0.);
	std::cout << vec1.getAngle(vec2) << std::endl;
}

void objRender() {
	Vector<double> origine = Vector<double>(-3.,0.,1.5);
	Vector<double> front = Vector<double>(1,0,-0.2);
	Camera cam = Camera(origine,front,1280,720);
	Environment env = Environment(&cam);

	cam.move(-Vector<double>(5.0,0.,-1.5));

	Material light = Materials::LIGHT;

	env.addSquare(Vector(20.,20.,0.),Vector(-20.,20.,0.),Vector(-20.,-20.,0.),Vector(20.,-20.,0.), Colors::WHITE);

	light.setColor(Colors::RED);
	env.addSquare(Vector(0.,-2.,0.)*2,Vector(0.,-2.,2.)*2,Vector(2.,-2.,2.)*2,Vector(2.,-2.,0.)*2, light); // left panel 
	light.setColor(Colors::GREEN);
	env.addSquare(Vector(0.,2.,0.)*2,Vector(2.,2.,0.)*2,Vector(2.,2.,2.)*2,Vector(0.,2.,2.)*2, light); // right panel

	//env.addSquare(Vector(0.,0.,0.),Vector(0.,0.,2.),Vector(2.,2.,2.),Vector(2.,2.,0.), Colors::YELLOW);
	//env.addSquare(Vector(0.,0.,0.),Vector(2.,-2.,0.),Vector(2.,-2.,2.),Vector(0.,0.,2.), Colors::CYAN);
	//env.addSquare(Vector(0.,0.,2.),Vector(2.,-2.,1.),Vector(2.,0.,2.),Vector(2.,2.,2.), Colors::MAGENTA);

	env.addObj("knight.obj",Vector<double>(0,0,0),0.5, Colors::WHITE);

	env.addBackground(Colors::BLACK);
	env.setMode(Mode::BVH_RAYTRACING);
	env.renderCudaBVH();
	std::string path = "./render4/image";
	std::string format = ".png";
	path.append(std::to_string(0));
	path.append(format);
	cam.renderImage(path.c_str());
}

int main() {
	//testCam();
	//testTriangle();
	//renderFadeBlackToWhite();
	//testMatrix();
	//testLine();
	//testVector();
	
	//firstRender();

	auto start = std::chrono::steady_clock::now();
	objRender();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "Render time:\t\t" << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS; 
}
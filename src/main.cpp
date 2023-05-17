#include <iostream> 
#include "Vector.hpp"
#include "Camera.hpp"
#include "Pixel.hpp"
#include "Environment.hpp"
#include "Face.hpp"
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

void testFace() {
	Vector<double> ex = Vector(0.,0.,0.);
	Vector<double> ey = Vector(0.,0.,1.);
	Vector<double> ez = Vector(1.,1.,1.);
	Vector<double> vec = Vector(1.,1.,0.);
	Face face = Face(ex,Pixel(255,0,0));
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

	Vector<double> vec2 = Vector(0.6,0.5,0.5);
	if (face.isOnPlane(vec2))
		std::cout << "Point on plane" << std::endl;
	else
		std::cout << "Point not on plane" << std::endl;

	Vector<double> vec3 = Vector(0.5,0.5,10.);
	if (face.isInPolygone(vec3))
		std::cout << "Point in polygone" << std::endl;
	else
		std::cout << "Point not in polygone" << std::endl;

	/*std::vector<double> planeEq = face.getPlaneEquation();
	for (uint i=0;i<planeEq.size();i++) {
		std::cout << planeEq[i] << std::endl;
	}*/

	face.getIntersection(Line(Vector<double>(0.5,0.,0.5),Vector<double>(0.,1.,0.))).printCoord();
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

void firstRender() {
	Vector<double> origine = Vector<double>(-3.,0.,1.5);
	Vector<double> front = Vector<double>(1,0,-0.2);
	Camera cam = Camera(origine,front,1280,720);
	Environment env = Environment(&cam);

	Pixel red = Pixel(255,0,0);
	Pixel green = Pixel(0,255,0);
	Pixel blue = Pixel(0,0,255);
	Pixel yellow = Pixel(255,255,0);
	Pixel cyan = Pixel(0,255,255);
	Pixel magenta = Pixel(255,0,255);
	Pixel black = Pixel(0,0,0);
	Pixel white = Pixel(255,255,255);

	Material light = Material(Pixel(255,255,255));
	light.setEmissionStrengh(1.);

	Material mirror = Material(Pixel(255,255,255));
	mirror.setSpecularSmoothness(1.);

	env.addSquare(Vector(20.,20.,0.),Vector(-20.,20.,0.),Vector(-20.,-20.,0.),Vector(20.,-20.,0.),mirror);

	env.addSquare(Vector(0.,0.,0.),Vector(0.,0.,1.),Vector(1.,1.,1.),Vector(1.,1.,0.),yellow);
	env.addSquare(Vector(0.,0.,0.),Vector(1.,-1.,0.),Vector(1.,-1.,1.),Vector(0.,0.,1.),cyan);
	env.addSquare(Vector(0.,0.,1.),Vector(1.,-1.,1.),Vector(2.,0.,1.),Vector(1.,1.,1.),magenta);

	env.addSquare(Vector(0.,-2.,0.),Vector(0.,-2.,2.),Vector(2.,-2.,2.),Vector(2.,-2.,0.),light); // left panel 
	env.addSquare(Vector(0.,2.,0.),Vector(0.,2.,2.),Vector(2.,2.,2.),Vector(2.,2.,0.),light); // right panel

	uint numberImage=10;
	auto start = std::chrono::steady_clock::now();

	for (uint i=0;i<numberImage;i++) {
		if (i%1==0)
			std::cout << "Rendering image N° " << i+1 << "/" << numberImage << std::endl;
		cam.setPosition(cam.getPosition()-Vector<double>(i/100.0,0.,0.));
		cam.rotate(0.01,ROT_UP);
		env.addBackground(Pixel(0,0,0));
		//env.render();
		env.renderCuda();
		std::string path = "./render2/image";
		std::string format = ".png";
		path.append(std::to_string(i));
		path.append(format);
		cam.renderImage(path.c_str());
	}

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "Elapsed time per image (render + png writing): " << elapsed_seconds.count()/numberImage << "s\n";
	std::cout << "Total time: " << elapsed_seconds.count() << "s\n";
}

int main() {
	//testCam();
	//testFace();
	//renderFadeBlackToWhite();
	//testMatrix();
	//testLine();
	//testVector();
	
	firstRender();

	return EXIT_SUCCESS; 
}
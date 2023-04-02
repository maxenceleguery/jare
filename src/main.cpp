#include <iostream> 
#include "Vector.hpp"
#include "Camera.hpp"
#include "Pixel.hpp"
#include "Environment.hpp"
#include "Face.hpp"
#include "Matrix.hpp"
#include "Line.hpp"
#include <omp.h>

#include <string>

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

	std::vector<double> planeEq = face.getPlaneEquation();
	for (uint i=0;i<planeEq.size();i++) {
		std::cout << planeEq[i] << std::endl;
	}

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
	Vector<double> origine = Vector<double>(-3,0.,1.5);
	Vector<double> orientation = Vector<double>(1,0,-0.2);
	Camera cam = Camera(origine,orientation,1280,720);
	Environment env = Environment(&cam);

	Vector<double> v1 = Vector(20.,20.,0.);
	Vector<double> v2 = Vector(20.,-20.,0.);
	Vector<double> v3 = Vector(-20.,-20.,0.);
	Vector<double> v4 = Vector(-20.,20.,0.);
	Face ground = Face(v1,Pixel(50,50,50));
	ground.addVectex(v2);
	ground.addVectex(v3);
	ground.addVectex(v4);
	env.addFace(ground);

	Vector<double> vec1 = Vector(0.,0.,0.);
	Vector<double> vec2 = Vector(0.,0.,1.);
	Vector<double> vec3 = Vector(1.,1.,1.);
	Vector<double> vec4 = Vector(1.,1.,0.);
	Face face1 = Face(vec1,Pixel(255,0,0));
	face1.addVectex(vec2);
	face1.addVectex(vec3);
	face1.addVectex(vec4);
	env.addFace(face1);

	Vector<double> vec5 = Vector(0.,0.,0.);
	Vector<double> vec6 = Vector(0.,0.,1.);
	Vector<double> vec7 = Vector(1.,-1.,1.);
	Vector<double> vec8 = Vector(1.,-1.,0.);
	Face face2 = Face(vec5,Pixel(0,255,0));
	face2.addVectex(vec6);
	face2.addVectex(vec7);
	face2.addVectex(vec8);
	env.addFace(face2);

	Vector<double> vec9 = Vector(0.,0.,1.);
	Vector<double> vec10 = Vector(1.,-1.,1.);
	Vector<double> vec11 = Vector(2.,0.,1.);
	Vector<double> vec12 = Vector(1.,1.,1.);
	Face face3 = Face(vec9,Pixel(0,0,255));
	face3.addVectex(vec10);
	face3.addVectex(vec11);
	face3.addVectex(vec12);
	env.addFace(face3);

	Material light = Material(Pixel(255,255,255));
	light.setEmissionStrengh(1.);
	Vector<double> vec13 = Vector(0.,0.,0.) + Vector(0.,-1.5,0.);
	Vector<double> vec14 = Vector(0.,0.,1.) + Vector(0.,-1.5,0.);
	Vector<double> vec15 = Vector(2.,0.,1.) + Vector(0.,-1.5,0.);
	Vector<double> vec16 = Vector(2.,0.,0.) + Vector(0.,-1.5,0.);
	Face face4 = Face(vec13,light);
	face4.addVectex(vec14);
	face4.addVectex(vec15);
	face4.addVectex(vec16);
	env.addFace(face4);

	uint numberImage=10;
	for (uint i=0;i<numberImage;i++) {
		if (i%10==0)
			std::cout << "Rendering image N° " << i << "/" << numberImage << std::endl;
		cam.setPosition(cam.getPosition()-Vector<double>(i/100.0,0.,0.));
		env.addBackground(Pixel(0,0,0));
		env.render();
		std::string name = "./render2/image";
		std::string format = ".png";
		name.append(std::to_string(i));
		name.append(format);
		cam.renderImage(name.c_str());
	}
}
 
int main() { 
	Pixel red = Pixel(255,0,0);
	Pixel green = Pixel(0,255,0);
	Pixel blue = Pixel(0,0,255);
	Pixel black = Pixel(0,0,0);
	Pixel white = Pixel(255,255,255);

	//testCam();
	//testFace();
	//renderFadeBlackToWhite();
	//testMatrix();
	//testLine();
	//testVector();

	firstRender();

	return EXIT_SUCCESS; 
}
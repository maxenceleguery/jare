#include <iostream> 
#include "Vector.hpp"
#include "Camera.hpp"
#include "Viewport.hpp"
#include "Pixel.hpp"
#include "Environment.hpp"
#include "Triangle.hpp"
#include "Matrix.hpp"
#include "Line.hpp"

#include <cuda_runtime.h>

#include <omp.h>
#include <string>
#include <chrono>
#include <thread>

void objRender() {
	Vector<float> origine = Vector<float>(-3.,0.,1.5);
	Vector<float> front = Vector<float>(1,0,-0.2);
	Camera cam = Camera(origine,front,1280,720);
	Viewport viewport = Viewport(&cam);
	Environment env = Environment(&cam);

	cam.move(-Vector<float>(5.0,0.,-1.5));

	Material light = Materials::LIGHT;

	env.addSquare(Vector(20.,20.,0.),Vector(-20.,20.,0.),Vector(-20.,-20.,0.),Vector(20.,-20.,0.), Colors::WHITE);

	light.setColor(Colors::RED);
	env.addSquare(Vector(0.,-2.,0.)*2,Vector(0.,-2.,2.)*2,Vector(2.,-2.,2.)*2,Vector(2.,-2.,0.)*2, light); // left panel 
	light.setColor(Colors::GREEN);
	env.addSquare(Vector(0.,2.,0.)*2,Vector(2.,2.,0.)*2,Vector(2.,2.,2.)*2,Vector(0.,2.,2.)*2, light); // right panel

	//env.addSquare(Vector(0.,0.,0.),Vector(0.,0.,2.),Vector(2.,2.,2.),Vector(2.,2.,0.), Material(Colors::WHITE, MaterialType::GLASS));
	//env.addSquare(Vector(0.,0.,0.),Vector(2.,-2.,0.),Vector(2.,-2.,2.),Vector(0.,0.,2.), Material(Colors::WHITE, MaterialType::GLASS));
	//env.addSquare(Vector(0.,0.,2.),Vector(2.,-2.,1.),Vector(2.,0.,2.),Vector(2.,2.,2.), Material(Colors::WHITE, MaterialType::GLASS));

	env.addObj("knight.obj",Vector<float>(0,0,0),0.5, Colors::WHITE);

	env.addBackground(Colors::BLACK);
	env.setMode(Mode::BVH_RAYTRACING);

	for (uint i = 0; i<10; i++) {
		env.renderCudaBVH();
		cam.move(Vector<float>(0., -0.5, 0.));
	}
	std::string path = "./render4/image";
	std::string format = ".png";
	path.append(std::to_string(0));
	path.append(format);
	cam.renderImage(path.c_str());
	viewport.stop();
}

void animObj() {
	Vector<float> origine = Vector<float>(-3.,0.,1.5);
	Vector<float> front = Vector<float>(1,0,-0.2);
	Camera cam = Camera(origine,front,1280,720);
	cam.init();
	Viewport viewport = Viewport(&cam);

	cam.move(-Vector<float>(5.0,0.,-1.5));
	cam.cuda();

	Environment env = Environment(&cam);

	Material light = Materials::LIGHT;

	env.addSquare(Vector(20.,20.,0.),Vector(-20.,20.,0.),Vector(-20.,-20.,0.),Vector(20.,-20.,0.), Colors::WHITE);

	light.setColor(Colors::GREEN);
	env.addSquare(Vector(0.,-2.,0.)*2,Vector(0.,-2.,2.)*2,Vector(2.,-2.,2.)*2,Vector(2.,-2.,0.)*2, light); // left panel 
	light.setColor(Colors::RED);
	env.addSquare(Vector(0.,2.,0.)*2,Vector(2.,2.,0.)*2,Vector(2.,2.,2.)*2,Vector(0.,2.,2.)*2, light); // right panel
	light.setColor(Colors::WHITE);

	//env.addSquare(Vector(2.,1.,0.)*2,Vector(2.,-1.,0.)*2,Vector(2.,-1.,2.)*2,Vector(2.,1.,2.)*2, light); // back panel

	//env.addSquare(Vector(0.,0.,0.),Vector(0.,0.,2.),Vector(2.,2.,2.),Vector(2.,2.,0.), Material(Colors::WHITE, MaterialType::GLASS));
	//env.addSquare(Vector(0.,0.,0.),Vector(2.,-2.,0.),Vector(2.,-2.,2.),Vector(0.,0.,2.), Material(Colors::WHITE, MaterialType::GLASS));
	//env.addSquare(Vector(0.,0.,2.),Vector(2.,-2.,1.),Vector(2.,0.,2.),Vector(2.,2.,2.), Material(Colors::WHITE, MaterialType::GLASS));

	env.addObj("knight.obj", Vector<float>(0,0,0), 0.5, Material(Colors::WHITE, MaterialType::DEFAULT));
	env.addObj("sphere.obj", Vector<float>(0,2,2), 0.5, Material(Colors::WHITE, MaterialType::MIRROR));

	//env.addBackground(Colors::BLACK);
	env.setMode(Mode::BVH_RAYTRACING);
	env.compute_bvhs();

	viewport.start();
	while (viewport.isOn()) {
		env.renderCudaBVH();
	}

	viewport.stop();
	cam.cpu();
	cam.free();
}

void test_random() {
	srand(time(NULL));
	
	RandomInterface random_test;
	uint state = 894965656;

	float threshold = 124.34f;
	uint n = 10000;
	uint k = 100;
	if (n/k < 5) {
		throw std::exception();
	}
	std::vector<float> observations = std::vector<float>(n);
	std::vector<int> c = std::vector<int>(k);
	for (uint j=0; j<k; j++) {
		c[j] = 0;
	}

	for (uint i = 0; i<n; i++) {
		observations[i] = random_test.randomValue(state);
		//observations[i] = (1.f*rand()) / (1.f*RAND_MAX);
		c[(int) (observations[i]*k)] += 1;
		//state = random_test.randomValue(state)*100000000000000;
	}

	float khi_square = 0;
	for (uint j=0; j<k; j++) {
		//std::cout << c[j] << std::endl;
		khi_square += std::pow(c[j] - (1.f*n)/(1.f*k), 2);
	}
	khi_square *= (1.f*k)/(1.f*n);
	if (khi_square > threshold) {
		throw Khi2Error(khi_square, threshold);
	}
	//std::cout << "khi² = " << khi_square << std::endl;
}

int main() {
	static_assert(std::is_base_of<CudaReady, Pixel>::value == false);
	static_assert(std::is_base_of<CudaReady, Array<double>>::value == true);
	static_assert(std::is_base_of<CudaReady, BVH>::value == true);

	for (uint i=0; i<10; i++)
		test_random();
	std::cout << "Tests on randomness passed" << std::endl;

	/*
	uint W = 1280;
	uint H = 720;
	uint w = 1000;
	uint h = 500;
	uint idx = h * W + w;
	std::cout << h << " == " << idx/W << std::endl;
	std::cout << w << " == " << idx%W << std::endl;
	*/
	
	Ray ray = Ray(Vector<float>(0, 0, -1), Vector<float>(0, 1, 1));
	Triangle tri = Triangle();
	tri.setvertex(0, Vector<float>(-1, 0, 0));
	tri.setvertex(1, Vector<float>(1, 0, 0));
	tri.setvertex(2, Vector<float>(0, 2, 0));
	Hit hit = ray.rayTriangle(tri);
	//hit.getNormal().printCoord();
	//hit.getPoint().printCoord();

	//return 0;
	auto start = std::chrono::steady_clock::now();
	animObj();
	auto end = std::chrono::steady_clock::now();

	std::chrono::duration<float> elapsed_seconds = end-start;
	std::cout << "Render time:\t\t" << elapsed_seconds.count() << "s\n";

	return EXIT_SUCCESS; 
}
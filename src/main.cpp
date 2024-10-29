#include <iostream> 
#include "Vector.hpp"
#include "Matrix4x4.hpp"
#include "Camera.hpp"
#include "Viewport.hpp"
#include "Pixel.hpp"
#include "Environment.hpp"
#include "Triangle.hpp"
#include "Matrix.hpp"
#include "Line.hpp"

#include "SceneObjectWindow.hpp"

#include <cuda_runtime.h>

#include <omp.h>
#include <string>
#include <chrono>
#include <thread>

void objRender() {
	Vector<float> origine = Vector<float>(-3.,0.,1.5);
	Vector<float> front = Vector<float>(1,0,-0.2);
	Camera cam = Camera(origine,front,1280,720);
	Environment env = Environment(&cam);
	Viewport viewport = Viewport(&env);

	cam.addRelativeOffset(-Vector<float>(5.0,0.,-1.5));

	Material light = Materials::LIGHT;

	env.addSquare(Vector(20.,20.,0.),Vector(-20.,20.,0.),Vector(-20.,-20.,0.),Vector(20.,-20.,0.), Colors::WHITE);

	light.setColor(Colors::RED);
	env.addSquare(Vector(0.,-2.,0.)*2,Vector(0.,-2.,2.)*2,Vector(2.,-2.,2.)*2,Vector(2.,-2.,0.)*2, light); // left panel 
	light.setColor(Colors::GREEN);
	env.addSquare(Vector(0.,2.,0.)*2,Vector(2.,2.,0.)*2,Vector(2.,2.,2.)*2,Vector(0.,2.,2.)*2, light); // right panel

	//env.addSquare(Vector(0.,0.,0.),Vector(0.,0.,2.),Vector(2.,2.,2.),Vector(2.,2.,0.), Material(Colors::WHITE, MaterialType::GLASS));
	//env.addSquare(Vector(0.,0.,0.),Vector(2.,-2.,0.),Vector(2.,-2.,2.),Vector(0.,0.,2.), Material(Colors::WHITE, MaterialType::GLASS));
	//env.addSquare(Vector(0.,0.,2.),Vector(2.,-2.,1.),Vector(2.,0.,2.),Vector(2.,2.,2.), Material(Colors::WHITE, MaterialType::GLASS));

	env.addObj("knight.obj", Vector<float>(0,0,0), 0.5, Vector<float>(-90,0,0), Colors::WHITE);

	env.addBackground(Colors::BLACK);
	env.setMode(Mode::BVH_RAYTRACING);

	for (uint i = 0; i<10; i++) {
		env.renderCudaBVH();
		cam.addRelativeOffset(Vector<float>(0., -0.5, 0.));
	}
	std::string path = "./render4/image";
	std::string format = ".png";
	path.append(std::to_string(0));
	path.append(format);
	cam.renderImage(path.c_str());
	viewport.stop();
}

void animObj() {
	Vector<float> origine = Vector<float>(-10, 0, 2.5);
	Vector<float> front = Vector<float>(1, 0, 0);
	Camera cam = Camera(origine, front, 1280, 720);
	cam.init();
	cam.cuda();

	Environment env = Environment(&cam);
	Viewport viewport = Viewport(&env);

	Material light = Materials::LIGHT;

	env.addSquare(Vector(20.,20.,0.),Vector(-20.,20.,0.),Vector(-20.,-20.,0.),Vector(20.,-20.,0.), Pixel(220, 220, 220));

	light.setColor(Colors::GREEN);
	env.addSquare(Vector(0.,-2.,0.)*2,Vector(0.,-2.,2.)*2,Vector(2.,-2.,2.)*2,Vector(2.,-2.,0.)*2, light); // left panel 
	light.setColor(Colors::RED);
	env.addSquare(Vector(0.,2.,0.)*2,Vector(2.,2.,0.)*2,Vector(2.,2.,2.)*2,Vector(0.,2.,2.)*2, light); // right panel
	light.setColor(Colors::WHITE);

	//env.addSquare(Vector(2.,1.,0.)*2,Vector(2.,-1.,0.)*2,Vector(2.,-1.,2.)*2,Vector(2.,1.,2.)*2, light); // back panel

	//env.addSquare(Vector(0.,0.,0.),Vector(0.,0.,2.),Vector(2.,2.,2.),Vector(2.,2.,0.), Material(Colors::WHITE, MaterialType::GLASS));
	//env.addSquare(Vector(0.,0.,0.),Vector(2.,-2.,0.),Vector(2.,-2.,2.),Vector(0.,0.,2.), Material(Colors::WHITE, MaterialType::GLASS));
	//env.addSquare(Vector(0.,0.,2.),Vector(2.,-2.,1.),Vector(2.,0.,2.),Vector(2.,2.,2.), Material(Colors::WHITE, MaterialType::GLASS));

	//env.addObj("knight.obj", Vector<float>(0, 0, 0), 1., Vector<float>(90, 0, 0), Material(Colors::WHITE, MaterialType::DEFAULT));
	env.addObj("sphere.obj", Vector<float>(0, 0, 0.5), 1., Vector<float>(90, 0, 0), Material(Colors::WHITE, MaterialType::MIRROR));

	//env.addBackground(Colors::BLACK);
	env.setMode(Mode::BVH_RAYTRACING);
	env.compute_bvhs();

	//std::cout << "Total allocated cuda memory : " << human_rep(allocated_cuda_memory) << std::endl;

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

	double threshold = 124.34;
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

	double khi_square = 0;
	for (uint j=0; j<k; j++) {
		//std::cout << c[j] << std::endl;
		khi_square += std::pow(c[j] - (1.f*n)/(1.f*k), 2);
	}
	khi_square *= (1.*k)/(1.*n);
	if (khi_square > threshold) {
		throw Khi2Error(khi_square, threshold);
	}
	//std::cout << "khi² = " << khi_square << std::endl;
}

void test_matrix4x4() {
	Matrix4x4 mat = Matrix4x4(
		Vector4<float>(1, 0, 0, 0),
		Vector4<float>(0, 1, 0, 0),
		Vector4<float>(0, 0, 1, 0),
		Vector4<float>(0, 0, 0, 1)
	);
	if (mat != mat.inverse()) {
		throw std::runtime_error("Inverse of identity wrong");
	}

	Matrix4x4 mat2 = Matrix4x4(
		Vector4<float>(2, 2, 3, 3),
		Vector4<float>(2, 3, 3, 2),
		Vector4<float>(5, 3, 7, 9),
		Vector4<float>(3, 2, 4, 7)
	);
	Matrix4x4 mat2_inv = Matrix4x4(
		Vector4<float>(26, -11, -7, 1),
		Vector4<float>(0, -1, 1, -1),
		Vector4<float>(-16, 7, 3, 1),
		Vector4<float>(-2, 1, 1, -1)
	)*-0.5;
	if (mat2.inverse() != mat2_inv) {
		throw std::runtime_error("Inverse wrong");
	}

	if (mat*mat2 != mat2) {
		throw std::runtime_error("Mult with ID wrong");
	}

	if (mat2*Vector4<float>(1, 1, 1, 1) != Vector4<float>(10, 10, 24, 16)) {
		throw std::runtime_error("Mult with vect4 wrong");
	}
}

void test_gtkmm() {
	//auto app = Gtk::Application::create();
	//SceneObjectWindow win = SceneObjectWindow();
	//app->run(win);
}

int main() {
	static_assert(std::is_base_of<CudaReady, Pixel>::value == false);
	static_assert(std::is_base_of<CudaReady, Array<double>>::value == true);
	static_assert(std::is_base_of<CudaReady, BVH>::value == true);

	test_matrix4x4();

	for (uint i=0; i<10; i++)
		test_random();
	std::cout << "Tests on randomness passed" << std::endl;

	//test_gtkmm();

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

	//std::cout << "Allocated cuda memory at the end : " << human_rep(allocated_cuda_memory) << std::endl;
	return EXIT_SUCCESS; 
}
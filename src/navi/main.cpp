#include <functional>
#include <iostream>
#include <random>

#define NDEBUG

#include "lumina.hpp"

#include "cuda/kernels.cuh"

using namespace lm;

int
main()
{
	// slam::sense	 sense(nullptr, "/dev/ttyTHS1");
	slam::camera camera("/dev/video0", 1280, 720, 2);

	// image<rgba> tof_frame;
	image<rgb> rgb_frame;

	// sense >> tof_frame;
	camera >> rgb_frame;

	// tof_frame.write("tof.png");
	rgb_frame.write("rgb.png");
}

#include <functional>
#include <iostream>
#include <random>

#define NDEBUG

#include "lumina.hpp"

#include "cuda/kernels.cuh"
#include "slam/camera.hpp"
#include "util/profiler.hpp"
#include "util/timer.hpp"

#include "slam/sense.hpp"

using namespace lm;

int
main(int argc, char** argv)
{
	profiler::begin("trace.json");

	slam::sense sense("/dev/video0", "/dev/ttyTHS1");

	image<byte> tof_frame;

	while (true) {
		sense >> tof_frame;
		tof_frame.write("out.png");
	}

	profiler::end();
}

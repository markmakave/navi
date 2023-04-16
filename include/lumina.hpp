/*
	 ___                                                
	/\_ \                        __                     
	\//\ \    __  __    ___ ___ /\_\    ___      __     
	  \ \ \  /\ \/\ \ /' __` __`\/\ \ /' _ `\  /'__`\   
	   \_\ \_\ \ \_\ \/\ \/\ \/\ \ \ \/\ \/\ \/\ \L\.\_ 
	   /\____\\ \____/\ \_\ \_\ \_\ \_\ \_\ \_\ \__/.\_\
	   \/____/ \/___/  \/_/\/_/\/_/\/_/\/_/\/_/\/__/\/_/
*/

#pragma once

#include "base/array.hpp"
#include "base/matrix.hpp"
#include "base/color.hpp"
#include "base/image.hpp"
#include "base/blas.hpp"
#include "base/memory.hpp"
#include "base/types.hpp"

#include "cuda/cuda.cuh"
#include "cuda/matrix.cuh"
#include "cuda/array.cuh"
#include "cuda/brief.cuh"

#include "slam/brief.hpp"
#include "slam/camera.hpp"
#include "slam/detect.hpp"

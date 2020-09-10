#ifndef EXTERNAL_HPP
#define EXTERNAL_HPP

#include <assert.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <sstream>
#include <thrust/copy.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <time.h>

#include <math.h>
#include <stdlib.h>

// stb image read & write
#define STB_IMAGE_IMPLEMENTATION
#include <nextweek/stb_image.h>

#endif

#pragma once

#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <string>
#include <chrono>
#include <functional>

//cpp
#include <vector>
#include <iostream>
#include <numeric>
#include <string>
#include <optional>
#include <thread>

#define h5_debug
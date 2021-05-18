#pragma once

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

//cpp
#include <vector>
#include <iostream>
#include <numeric>
#include <string>
#include <optional>
#include <thread>

template<typename T>
using Tensor = std::vector<std::vector<std::vector<T>>>;

template<typename T>
using Image = std::vector<std::vector<T>>;

#define h5_debug
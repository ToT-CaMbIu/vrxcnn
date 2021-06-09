#pragma once

#include "../connected_libs.h"
#include "../utils.h"
#include "../opencl_environment.h"

void opencl_create_program_test3d(CLVars& cl_vars,
                                  const char* kernel_name,
                                  float *A,
                                  float *C,
                                  int x, int y, int z);

std::vector<float> make_test3d(CLVars& cl_vars);

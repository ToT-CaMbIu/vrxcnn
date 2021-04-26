#pragma once

#include "../connected_libs.h"
#include "../utils.h"
#include "../opencl_environment.h"

void opencl_create_program_max_pool(CLVars& cl_vars,
                                    const char* kernel_name,
                                    float *A,
                                    float *C,
                                    int n, int m);

std::vector<float> make_max_pool(CLVars& cl_vars);
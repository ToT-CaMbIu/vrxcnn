#pragma once

#include "../connected_libs.h"
#include "../utils.h"
#include "../opencl_environment.h"


void opencl_create_program_conv(CLVars& cl_vars,
                                const char* kernel_name,
                                float *A,
                                float *Filter,
                                float *C,
                                int n, int m,
                                int n1, int m1,
                                int n2, int m2,
                                int nM, int mM,
                                int ts,
                                int blocks_counter_row);

std::vector<float> make_convolution(CLVars& cl_vars);
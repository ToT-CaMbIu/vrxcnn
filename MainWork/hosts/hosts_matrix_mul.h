#pragma once

#include "../connected_libs.h"
#include "../utils.h"
#include "../opencl_environment.h"

void opencl_create_program_matrix_mul(CLVars& cl_vars,
                                      const char* kernel_name,
                                      float *A,
                                      float *B,
                                      float *C,
                                      int n, int m,
                                      int k,
                                      int ts);

std::vector<float> make_matrix_mul(CLVars& cl_vars,
                                   const std::vector<std::vector<float>>& A,
                                   const std::vector<std::vector<float>>& B,
                                   bool standard_definition = true);
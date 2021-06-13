#pragma once

#include "../connected_libs.h"
#include "../utils.h"
#include "../models.h"
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

std::vector<float> make_convolution(CLVars& cl_vars,
                                    bool standard_definition = true);

void opencl_create_program_conv_3d(CLVars& cl_vars,
                                   const char* kernel_name,
                                   const float *A,
                                   const float *Filter,
                                   const float *Bias,
                                   float *C,
                                   int n, int m,
                                   int n1, int m1,
                                   int n2, int m2,
                                   int nM, int mM,
                                   int ts,
                                   int blocks_counter_row,
                                   int z, int count_of_weights);

Tensor<float> make_convolution_3d(CLVars& cl_vars,
                                  const Tensor<float>& tensor,
                                  const Tensor<float>& filters,
                                  const std::vector<float>& bias,
                                  std::function<float(float num)> activation,
                                  bool standard_definition = true);
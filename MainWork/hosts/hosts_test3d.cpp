#include "hosts_test3d.h"

void opencl_create_program_test3d(CLVars& cl_vars,
                                  const char* kernel_name,
                                  float *A,
                                  float *C,
                                  int x, int y, int z) {

    cl_mem A_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                 x * y * z * sizeof(float), nullptr, &cl_vars.clStatus);
    cl_mem C_cl = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY,
                                 x * y * z * sizeof(float), nullptr, &cl_vars.clStatus);

    CL_CHECK(clEnqueueWriteBuffer(cl_vars.command_queue, A_cl, CL_TRUE, 0,
                                  x * y * z * sizeof(float), A, 0, nullptr, nullptr));

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, "-I..", nullptr, nullptr);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &x);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &y);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &z);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(cl_mem), (void *) &A_cl);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(cl_mem), (void *) &C_cl);

    size_t global_size[3];
    size_t local_size[3];

    global_size[0] = z;
    global_size[1] = x;
    global_size[2] = y;
    local_size[0] = 1;
    local_size[1] = 2;
    local_size[2] = 2;

    auto time_start = std::chrono::high_resolution_clock::now();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 3, nullptr,
                                    global_size, local_size, 0, nullptr, nullptr));
    CL_CHECK(clFinish(cl_vars.command_queue));

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, C_cl, CL_TRUE, 0,
                                 x * y * z * sizeof(float), C, 0, nullptr, nullptr));

    std::cout << "kernels took " << elapsed << " ms to execute" << std::endl;

    clReleaseMemObject(A_cl);
    clReleaseMemObject(C_cl);
}

std::vector<float> make_test3d(CLVars& cl_vars) {

    opencl_environment_definition(cl_vars, "kernels/kernel_3d_test.cl");

    int x = 4, y = 4, z = 99;

    std::vector<float> A(x * y * z);
    std::vector<float> C(x * y * z);

    for(size_t k = 0; k < z; ++k) {
        for (size_t i = 0; i < x; ++i) {
            for (size_t j = 0; j < y; ++j) {
                A[k * x * y + i * y + j] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
            }
        }
    }

    opencl_create_program_test3d(cl_vars, "test3d",
                                   A.data(), C.data(), x, y, z);

    print_matrix(C, x, y);

    std::cout << ((A == C) ? "Passed!" : "Failed!") << std::endl;

    return C;
}

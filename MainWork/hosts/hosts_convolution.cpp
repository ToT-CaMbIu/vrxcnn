#include "hosts_convolution.h"

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
                                int blocks_counter_row)   {

    cl_mem A_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, n * m * sizeof(float),
                                 nullptr, &cl_vars.clStatus);
    cl_mem Filter_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, n1 * m1 * sizeof(float),
                                      nullptr, &cl_vars.clStatus);
    cl_mem C_cl = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY, n2 * m2 * sizeof(float),
                                 nullptr, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_cl, CL_TRUE, 0,
                         n * m * sizeof(float), A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter_cl, CL_TRUE, 0,
                         n1 * m1 * sizeof(float), Filter, 0, nullptr, nullptr);

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, "-I..", nullptr, nullptr);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    int n1_changed = 1;
    int m1_changed = n1 * m1;

    int k_aligned = m1_changed;

    if(k_aligned % ts != 0) {
        k_aligned += ts - (k_aligned % ts);
    }

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m1);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &n1_changed);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &m1_changed);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(int), (void *) &nM);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(int), (void *) &mM);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(int), (void *) &k_aligned);
    clSetKernelArg(cl_vars.kernel, 7, sizeof(int), (void *) &ts);
    clSetKernelArg(cl_vars.kernel, 8, sizeof(int), (void *) &blocks_counter_row);
    clSetKernelArg(cl_vars.kernel, 9, sizeof(cl_mem), (void *) &A_cl);
    clSetKernelArg(cl_vars.kernel, 10, sizeof(cl_mem), (void *) &Filter_cl);
    clSetKernelArg(cl_vars.kernel, 11, sizeof(cl_mem), (void *) &C_cl);
    clSetKernelArg(cl_vars.kernel, 12, ts * ts * sizeof(float), nullptr);
    clSetKernelArg(cl_vars.kernel, 13, ts * ts * sizeof(float), nullptr);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n1_changed;
    global_size[1] = mM;

    if(global_size[0] % ts != 0) {
        global_size[0] += ts - (global_size[0] % ts);
    }
    if(global_size[1] % ts != 0) {
        global_size[1] += ts - (global_size[1] % ts);
    }

    local_size[0] = ts;
    local_size[1] = ts;

    auto time_start = std::chrono::high_resolution_clock::now();

    cl_vars.clStatus |= clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, nullptr,
                                               global_size, local_size, 0, nullptr, nullptr);

    cl_vars.clStatus |= clEnqueueReadBuffer(cl_vars.command_queue, C_cl, CL_TRUE, 0,
                                            n2 * m2 * sizeof(float), C, 0, nullptr, nullptr);

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    std::cout << "kernels took " << elapsed << " ms to execute" << std::endl;

    clReleaseMemObject(A_cl);
    clReleaseMemObject(Filter_cl);
    clReleaseMemObject(C_cl);
}

std::vector<float> make_convolution(CLVars& cl_vars) {

    opencl_environment_definition(cl_vars, "kernels/kernel_conv.cl");

    //input parameters
    int n = rand() % 5000 + 1000, m = rand() % 5000 + 1000;
    int n1 = 5, m1 = 5;

    if(n < n1 || m < m1) {
        throw "Incorrect parameters of the kernel";
    }

    int n2 = n - n1 + 1;
    int m2 = m - m1 + 1;
    int ts = 15;

    std::vector<float> A(n * m);
    std::vector<float> Filter(n1 * m1);
    std::vector<float> C(n2 * m2);

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; ++j) {
            A[i * m + j] = 2.0 * (rand() % (j + 1));
        }
    }

    for (size_t i = 0; i < n1; i++) {
        for(size_t j = 0; j < m1; ++j) {
            Filter[i * m1 + j] = 2.0 * (rand() % (j + 1));
        }
    }

    opencl_create_program_conv(cl_vars, "matrix_convolutional_transformation", A.data(),
                               Filter.data(), C.data(), n, m, n1, m1, n2, m2, n1 * m1, n2 * m2, ts, m2);

    assert(test_convolution_valid(n, m, n1, m1, n2, m2, A, Filter, C));

    return C;
}

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
    CL_CHECK(clFinish(cl_vars.command_queue));
    
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    cl_vars.clStatus |= clEnqueueReadBuffer(cl_vars.command_queue, C_cl, CL_TRUE, 0,
                                            n2 * m2 * sizeof(float), C, 0, nullptr, nullptr);

    std::cout << "kernels took " << elapsed << " ms to execute" << std::endl;

    clReleaseMemObject(A_cl);
    clReleaseMemObject(Filter_cl);
    clReleaseMemObject(C_cl);
}

std::vector<float> make_convolution(CLVars& cl_vars) {

    opencl_environment_definition(cl_vars, "kernels/kernel_conv.cl");

    //input parameters
    int n = rand() % 5000 + 1000, m = rand() % 5000 + 1000;
    int n1 = 27, m1 = 27;
    int ts = 15;

    std::cout << "convolution" << std::endl;
    std::cout << "n: " << n << " m: " << m << " block_x: " <<
        n1 << " block_y " << m1 << " ts: " << ts << std::endl;

    if(n < n1 || m < m1) {
        throw "Incorrect parameters of the kernel";
    }

    int n2 = n - n1 + 1;
    int m2 = m - m1 + 1;

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

void opencl_create_program_conv_3d(CLVars& cl_vars,
                                   const char* kernel_name,
                                   float *A,
                                   float *Filter,
                                   float *C,
                                   int n, int m,
                                   int n1, int m1,
                                   int n2, int m2,
                                   int nM, int mM,
                                   int ts,
                                   int blocks_counter_row,
                                   int z, int count_of_weights) {

    cl_mem A_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, z * n * m * sizeof(float),
                                 nullptr, &cl_vars.clStatus);
    cl_mem Filter_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, count_of_weights *
                                 n1 * m1 * sizeof(float), nullptr, &cl_vars.clStatus);
    cl_mem C_cl = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY, count_of_weights *
                                 n2 * m2 * sizeof(float), nullptr, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_cl, CL_TRUE, 0,
                         z * n * m * sizeof(float), A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter_cl, CL_TRUE, 0,
                         count_of_weights * n1 * m1 * sizeof(float), Filter, 0, nullptr, nullptr);

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, "-I..", nullptr, nullptr);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    int n1_changed = 1;
    int m1_changed = n1 * m1;

    int k_aligned = m1_changed;

    if(k_aligned % ts != 0) {
        k_aligned += ts - (k_aligned % ts);
    }

    int weights_per_matrix = count_of_weights / z;

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &m1);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &n1_changed);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(int), (void *) &m1_changed);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(int), (void *) &nM);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(int), (void *) &mM);
    clSetKernelArg(cl_vars.kernel, 7, sizeof(int), (void *) &k_aligned);
    clSetKernelArg(cl_vars.kernel, 8, sizeof(int), (void *) &ts);
    clSetKernelArg(cl_vars.kernel, 9, sizeof(int), (void *) &blocks_counter_row);
    clSetKernelArg(cl_vars.kernel, 10, sizeof(int), (void *) &weights_per_matrix);
    clSetKernelArg(cl_vars.kernel, 11, sizeof(cl_mem), (void *) &A_cl);
    clSetKernelArg(cl_vars.kernel, 12, sizeof(cl_mem), (void *) &Filter_cl);
    clSetKernelArg(cl_vars.kernel, 13, sizeof(cl_mem), (void *) &C_cl);
    clSetKernelArg(cl_vars.kernel, 14, ts * ts * sizeof(float), nullptr);
    clSetKernelArg(cl_vars.kernel, 15, ts * ts * sizeof(float), nullptr);

    size_t global_size[3];
    size_t local_size[3];

    global_size[0] = count_of_weights;
    global_size[1] = n1_changed;
    global_size[2] = mM;

    if(global_size[1] % ts != 0) {
        global_size[1] += ts - (global_size[1] % ts);
    }
    if(global_size[2] % ts != 0) {
        global_size[2] += ts - (global_size[2] % ts);
    }

    local_size[0] = 1;
    local_size[1] = ts;
    local_size[2] = ts;

    auto time_start = std::chrono::high_resolution_clock::now();

    cl_vars.clStatus |= clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 3, nullptr,
                                               global_size, local_size, 0, nullptr, nullptr);
    CL_CHECK(clFinish(cl_vars.command_queue));

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    cl_vars.clStatus |= clEnqueueReadBuffer(cl_vars.command_queue, C_cl, CL_TRUE, 0,
                                            count_of_weights * n2 * m2 * sizeof(float), C, 0, nullptr, nullptr);

    std::cout << "kernels took " << elapsed << " ms to execute" << std::endl;

    clReleaseMemObject(A_cl);
    clReleaseMemObject(Filter_cl);
    clReleaseMemObject(C_cl);
}

std::vector<float> make_convolution_3d(CLVars& cl_vars) {

    opencl_environment_definition(cl_vars, "kernels/kernel_conv_3d.cl");

    //input parameters
    int n = rand() % 500 + 100, m = rand() % 500 + 100, z = 32, count_of_weights = 64;
    int n1 = 20, m1 = 20;
    int ts = 15;

    /*int n = 4, m = 4, z = 5, count_of_weights = 10;
    int n1 = 3, m1 = 3;
    int ts = 2;*/

    std::cout << "convolution" << std::endl;
    std::cout << "n: " << n << " m: " << m << " block_x: " <<
              n1 << " block_y " << m1 << " ts: " << ts << " z: " << z << std::endl;

    if(n < n1 || m < m1 || count_of_weights % z != 0) {
        throw "Incorrect parameters of the kernel";
    }

    int weights_per_matrix = count_of_weights / z;

    int n2 = n - n1 + 1;
    int m2 = m - m1 + 1;

    std::vector<float> A(z * n * m);
    std::vector<float> Filter(count_of_weights * n1 * m1);
    std::vector<float> C(count_of_weights * n2 * m2);

    for(int k = 0; k < z; ++k) {
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; ++j) {
                A[k * n * m + i * m + j] = 3.1 * (float)(rand() % 3 + 1);;
            }
        }
    }

    for(int k = 0; k < count_of_weights; ++k) {
        for (size_t i = 0; i < n1; i++) {
            for (size_t j = 0; j < m1; ++j) {
                Filter[k * n1 * m1 + i * m1 + j] = 2.3 * (float)(rand() % 3 + 1);;
            }
        }
    }

    opencl_create_program_conv_3d(cl_vars, "matrix_convolutional_transformation", A.data(),
                                  Filter.data(), C.data(), n, m, n1, m1, n2, m2,
                                  n1 * m1, n2 * m2, ts, m2, z, count_of_weights);

    std::vector<float> A_copy(n * m);
    std::vector<float> Filter_copy(n1 * m1);
    std::vector<float> C_copy(n2 * m2);

    for(int k = 0; k < count_of_weights; ++k) {
        std::copy(A.begin() + (k / weights_per_matrix) * n * m,
                  A.begin() + (k / weights_per_matrix + 1) * n * m, A_copy.begin());
        std::copy(Filter.begin() + k * n1 * m1, Filter.begin() + (k + 1) * n1 * m1,
                  Filter_copy.begin());
        std::copy(C.begin() + k * n2 * m2, C.begin() + (k + 1) * n2 * m2, C_copy.begin());

        /*print_matrix(A_copy, n, m);
        print_matrix(Filter_copy, n1, m1);
        print_matrix(C_copy, n2, m2);*/

        assert(test_convolution_valid(n, m, n1, m1, n2, m2, A_copy, Filter_copy, C_copy));
    }

    return C;
}

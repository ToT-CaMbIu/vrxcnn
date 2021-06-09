#include "hosts_matrix_mul.h"

void opencl_create_program_matrix_mul(CLVars& cl_vars,
                                      const char* kernel_name,
                                      float *A,
                                      float *B,
                                      float *C,
                                      int n, int m,
                                      int k,
                                      int ts) {

    cl_mem A_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                 n * k * sizeof(float), nullptr, &cl_vars.clStatus);
    cl_mem B_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                 k * m * sizeof(float), nullptr, &cl_vars.clStatus);
    cl_mem C_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
                                 n * m * sizeof(float), nullptr, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_cl, CL_TRUE, 0,
                         n * k * sizeof(float), A, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(cl_vars.command_queue, B_cl, CL_TRUE, 0,
                         k * m * sizeof(float), B, 0, nullptr, nullptr);

    CL_CHECK(clBuildProgram(cl_vars.program, 1, cl_vars.device_list, "-I..", nullptr, nullptr));

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    int k1 = k;

    if(k % ts != 0) {
        k += ts - (k % ts);
    }

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &k);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &k1);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(int), (void *) &ts);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &A_cl);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(cl_mem), (void *) &B_cl);
    clSetKernelArg(cl_vars.kernel, 7, sizeof(cl_mem), (void *) &C_cl);
    clSetKernelArg(cl_vars.kernel, 8, ts * ts * sizeof(float), nullptr);
    clSetKernelArg(cl_vars.kernel, 9, ts * ts * sizeof(float), nullptr);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;

    if(global_size[0] % ts != 0) {
        global_size[0] += ts - (global_size[0] % ts);
    }
    if(global_size[1] % ts != 0) {
        global_size[1] += ts - (global_size[1] % ts);
    }

    local_size[0] = ts;
    local_size[1] = ts;

    auto time_start = std::chrono::high_resolution_clock::now();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, nullptr,
                                    global_size, local_size, 0, nullptr, nullptr));

    CL_CHECK(clFinish(cl_vars.command_queue));

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    clEnqueueReadBuffer(cl_vars.command_queue, C_cl, CL_TRUE, 0,
                        n * m * sizeof(float), C, 0, nullptr, nullptr);

    std::cout << "kernels took " << elapsed << " ms to execute" << std::endl;

    clReleaseMemObject(A_cl);
    clReleaseMemObject(B_cl);
    clReleaseMemObject(C_cl);
}

std::vector<float> make_matrix_mul(CLVars& cl_vars,
                                   const std::vector<std::vector<float>>& A,
                                   const std::vector<std::vector<float>>& B) {

    opencl_environment_definition(cl_vars, "kernels/kernel_matrix_mul.cl");

    int n = A.size(), m = B[0].size(), k = A[0].size(), ts = 15;

    std::cout << "matrix multiplication" << std::endl;
    std::cout << "n: " << n << " m: " << m << " k: " << k << " ts: " << ts << std::endl;

    std::vector<float> A_copy(n * k);
    std::vector<float> B_copy(k * m);
    std::vector<float> C(n * m);

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < k; ++j) {
            A_copy[i * k + j] = A[i][j];
        }
    }

    for (size_t i = 0; i < k; i++) {
        for(size_t j = 0; j < m; ++j) {
            B_copy[i * m + j] = B[i][j];
        }
    }

    opencl_create_program_matrix_mul(cl_vars, "matrix_mul",
                                     A_copy.data(), B_copy.data(), C.data(), n, m, k, ts);

    assert(test_matrix_mul(n, m, k, A_copy, B_copy, C));

    return C;
}

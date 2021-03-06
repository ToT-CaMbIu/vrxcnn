#include "hosts_max_pool.h"

void opencl_create_program_max_pool(CLVars& cl_vars,
                                    const char* kernel_name,
                                    float *A,
                                    float *C,
                                    int n, int m) {
    int nc = n;
    int mc = m;
    n = (n + (n & 1));
    m = (m + (m & 1));

    int n1 = n / 2;
    int m1 = m / 2;

    cl_mem A_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                 nc * mc * sizeof(float), nullptr, &cl_vars.clStatus);
    cl_mem C_cl = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY,
                                 n1 * m1 * sizeof(float), nullptr, &cl_vars.clStatus);

    CL_CHECK(clEnqueueWriteBuffer(cl_vars.command_queue, A_cl, CL_TRUE, 0,
                                  nc * mc * sizeof(float), A, 0, nullptr, nullptr));

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, "-I..", nullptr, nullptr);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &nc);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &mc);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(cl_mem), (void *) &A_cl);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &C_cl);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;
    local_size[0] = 2;
    local_size[1] = 2;

    auto time_start = std::chrono::high_resolution_clock::now();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, nullptr,
                                    global_size, local_size, 0, nullptr, nullptr));
    CL_CHECK(clFinish(cl_vars.command_queue));

    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, C_cl, CL_TRUE, 0,
                                 n1 * m1 * sizeof(float), C, 0, nullptr, nullptr));

    std::cout << "kernels took " << elapsed << " ms to execute" << std::endl;

    clReleaseMemObject(A_cl);
    clReleaseMemObject(C_cl);
}

std::vector<float> make_max_pool(CLVars& cl_vars,
                                 bool standard_definition) {

    if(standard_definition) {
        opencl_environment_definition(cl_vars, "kernels/kernel_max_pool.cl");
    }

    int n = rand() % 1000 + 3, m = rand() % 1122 + 3;

    std::cout << "max pooling" << std::endl;
    std::cout << "n: " << n << " m: " << m << std::endl;

    int n1 = (n + (n & 1)) / 2;
    int m1 = (m + (m & 1)) / 2;

    std::vector<float> A(n * m);
    std::vector<float> C(n1 * m1);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            A[i * m + j] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
        }
    }

    opencl_create_program_max_pool(cl_vars, "matrix_max_pool_transformation",
                                   A.data(), C.data(), n, m);

    assert(test_max_pool(n, m, n1, m1, A, C));

    return C;
}

void opencl_create_program_max_pool_3d(CLVars& cl_vars,
                                    const char* kernel_name,
                                    float *A,
                                    float *C,
                                    int n, int m, int z) {
    int nc = n;
    int mc = m;
    n = (n + (n & 1));
    m = (m + (m & 1));

    int n1 = n / 2;
    int m1 = m / 2;

    cl_mem A_cl = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                 z * nc * mc * sizeof(float), nullptr, &cl_vars.clStatus);
    cl_mem C_cl = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY,
                                 z * n1 * m1 * sizeof(float), nullptr, &cl_vars.clStatus);

    CL_CHECK(clEnqueueWriteBuffer(cl_vars.command_queue, A_cl, CL_TRUE, 0,
                                  z * nc * mc * sizeof(float), A, 0, nullptr, nullptr));

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, "-I..", nullptr, nullptr);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    size_t thread_skip = z / cl_vars.work_per_thread;
    thread_skip += (z % cl_vars.work_per_thread != 0);

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &z);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &nc);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(int), (void *) &mc);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(int), (void *) &thread_skip);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(cl_mem), (void *) &A_cl);
    clSetKernelArg(cl_vars.kernel, 7, sizeof(cl_mem), (void *) &C_cl);

    size_t global_size[3];
    size_t local_size[3];

    global_size[0] = thread_skip;
    global_size[1] = n;
    global_size[2] = m;

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
                                 z * n1 * m1 * sizeof(float), C, 0, nullptr, nullptr));

    std::cout << "kernels took " << elapsed << " ms to execute" << std::endl;

    clReleaseMemObject(A_cl);
    clReleaseMemObject(C_cl);
}

Tensor<float> make_max_pool_3d(CLVars& cl_vars,
                               const Tensor<float>& tensor,
                               bool standard_definition) {

    if(standard_definition) {
        opencl_environment_definition(cl_vars, "kernels/kernel_max_pool_3d.cl");
    }

    int n = tensor.get_x(), m = tensor.get_y(), z = tensor.get_z();

    std::cout << "max pooling 3d" << std::endl;
    std::cout << "x: " << n << " y: " << m << " z: " << z << std::endl;

    int n1 = (n + (n & 1)) / 2;
    int m1 = (m + (m & 1)) / 2;

    std::vector<float> A = tensor.get_data();
    std::vector<float> C(z * n1 * m1);

    opencl_create_program_max_pool_3d(cl_vars, "matrix_max_pool_transformation_3d",
                                      A.data(), C.data(), n, m, z);

    std::vector<float> A_copy(n * m);
    std::vector<float> C_copy(n1 * m1);

    double elapsed = 0.0;

    for(int k = 0; k < z; ++k) {
        std::copy(A.begin() + k * n * m, A.begin() + (k + 1) * n * m, A_copy.begin());
        std::copy(C.begin() + k * n1 * m1, C.begin() + (k + 1) * n1 * m1, C_copy.begin());

        auto time_start = std::chrono::high_resolution_clock::now();
        assert(test_max_pool(n, m, n1, m1, A_copy, C_copy));
        auto time_end = std::chrono::high_resolution_clock::now();
        elapsed += std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    }

    std::cout << "cpu took " << elapsed << " ms to execute" << std::endl;

    std::cout << "n: " << n1 << " m1: " << m1 << std::endl;

    return Tensor(z, n1, m1, std::move(C));
}
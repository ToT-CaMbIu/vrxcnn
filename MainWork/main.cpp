#include "connected_libs.h"

#include "utils.h"

double time_taken = 0.0;
double eps = 1e-7;

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     break;                                                            \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       break;                                                          \
     }                                                                 \
     _ret;                                                             \
   })

//OpenCl
struct CLVars {
    cl_platform_id *platforms = nullptr;
    cl_uint num_platforms;
    cl_int clStatus;
    cl_device_id *device_list = nullptr;
    cl_uint num_devices;
    cl_context context = nullptr;
    cl_kernel kernel = nullptr;
    cl_command_queue command_queue = nullptr;
    cl_program program = nullptr;
    char *kernel_string = nullptr;

    //vortex
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
};
//

void cl_clean(CLVars& cl_vars) {
    if(cl_vars.device != nullptr) {
        clReleaseDevice(cl_vars.device);
    }
    if (cl_vars.command_queue != nullptr) {
        clReleaseCommandQueue(cl_vars.command_queue);
    }
    if (cl_vars.kernel != nullptr) {
        clReleaseKernel(cl_vars.kernel);
    }
    if (cl_vars.program != nullptr) {
        clReleaseProgram(cl_vars.program);
    }
    if (cl_vars.context != nullptr) {
        clReleaseContext(cl_vars.context);
    }
    if(cl_vars.platforms != nullptr) {
        free(cl_vars.platforms);
    }
    if(cl_vars.device_list != nullptr) {
        free(cl_vars.device_list);
    }
}

void opencl_environment_definition_vortex(CLVars& cl_vars,
                                   const char* binary_source) {

    //currently in work!!!
    uint8_t *kernel_bin = NULL;
    size_t kernel_size;

    clGetPlatformIDs(1, &cl_vars.platform, NULL);
    clGetDeviceIDs(cl_vars.platform, CL_DEVICE_TYPE_DEFAULT, 1, &cl_vars.device, NULL);
    cl_vars.context = clCreateContext(NULL, 1, &cl_vars.device, NULL, NULL, &cl_vars.clStatus);

    if (read_kernel_binary(binary_source, &kernel_bin, &kernel_size) == false) {
        return;
    }

    cl_vars.program = clCreateProgramWithBinary(cl_vars.context, 1, &cl_vars.device, &kernel_size,
                                                (const uint8_t**)&kernel_bin, &cl_vars.clStatus, NULL);
    if (cl_vars.program == NULL) {
        printf("Binary file load failed!");
        return;
    }
    clBuildProgram(cl_vars.program, 1, &cl_vars.device, NULL, NULL, NULL);
    cl_vars.command_queue = clCreateCommandQueue(cl_vars.context, cl_vars.device, 0, &cl_vars.clStatus);
}

void opencl_environment_definition(CLVars& cl_vars,
                                   const char* kernel_source) {
    clGetPlatformIDs(0, NULL, &cl_vars.num_platforms);
    cl_vars.platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * cl_vars.num_platforms);
    clGetPlatformIDs(cl_vars.num_platforms, cl_vars.platforms, NULL);
    clGetDeviceIDs(cl_vars.platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &cl_vars.num_devices);
    cl_vars.device_list = (cl_device_id *) malloc(sizeof(cl_device_id) * cl_vars.num_devices);
    clGetDeviceIDs(cl_vars.platforms[0], CL_DEVICE_TYPE_GPU, cl_vars.num_devices, cl_vars.device_list, NULL);
    cl_vars.context = clCreateContext(NULL, cl_vars.num_devices, cl_vars.device_list, NULL, NULL, &cl_vars.clStatus);
    cl_vars.command_queue = clCreateCommandQueue(cl_vars.context, cl_vars.device_list[0], 0, &cl_vars.clStatus);

    if(cl_vars.kernel_string == nullptr) {
        cl_vars.kernel_string = read_kernel_from_file(kernel_source);
    }
    const char* cKernel_string = cl_vars.kernel_string;

    cl_vars.program = clCreateProgramWithSource(cl_vars.context, 1, &cKernel_string, NULL, &cl_vars.clStatus);
}

void opencl_create_program_conv(CLVars& cl_vars,
                                const char* kernel_name,
                                float *A,
                                float *Filter,
                                float *C,
                                int n, int m, int n1, int m1)   {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, n * m * sizeof(float),
                                    NULL, &cl_vars.clStatus);
    cl_mem Filter_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY, n1 * m1 * sizeof(float),
                                         NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY, n * m * sizeof(float), NULL,
                                    &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, Filter_clmem, CL_TRUE, 0,
                                            n1 * m1 * sizeof(float), Filter, 0, NULL, NULL);

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &n1);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &m1);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &Filter_clmem);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2];
    global_size[0] = n;
    global_size[1] = m;

    clock_t t;
    t = clock();

    cl_vars.clStatus |= clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, NULL,
                                              global_size, NULL, 0, NULL, NULL);

    cl_vars.clStatus |= clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           n * m * sizeof(float), C, 0, NULL, NULL);

    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(Filter_clmem);
    clReleaseMemObject(C_clmem);
}

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

    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    nc * mc * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY,
                                    n1 * m1 * sizeof(float), NULL, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                            nc * mc * sizeof(float), A, 0, NULL, NULL);

    clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &nc);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &mc);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;
    local_size[0] = 2;
    local_size[1] = 2;

    clock_t t;
    t = clock();

    clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, NULL,
                                              global_size, local_size, 0, NULL, NULL);
    CL_CHECK(clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           n1 * m1 * sizeof(float), C, 0, NULL, NULL));

    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(C_clmem);
}

void opencl_create_program_matrix_mul(CLVars& cl_vars,
                                      const char* kernel_name,
                                      float *A,
                                      float *B,
                                      float *C,
                                      int n, int m, int k, int TS) {

    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    n * k * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem B_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    k * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
                                    n * m * sizeof(float), NULL, &cl_vars.clStatus);

    clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                             n * k * sizeof(float), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(cl_vars.command_queue, B_clmem, CL_TRUE, 0,
                                             k * m * sizeof(float), B, 0, NULL, NULL);

    CL_CHECK(clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL));

    cl_vars.kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    int k1 = k;

    if(k % TS != 0) {
        k += TS - (k % TS);
    }

    std::cout << k << std::endl;

    clSetKernelArg(cl_vars.kernel, 0, sizeof(int), (void *) &n);
    clSetKernelArg(cl_vars.kernel, 1, sizeof(int), (void *) &m);
    clSetKernelArg(cl_vars.kernel, 2, sizeof(int), (void *) &k);
    clSetKernelArg(cl_vars.kernel, 3, sizeof(int), (void *) &k1);
    clSetKernelArg(cl_vars.kernel, 4, sizeof(int), (void *) &TS);
    clSetKernelArg(cl_vars.kernel, 5, sizeof(cl_mem), (void *) &A_clmem);
    clSetKernelArg(cl_vars.kernel, 6, sizeof(cl_mem), (void *) &B_clmem);
    clSetKernelArg(cl_vars.kernel, 7, sizeof(cl_mem), (void *) &C_clmem);
    clSetKernelArg(cl_vars.kernel, 8, TS * TS * sizeof(float), NULL);
    clSetKernelArg(cl_vars.kernel, 9, TS * TS * sizeof(float), NULL);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;

    if(global_size[0] % TS != 0) {
        global_size[0] += TS - (global_size[0] % TS);
    }
    if(global_size[1] % TS != 0) {
        global_size[1] += TS - (global_size[1] % TS);
    }

    local_size[0] = TS;
    local_size[1] = TS;

    clock_t t;
    t = clock();

    CL_CHECK(clEnqueueNDRangeKernel(cl_vars.command_queue, cl_vars.kernel, 2, NULL,
                                               global_size, local_size, 0, NULL, NULL));

    CL_CHECK(clFinish(cl_vars.command_queue));

    clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), C, 0, NULL, NULL);

    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clReleaseMemObject(A_clmem);
    clReleaseMemObject(B_clmem);
    clReleaseMemObject(C_clmem);
}

std::vector<float> make_matrix_mul(CLVars& cl_vars) {
    opencl_environment_definition(cl_vars, "kernel_matrix_mul.cl");

    int n = rand() % 1000 + 3, m = rand() % 1000 + 3, k = rand() % 1000 + 3, TS = 8;

    std::cout << n << " " << m << " " << k << std::endl;

    std::vector<float> A(n * k);
    std::vector<float> B(k * m);
    std::vector<float> C(n * m);

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < k; ++j) {
            A[i * k + j] = 3.1 * (float)(rand() % 3 + 1);
        }
    }

    for (size_t i = 0; i < k; i++) {
        for(size_t j = 0; j < m; ++j) {
            B[i * m + j] = 3.3 * (float)(rand() % 3 + 1);
        }
    }

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; ++j) {
            C[i * m + j] = 0.0;
        }
    }

    //print_matrix(A, n, k);
    //print_matrix(B, k, m);

    opencl_create_program_matrix_mul(cl_vars, "matrix_mul",
                                     A.data(), B.data(), C.data(), n, m, k, TS);

    //print_matrix(C, n, m);

    assert(test_matrix_mul(n, m, k, A, B, C, eps));

    printf("kernels took %f seconds to execute \n", time_taken);

    time_taken = 0.0f;

    return C;
}

std::vector<float> make_convolution(CLVars& cl_vars) {
     opencl_environment_definition(cl_vars, "kernel_conv.cl");

     int n = 1000, m = 1000;
     int n1 = 3, m1 = 3;

     std::vector<float> A(n * m);
     std::vector<float> Filter(n1 * m1);
     std::vector<float> C(n * m);

     for (size_t i = 0; i < n; i++) {
         for(size_t j = 0; j < m; ++j) {
             A[i * m + j] = 2.0 * (rand() % (j + 1));
             C[i * m + j] = 0;
         }
     }

     for (size_t i = 0; i < n1; i++) {
         for(size_t j = 0; j < m1; ++j) {
             Filter[i * m1 + j] = 1.0;
         }
     }

     opencl_create_program_conv(cl_vars, "matrix_convolutional_transformation", A.data(),
                                Filter.data(), C.data(), n, m, n1, m1);

     assert(test_convolution(n, m, n1, m1, A, Filter, C));

     printf("kernels took %f seconds to execute \n", time_taken);

     time_taken = 0.0f;

     return C;
 }

std::vector<float> make_max_pool(CLVars& cl_vars) {
    opencl_environment_definition(cl_vars, "kernel_max_pool.cl");

    int n = rand() % 1000 + 3, m = rand() % 1122 + 3;

    std::cout << n << " " << m << std::endl;

    int n1 = (n + (n & 1)) / 2;
    int m1 = (m + (m & 1)) / 2;

    std::vector<float> A(n * m);
    std::vector<float> C(n1 * m1);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            A[i * m + j] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
        }
    }
    
    //print_matrix(A, n, m);

    opencl_create_program_max_pool(cl_vars, "matrix_max_pool_transformation",
                                   A.data(), C.data(), n, m);
    
    //print_matrix(C, n1, m1);

    assert(test_max_pool(n, m, n1, m1, A, C));

    printf("kernels took %f seconds to execute \n", time_taken);

    time_taken = 0.0f;

    return C;
}

int main (int argc, char **argv) {

    srand(time(nullptr));

    CLVars cl_vars;

    for(int i = 0; i < 100; ++i) {
        make_max_pool(cl_vars);
        cl_clean(cl_vars);
    }

    free(cl_vars.kernel_string);

    return 0;
}

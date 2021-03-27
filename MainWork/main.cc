#define CL_TARGET_OPENCL_VERSION 120
#include "connected_libs.h"

#include "utils.h"
#include "hdf5.h"

const int mx_size = 10000;
double time_taken = 0.0;
double eps = 1e-7;
//OpenCl
struct CLVars {
    cl_platform_id *platforms;
    cl_uint num_platforms;
    cl_int clStatus;
    cl_device_id *device_list;
    cl_uint num_devices;
    cl_context context;
    cl_command_queue command_queue;
    cl_program program;
    char *kernel_string;
};
//

CLVars cl_vars;

void opencl_environment_definition(char* kernel_source) {
    cl_int clStatus = clGetPlatformIDs(0, NULL, &cl_vars.num_platforms);
    cl_vars.platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * cl_vars.num_platforms);
    clStatus = clGetPlatformIDs(cl_vars.num_platforms, cl_vars.platforms, NULL);
    clStatus = clGetDeviceIDs(cl_vars.platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &cl_vars.num_devices);
    cl_vars.device_list = (cl_device_id *) malloc(sizeof(cl_device_id) * cl_vars.num_devices);
    clStatus = clGetDeviceIDs(cl_vars.platforms[0], CL_DEVICE_TYPE_GPU, cl_vars.num_devices, cl_vars.device_list, NULL);
    cl_vars.context = clCreateContext(NULL, cl_vars.num_devices, cl_vars.device_list, NULL, NULL, &clStatus);
    cl_vars.command_queue = clCreateCommandQueue(cl_vars.context, cl_vars.device_list[0], 0, &clStatus);

    if(cl_vars.kernel_string == nullptr) {
        cl_vars.kernel_string = read_kernel_from_file(kernel_source);
    }
    const char* cKernel_string = cl_vars.kernel_string;

    cl_vars.program = clCreateProgramWithSource(cl_vars.context, 1, &cKernel_string, NULL, &clStatus);
}

void opencl_environment_clear() {
    cl_vars.clStatus = clReleaseContext(cl_vars.context);
    cl_vars.clStatus = clReleaseCommandQueue(cl_vars.command_queue);
}

void opencl_create_program_conv(char* kernel_name,
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

    cl_vars.clStatus = clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), A, 0, NULL, NULL);
    cl_vars.clStatus = clEnqueueWriteBuffer(cl_vars.command_queue, Filter_clmem, CL_TRUE, 0,
                                            n1 * m1 * sizeof(float), Filter, 0, NULL, NULL);

    cl_vars.clStatus = clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    cl_vars.clStatus &= clSetKernelArg(kernel, 0, sizeof(int), (void *) &n);
    cl_vars.clStatus &= clSetKernelArg(kernel, 1, sizeof(int), (void *) &m);
    cl_vars.clStatus &= clSetKernelArg(kernel, 2, sizeof(int), (void *) &n1);
    cl_vars.clStatus &= clSetKernelArg(kernel, 3, sizeof(int), (void *) &m1);
    cl_vars.clStatus &= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    cl_vars.clStatus &= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &Filter_clmem);
    cl_vars.clStatus &= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2];
    global_size[0] = n;
    global_size[1] = m;

    clock_t t;
    t = clock();

    cl_vars.clStatus &= clEnqueueNDRangeKernel(cl_vars.command_queue, kernel, 2, NULL,
                                              global_size, NULL, 0, NULL, NULL);

    cl_vars.clStatus &= clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           n * m * sizeof(float), C, 0, NULL, NULL);

    cl_vars.clStatus &= clFlush(cl_vars.command_queue);
    cl_vars.clStatus &= clFinish(cl_vars.command_queue);
    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    cl_vars.clStatus &= clReleaseKernel(kernel);
    cl_vars.clStatus &= clReleaseProgram(cl_vars.program);
    cl_vars.clStatus &= clReleaseMemObject(A_clmem);
    cl_vars.clStatus &= clReleaseMemObject(Filter_clmem);
    cl_vars.clStatus &= clReleaseMemObject(C_clmem);
}

void opencl_create_program_max_pool(char* kernel_name,
                                    float *A,
                                    float *C,
                                    int n, int m) {
    int n1 = n / 2;
    int m1 = m / 2;

    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    n * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY,
                                    n1 * m1 * sizeof(float), NULL, &cl_vars.clStatus);

    cl_vars.clStatus &= clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), A, 0, NULL, NULL);

    cl_vars.clStatus &= clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    cl_vars.clStatus &= clSetKernelArg(kernel, 0, sizeof(int), (void *) &n);
    cl_vars.clStatus &= clSetKernelArg(kernel, 1, sizeof(int), (void *) &m);
    cl_vars.clStatus &= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &A_clmem);
    cl_vars.clStatus &= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;
    local_size[0] = 2;
    local_size[1] = 2;

    clock_t t;
    t = clock();

    cl_vars.clStatus &= clEnqueueNDRangeKernel(cl_vars.command_queue, kernel, 2, NULL,
                                              global_size, local_size, 0, NULL, NULL);
    cl_vars.clStatus &= clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           n1 * m1 * sizeof(float), C, 0, NULL, NULL);

    cl_vars.clStatus &= clFlush(cl_vars.command_queue);
    cl_vars.clStatus &= clFinish(cl_vars.command_queue);
    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    cl_vars.clStatus &= clReleaseKernel(kernel);
    cl_vars.clStatus &= clReleaseProgram(cl_vars.program);
    cl_vars.clStatus &= clReleaseMemObject(A_clmem);
    cl_vars.clStatus &= clReleaseMemObject(C_clmem);
}

void opencl_create_program_matrix_mul(char* kernel_name,
                                    float *A,
                                    float *B,
                                    float *C,
                                    int n, int m, int k) {
    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    n * k * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem B_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    k * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_WRITE,
                                    n * m * sizeof(float), NULL, &cl_vars.clStatus);

    cl_vars.clStatus &= clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                             n * k * sizeof(float), A, 0, NULL, NULL);

    cl_vars.clStatus &= clEnqueueWriteBuffer(cl_vars.command_queue, B_clmem, CL_TRUE, 0,
                                             k * m * sizeof(float), B, 0, NULL, NULL);

    cl_vars.clStatus &= clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    int TS = k / 2;

    cl_vars.clStatus &= clSetKernelArg(kernel, 0, sizeof(int), (void *) &n);
    cl_vars.clStatus &= clSetKernelArg(kernel, 1, sizeof(int), (void *) &m);
    cl_vars.clStatus &= clSetKernelArg(kernel, 2, sizeof(int), (void *) &k);
    cl_vars.clStatus &= clSetKernelArg(kernel, 3, sizeof(int), (void *) &TS);
    cl_vars.clStatus &= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    cl_vars.clStatus &= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &B_clmem);
    cl_vars.clStatus &= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &C_clmem);
    cl_vars.clStatus &= clSetKernelArg(kernel, 7, TS * TS * sizeof(float), NULL);
    cl_vars.clStatus &= clSetKernelArg(kernel, 8, TS * TS * sizeof(float), NULL);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;
    local_size[0] = TS;
    local_size[1] = TS;

    cl_event event = NULL;

    clock_t t;
    t = clock();

    cl_vars.clStatus &= clEnqueueNDRangeKernel(cl_vars.command_queue, kernel, 2, NULL,
                                               global_size, local_size, 0, NULL, &event);

    cl_vars.clStatus &= clWaitForEvents(1, &event);

    cl_vars.clStatus &= clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), C, 0, NULL, NULL);

    cl_vars.clStatus &= clFlush(cl_vars.command_queue);
    cl_vars.clStatus &= clFinish(cl_vars.command_queue);

    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    cl_vars.clStatus &= clReleaseKernel(kernel);
    cl_vars.clStatus &= clReleaseProgram(cl_vars.program);
    cl_vars.clStatus &= clReleaseMemObject(A_clmem);
    cl_vars.clStatus &= clReleaseMemObject(B_clmem);
    cl_vars.clStatus &= clReleaseMemObject(C_clmem);
}

float* make_matrix_mul() {
    opencl_environment_definition("kernel_matrix_mul.cl");

    int n = 2, m = 2, k = 3;

    float *A = (float *) malloc(sizeof(float) * n * k);
    float *B = (float *) malloc(sizeof(float) * k * m);
    float *C = (float *) malloc(sizeof(float) * n * m);

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < k; ++j) {
            A[i * k + j] = 1.0 + rand() % 3;
        }
    }

    for (size_t i = 0; i < k; i++) {
        for(size_t j = 0; j < m; ++j) {
            B[i * m + j] = 1.0 + rand() % 3;
        }
    }

    for (size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < m; ++j) {
            C[i * m + j] = 0.0;
        }
    }

    print_matrix(A, n, k);
    print_matrix(B, k, m);

    opencl_create_program_matrix_mul("matrix_mul", A, B, C, n, m, k);

    test_matrix_mul(n, m, k, A, B, C);

    print_matrix(C, n, m);

    printf("kernels took %f seconds to execute \n", time_taken);

    time_taken = 0.0f;

    opencl_environment_clear();

    free(A);
    free(B);
    free(C);
    free(cl_vars.platforms);
    free(cl_vars.device_list);

    return C;
}

float* make_convolution() {
     opencl_environment_definition("kernel_conv.cl");

     int n = 1000, m = 1000;
     int n1 = 3, m1 = 3;

     float *A = (float *) malloc(sizeof(float) * n * m);
     float *Filter = (float *) malloc(sizeof(float) * n1 * m1);
     float *C = (float *) malloc(sizeof(float) * n * m);

     for (size_t i = 0; i < n; i++) {
         for(size_t j = 0; j < m; ++j) {
             A[i * m + j] = j;
             C[i * m + j] = 0;
         }
     }

     for (size_t i = 0; i < n1; i++) {
         for(size_t j = 0; j < m1; ++j) {
             Filter[i * m1 + j] = 1;
         }
     }

     opencl_create_program_conv("matrix_convolutional_transformation", A, Filter, C, n, m, n1, m1);

     test_convolution(n, m, n1, m1, A, Filter, C);

     printf("kernels took %f seconds to execute \n", time_taken);

     opencl_environment_clear();

     free(A);
     free(Filter);
     free(C);
 }

float* make_max_pool() {
    opencl_environment_definition("kernel_max_pool.cl");

    int n = rand() % 9997 + 3, m = rand() % 9997 + 3;
    printf("n = %d, m = %d\n", n, m);
    
    int nc = n + (n & 1), mc = m + (m & 1);

    int n1 = nc / 2;
    int m1 = mc / 2;

    float *A = (float *) malloc(sizeof(float) * nc * mc);
    float *C = (float *) malloc(sizeof(float) * n1 * m1);

    int pos = 0;
    for (int i = 0; i < nc; ++i) {
        for (int j = 0; j < mc; ++j) {
            if (i >= n || j >= m) {
                A[pos++] = 0.0;
                continue;
            }
            A[pos++] = rand() % 3 + 1.0 / (1.0 + rand() % 3);
        }
    }
    
    //print_matrix(A, mc, nc);

    opencl_create_program_max_pool("matrix_max_pool_transformation", A, C, nc, mc);
    
    //print_matrix(C, m1, n1);

    test_max_pool(nc, mc, n1, m1, A, C);

    printf("kernels took %f seconds to execute \n", time_taken);

    time_taken = 0.0f;

    opencl_environment_clear();

    free(A);
    free(C);
    free(cl_vars.platforms);
    free(cl_vars.device_list);

    return C;
}

int main (int argc, char **argv) {

    srand(time(nullptr));
    
    for(int i = 0; i < 100; ++i)
        make_matrix_mul();

    return 0;
}

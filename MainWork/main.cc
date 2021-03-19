#define CL_TARGET_OPENCL_VERSION 120
#include "connected_libs.h"

#include "utils.h"

#define KERNEL_NAME "kernel.cl"

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

    char *kernel_string = read_kernel_from_file(kernel_source);
    const char* cKernel_string = kernel_string;

    cl_vars.program = clCreateProgramWithSource(cl_vars.context, 1, &cKernel_string, NULL, &clStatus);
}

void opencl_environment_clear() {
    cl_vars.clStatus = clReleaseContext(cl_vars.context);
    cl_vars.clStatus = clReleaseCommandQueue(cl_vars.command_queue);
}

/*
void opencl_create_program_vector(char* kernel_source, char* kernel_name, float *A, float *B, float *C) {
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, sz * sizeof(float), NULL, &clStatus);
    cl_mem B_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, sz * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sz * sizeof(float), NULL, &clStatus);

    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, sz * sizeof(float), A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, sz * sizeof(float), B, 0, NULL, NULL);

    char *kernel_string = read_kernel_from_file(kernel_source);
    const char* cKernel_string = kernel_string;

    cl_program program = clCreateProgramWithSource(context, 1, &cKernel_string, NULL, &clStatus);

    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, kernel_name, &clStatus);

    clStatus = clSetKernelArg(kernel, 0, sizeof(int), (void *) &sz);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &A_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &B_clmem);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size = 1024;
    size_t local_size = 16;
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, sz * sizeof(float), C, 0, NULL, NULL);

    clock_t t;
    t = clock();
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(B_clmem);
    clStatus = clReleaseMemObject(C_clmem);

    free(kernel_string);
}
*/

void opencl_create_program_conv(char* kernel_name, float *A, float *Filter, float *C,
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

    cl_vars.clStatus = clSetKernelArg(kernel, 0, sizeof(int), (void *) &n);
    cl_vars.clStatus = clSetKernelArg(kernel, 1, sizeof(int), (void *) &m);
    cl_vars.clStatus = clSetKernelArg(kernel, 2, sizeof(int), (void *) &n1);
    cl_vars.clStatus = clSetKernelArg(kernel, 3, sizeof(int), (void *) &m1);
    cl_vars.clStatus = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    cl_vars.clStatus = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &Filter_clmem);
    cl_vars.clStatus = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2];
    size_t local_size[2];

    local_size[0] = 10;
    local_size[1] = 10;
    global_size[0] = 10000;
    global_size[1] = 10000;

    cl_vars.clStatus = clEnqueueNDRangeKernel(cl_vars.command_queue, kernel, 2, NULL,
                                              global_size, local_size, 0, NULL, NULL);
    cl_vars.clStatus = clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           n * m * sizeof(float), C, 0, NULL, NULL);

    clock_t t;
    t = clock();
    cl_vars.clStatus = clFlush(cl_vars.command_queue);
    cl_vars.clStatus = clFinish(cl_vars.command_queue);
    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    cl_vars.clStatus = clReleaseKernel(kernel);
    cl_vars.clStatus = clReleaseProgram(cl_vars.program);
    cl_vars.clStatus = clReleaseMemObject(A_clmem);
    cl_vars.clStatus = clReleaseMemObject(Filter_clmem);
    cl_vars.clStatus = clReleaseMemObject(C_clmem);
}

void opencl_create_program_max_pool(char* kernel_name, float *A, float *C, int n, int m) {
    int n1 = n / 2;
    int m1 = m / 2;

    cl_mem A_clmem = clCreateBuffer(cl_vars.context, CL_MEM_READ_ONLY,
                                    n * m * sizeof(float), NULL, &cl_vars.clStatus);
    cl_mem C_clmem = clCreateBuffer(cl_vars.context, CL_MEM_WRITE_ONLY,
                                    n1 * m1 * sizeof(float), NULL, &cl_vars.clStatus);

    cl_vars.clStatus = clEnqueueWriteBuffer(cl_vars.command_queue, A_clmem, CL_TRUE, 0,
                                            n * m * sizeof(float), A, 0, NULL, NULL);

    cl_vars.clStatus = clBuildProgram(cl_vars.program, 1, cl_vars.device_list, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(cl_vars.program, kernel_name, &cl_vars.clStatus);

    cl_vars.clStatus = clSetKernelArg(kernel, 0, sizeof(int), (void *) &n);
    cl_vars.clStatus = clSetKernelArg(kernel, 1, sizeof(int), (void *) &m);
    cl_vars.clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &A_clmem);
    cl_vars.clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2];
    size_t local_size[2];

    global_size[0] = n;
    global_size[1] = m;
    local_size[0] = 2;
    local_size[1] = 2;

    cl_vars.clStatus = clEnqueueNDRangeKernel(cl_vars.command_queue, kernel, 2, NULL,
                                              global_size, local_size, 0, NULL, NULL);
    cl_vars.clStatus = clEnqueueReadBuffer(cl_vars.command_queue, C_clmem, CL_TRUE, 0,
                                           n1 * m1 * sizeof(float), C, 0, NULL, NULL);

    clock_t t;
    t = clock();
    cl_vars.clStatus = clFlush(cl_vars.command_queue);
    cl_vars.clStatus = clFinish(cl_vars.command_queue);
    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    cl_vars.clStatus = clReleaseKernel(kernel);
    cl_vars.clStatus = clReleaseProgram(cl_vars.program);
    cl_vars.clStatus = clReleaseMemObject(A_clmem);
    cl_vars.clStatus = clReleaseMemObject(C_clmem);
}

float* make_convolution() {
    opencl_environment_definition("kernel_conv.cl");

    int n = 5, m = 5;
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
            Filter[i * m1 + j] = rand() % 10;
        }
    }

    opencl_create_program_conv("matrix_convolutional_transformation", A, Filter, C, n, m, n1, m1);

    bool isPassed = true;
    for (int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            printf("%f ", C[i * m + j]);

            float val = 0;
            int x = i - 1;

            for(int i1 = 0; i1 < n1; ++i1, ++x) {
                int y = j - 1;
                for(int j1 = 0; j1 < m1; ++j1, ++y) {
                    if(x >= 0 && y >= 0 && x < n && y < m) {
                        val += (Filter[i1 * m1 + j1] * A[x * m + y]);
                    }
                }
            }

            isPassed &= val == C[i * m + j];
        }
        printf("\n");
    }

    printf("kernels took %f seconds to execute \n", time_taken);

    if(isPassed) {
        printf("Passed!\n");
    }
    else {
        printf("Failed!\n");
    }

    opencl_environment_clear();

    free(A);
    free(Filter);
    free(C);
    free(cl_vars.platforms);
    free(cl_vars.device_list);

    return C;
}

float* make_max_pool() {
    opencl_environment_definition("kernel_max_pool.cl");

    int n = 200, m = 122;
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
            A[pos++] = rand() % 10;
        }
    }

    opencl_create_program_max_pool("matrix_max_pool_transformation", A, C, nc, mc);

    for (int i = 0; i < nc; ++i) {
        for(int j = 0; j < mc; ++j) {
            printf("%f ", A[i * mc + j]);
        }
        printf("\n");
    }

    bool isPassed = true;
    for (int i = 0; i < n1; ++i) {
        for(int j = 0; j < m1; ++j) {
            float a1 = -1e9,a2 = -1e9,a3 = -1e9,a4 = -1e9;
            if(i * 2 * mc + j * 2 < mc * nc) {
                a1 = A[i * 2 * mc + j * 2];
            }
            if(i * 2 * mc + j * 2 + 1 < mc * nc) {
                a2 = A[i * 2 * mc + j * 2 + 1];
            }
            if((i * 2 + 1)* mc + j * 2 < mc * nc) {
                a3 = A[(i * 2 + 1) * mc + j * 2];
            }
            if((i * 2 + 1) * mc + (j * 2 + 1) < mc * nc) {
                a4 = A[(i * 2 + 1) * mc + (j * 2 + 1)];
            }

            a1 = fmax(a1, a2);
            a3 = fmax(a3, a4);
            a1 = fmax(a1, a3);

            printf("%f ", C[i * m1 + j]);
            isPassed &= C[i * m1 + j] == a1;
        }
        printf("\n");
    }

    printf("kernels took %f seconds to execute \n", time_taken);

    if(isPassed) {
        printf("Passed!\n");
    }
    else {
        printf("Failed!\n");
    }

    opencl_environment_clear();

    free(A);
    free(C);
    free(cl_vars.platforms);
    free(cl_vars.device_list);

    return C;
}

int main (int argc, char **argv) {

    make_max_pool();

    return 0;
}

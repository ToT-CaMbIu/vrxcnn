#define CL_TARGET_OPENCL_VERSION 120
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <time.h>

#define KERNEL_NAME "kernel.cl"

const int mx_size = 10000;
int sz = 1021;
double time_taken = 0.0;
//OpenCl
cl_platform_id *platforms;
cl_uint num_platforms;
cl_int clStatus;
cl_device_id *device_list;
cl_uint num_devices;
cl_context context;
cl_command_queue command_queue;
cl_program program;
//

char* read_kernel_from_file(char* filename) {
    FILE *fp = fopen(filename, "r");
    char* kernel_string = (char *) malloc(sizeof(char) * mx_size);
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    char ch;
    int i = 0;
    while ((ch = fgetc(fp)) != '$') {
        kernel_string[i++] = ch;
    }

    fclose(fp);

    printf("%s\n", kernel_string);

    return kernel_string;
}

void opencl_environment_definition(char* kernel_source) {
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
    clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
    context = clCreateContext(NULL, num_devices, device_list, NULL, NULL, &clStatus);
    command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

    char *kernel_string = read_kernel_from_file(kernel_source);
    const char* cKernel_string = kernel_string;

    program = clCreateProgramWithSource(context, 1, &cKernel_string, NULL, &clStatus);
}

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

void opencl_create_program_conv(char* kernel_name, float *A, float *Filter, float *C,
                                int n, int m, int n1, int m1) {
    cl_mem A_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, n * m * sizeof(float), NULL, &clStatus);
    cl_mem Filter_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY, n1 * m1 * sizeof(float), NULL, &clStatus);
    cl_mem C_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * m * sizeof(float), NULL, &clStatus);

    clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, n * m * sizeof(float), A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, Filter_clmem, CL_TRUE, 0, n1 * m1 * sizeof(float), Filter, 0, NULL, NULL);

    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, kernel_name, &clStatus);

    clStatus = clSetKernelArg(kernel, 0, sizeof(int), (void *) &n);
    clStatus = clSetKernelArg(kernel, 1, sizeof(int), (void *) &m);
    clStatus = clSetKernelArg(kernel, 2, sizeof(int), (void *) &n1);
    clStatus = clSetKernelArg(kernel, 3, sizeof(int), (void *) &m1);
    clStatus = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &A_clmem);
    clStatus = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *) &Filter_clmem);
    clStatus = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *) &C_clmem);

    size_t global_size[2] = {1024, 1024};
    size_t local_size[2]  = {64, 64};

    local_size[0] = 16;
    local_size[1] = 16;
    global_size[0] = 1024;
    global_size[1] = 1024;

    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    clStatus = clEnqueueReadBuffer(command_queue, C_clmem, CL_TRUE, 0, n * m * sizeof(float), C, 0, NULL, NULL);

    clock_t t;
    t = clock();
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    t = clock() - t;
    time_taken += ((double)t)/CLOCKS_PER_SEC; // in seconds

    clStatus = clReleaseKernel(kernel);
    clStatus = clReleaseProgram(program);
    clStatus = clReleaseMemObject(A_clmem);
    clStatus = clReleaseMemObject(Filter_clmem);
    clStatus = clReleaseMemObject(C_clmem);
}

void make_convolution() {
    opencl_environment_definition("kernel.cl");

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

    bool isPassed = true;
    for (int i = 0; i < n; i++) {
        for(int j = 0; j < m; ++j) {
            printf("%f ", C[i * n + j]);

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

            isPassed &= val == C[i * n + j];
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
    free(platforms);
    free(device_list);
}

void opencl_environment_clear() {
    clStatus = clReleaseContext(context);
    clStatus = clReleaseCommandQueue(command_queue);
}

int main (int argc, char **argv) {

    make_convolution();

    return 0;
}

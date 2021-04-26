#include "opencl_environment.h"
#include "utils.h"

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
    uint8_t *kernel_bin = nullptr;
    size_t kernel_size;

    if (read_kernel_binary(binary_source, &kernel_bin, &kernel_size) == false) {
        return;
    }

    CL_CHECK(clGetPlatformIDs(1, &cl_vars.platform, nullptr));
    CL_CHECK(clGetDeviceIDs(cl_vars.platform, CL_DEVICE_TYPE_DEFAULT, 1, &cl_vars.device, nullptr));
    cl_vars.context = CL_CHECK2(clCreateContext(NULL, 1, &cl_vars.device, nullptr, nullptr, &cl_vars.clStatus));

    cl_vars.program = CL_CHECK2(clCreateProgramWithBinary(cl_vars.context, 1, &cl_vars.device, &kernel_size,
                                                          (const uint8_t**)&kernel_bin, &cl_vars.clStatus, nullptr));
    if (cl_vars.program == nullptr) {
        std::cout << "Binary file load failed!" << std::endl;
        cl_clean(cl_vars);
        return;
    }
    CL_CHECK(clBuildProgram(cl_vars.program, 1, &cl_vars.device, nullptr, nullptr, nullptr));
    cl_vars.command_queue = CL_CHECK2(clCreateCommandQueue(cl_vars.context, cl_vars.device, 0, &cl_vars.clStatus));
}

void opencl_environment_definition(CLVars& cl_vars,
                                   const char* kernel_source) {
    clGetPlatformIDs(0, nullptr, &cl_vars.num_platforms);
    cl_vars.platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * cl_vars.num_platforms);
    clGetPlatformIDs(cl_vars.num_platforms, cl_vars.platforms, nullptr);
    clGetDeviceIDs(cl_vars.platforms[0], CL_DEVICE_TYPE_GPU, 0, nullptr, &cl_vars.num_devices);
    cl_vars.device_list = (cl_device_id *) malloc(sizeof(cl_device_id) * cl_vars.num_devices);
    clGetDeviceIDs(cl_vars.platforms[0], CL_DEVICE_TYPE_GPU, cl_vars.num_devices, cl_vars.device_list, nullptr);
    cl_vars.context = clCreateContext(nullptr, cl_vars.num_devices, cl_vars.device_list, nullptr,
                                      nullptr, &cl_vars.clStatus);
    cl_vars.command_queue = clCreateCommandQueue(cl_vars.context, cl_vars.device_list[0], 0, &cl_vars.clStatus);

    if(cl_vars.kernel_string == nullptr) {
        cl_vars.kernel_string = read_kernel_from_file(kernel_source);
    }
    const char* cKernel_string = cl_vars.kernel_string;

    cl_vars.program = clCreateProgramWithSource(cl_vars.context, 1, &cKernel_string, nullptr, &cl_vars.clStatus);
}

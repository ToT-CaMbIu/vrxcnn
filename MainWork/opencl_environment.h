#pragma once

#include "connected_libs.h"
#include "utils.h"

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
     }                                                                 \
     _ret;                                                             \
   })

//OpenCl
struct CLVars {
    size_t work_per_thread = 10;

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

void cl_clean(CLVars& cl_vars);

void opencl_environment_definition_vortex(CLVars& cl_vars,
                                          const char* binary_source);

void opencl_environment_definition(CLVars& cl_vars,
                                   const char* kernel_source);


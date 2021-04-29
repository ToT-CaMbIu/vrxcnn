#include "connected_libs.h"
#include "opencl_environment.h"
#include "hosts/hosts_convolution.h"
#include "hosts/hosts_matrix_mul.h"
#include "hosts/hosts_max_pool.h"
#include "hosts/hosts_test3d.h"

#ifdef h5_debug
    #include "h5_helper.h"
#endif

int main (int argc, char **argv) {

#ifdef h5_debug
    h5_test();
#else
    srand(time(nullptr));

    CLVars cl_vars_convolution;
    CLVars cl_vars_max_pool;
    CLVars cl_vars_max_pool_3d;
    CLVars cl_vars_matrix_mul;
    CLVars cl_vars_test3d;

    /*for(int i = 0; i < 10; ++i) {
        make_max_pool_3d(cl_vars_max_pool_3d);
        cl_clean(cl_vars_max_pool_3d);
    }
    free(cl_vars_max_pool_3d.kernel_string);*/

    for(int i = 0; i < 10; ++i) {
        make_convolution(cl_vars_convolution);
        cl_clean(cl_vars_convolution);
        make_max_pool(cl_vars_max_pool);
        cl_clean(cl_vars_max_pool);
        make_matrix_mul(cl_vars_matrix_mul);
        cl_clean(cl_vars_matrix_mul);
    }

    free(cl_vars_convolution.kernel_string);
    free(cl_vars_max_pool.kernel_string);
    free(cl_vars_matrix_mul.kernel_string);
#endif

    return 0;
}

#include "connected_libs.h"
#include "opencl_environment.h"
#include "hosts/hosts_convolution.h"
#include "hosts/hosts_matrix_mul.h"
#include "hosts/hosts_max_pool.h"

#ifdef h5_debug
    #include "h5_helper.h"
#endif

int main (int argc, char **argv) {

#ifdef h5_debug
    h5_test();
#else
    srand(time(nullptr));

    CLVars cl_vars;

    for(int i = 0; i < 100; ++i) {
        make_max_pool(cl_vars);
        cl_clean(cl_vars);
    }

    free(cl_vars.kernel_string);
#endif

    return 0;
}

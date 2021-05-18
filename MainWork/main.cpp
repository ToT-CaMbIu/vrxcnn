#include "connected_libs.h"
#include "opencl_environment.h"
#include "hosts/hosts_convolution.h"
#include "hosts/hosts_matrix_mul.h"
#include "hosts/hosts_max_pool.h"
#include "hosts/hosts_test3d.h"
#include "models.h"

#ifdef h5_debug
    #include "h5_helper.h"
#endif

int main (int argc, char **argv) {

#ifdef h5_debug
    std::string input_file = "./bmp/train_47.bmp";
    int x, y;
    std::vector<float> input_image = read_image(input_file.data(), x, y);

    Image<float> image(x, std::vector<float>(y));

    for(int i = 0; i < x; ++i) {
        for(int j = 0; j < y; ++j) {
            image[i][j] = input_image[i * y + j];
        }
    }

    Tensor<float> tensor;
    tensor.push_back(image);

    auto opt = h5_test_convolution();

    if(!opt.has_value()) {
        std::cerr << "h5 read error!" << std::endl;
        return -1;
    }

    Tensor<float> filters = opt.value();

    CLVars cl_vars_convolution_3d;
    auto conv1 = make_convolution_3d(cl_vars_convolution_3d, tensor, filters);
    cl_clean(cl_vars_convolution_3d);
    free(cl_vars_convolution_3d.kernel_string);

    CLVars cl_vars_max_pool_3d;
    make_max_pool_3d(cl_vars_max_pool_3d, conv1);
    cl_clean(cl_vars_max_pool_3d);
    free(cl_vars_max_pool_3d.kernel_string);

#else
    srand(time(nullptr));

    CLVars cl_vars_convolution;
    CLVars cl_vars_max_pool;
    CLVars cl_vars_matrix_mul;
    CLVars cl_vars_max_pool_3d;
    CLVars cl_vars_convolution_3d;
    CLVars cl_vars_test3d;

    cl_vars_max_pool_3d.work_per_thread = 20;

    for(int i = 0; i < 1; ++i) {
        make_max_pool_3d(cl_vars_max_pool_3d);
        //cl_clean(cl_vars_convolution_3d);
    }
    cl_clean(cl_vars_max_pool_3d);
    free(cl_vars_max_pool_3d.kernel_string);

    /*for(int i = 0; i < 10; ++i) {
        make_convolution(cl_vars_convolution);
        cl_clean(cl_vars_convolution);
        make_max_pool(cl_vars_max_pool);
        cl_clean(cl_vars_max_pool);
        make_matrix_mul(cl_vars_matrix_mul);
        cl_clean(cl_vars_matrix_mul);
    }

    free(cl_vars_convolution.kernel_string);
    free(cl_vars_max_pool.kernel_string);
    free(cl_vars_matrix_mul.kernel_string);*/
#endif

    return 0;
}

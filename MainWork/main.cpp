#include "connected_libs.h"
#include "opencl_environment.h"
#include "hosts/hosts_convolution.h"
#include "hosts/hosts_matrix_mul.h"
#include "hosts/hosts_max_pool.h"
#include "models.h"

#ifdef h5_debug
    #include "h5_helper.h"
#endif

int main (int argc, char **argv) {

#ifdef h5_debug

    CLVars cl_vars_convolution_3d;
    CLVars cl_vars_max_pool_3d;
    CLVars cl_vars_matrix_mul;

    std::string input_file = "./bmp/train_932.bmp";
    int x, y;
    std::vector<float> input_image = read_image(input_file.data(), x, y);

    H5Helper<float> h5_helper;

    auto ReLu = [](float val) -> float{
        return fmax(val, 0.0f);
    };

    Tensor<float> tensor(x, y);
    tensor.add_kernel(input_image, x, y);

    std::vector<size_t> dims = {3,3,1,32};
    auto convolution1_optional = h5_helper.h5_convolution_wrapper("../PythonNeuro/mnist_model.h5",
                                                                  "/model_weights//conv2d/conv2d/kernel:0",
                                                                  dims);

    if(!convolution1_optional.has_value()) {
        std::cerr << "h5 read error!" << std::endl;
        return -1;
    }

    Tensor<float> filters_conv1 = std::move(convolution1_optional.value());

    dims = {32};
    auto bias_conv1_optional = h5_helper.read_weights_from_file("../PythonNeuro/mnist_model.h5",
                                                                "/model_weights//conv2d/conv2d/bias:0",
                                                                dims);

    if(!bias_conv1_optional.has_value())
    {
        std::cerr << "h5 read error!" << std::endl;
        return -1;
    }

    std::vector<float> bias_conv1 = std::move(bias_conv1_optional.value());
    auto conv1 = make_convolution_3d(cl_vars_convolution_3d, tensor, filters_conv1, bias_conv1, ReLu);
    auto pool1 = make_max_pool_3d(cl_vars_max_pool_3d, conv1);

    dims = {3,3,32,64};
    auto convolution2_optional = h5_helper.h5_convolution_wrapper("../PythonNeuro/mnist_model.h5",
                                                                  "/model_weights/conv2d_1/conv2d_1/kernel:0",
                                                                  dims);

    if(!convolution2_optional.has_value()) {
        std::cerr << "h5 read error!" << std::endl;
        return -1;
    }

    Tensor<float> filters_conv2 = std::move(convolution2_optional.value());

    dims = {64};
    auto bias_conv2_optional = h5_helper.read_weights_from_file("../PythonNeuro/mnist_model.h5",
                                                                "/model_weights/conv2d_1/conv2d_1/bias:0",
                                                                dims);

    if(!bias_conv2_optional.has_value())
    {
        std::cerr << "h5 read error!" << std::endl;
        return -1;
    }

    std::vector<float> bias_conv2 = std::move(bias_conv2_optional.value());
    auto conv2 = make_convolution_3d(cl_vars_convolution_3d, pool1, filters_conv2, bias_conv2, ReLu);
    auto pool2 = make_max_pool_3d(cl_vars_max_pool_3d, conv2);

    dims = {2304, 10};
    auto filters_dense_optional = h5_helper.h5_dense_wrapper("../PythonNeuro/mnist_model.h5",
                                                             "/model_weights/dense/dense/kernel:0",
                                                             dims);

    if(!filters_dense_optional.has_value())
    {
        std::cerr << "h5 read error!" << std::endl;
        return -1;
    }

    dims = {10};
    auto bias_dense_optional = h5_helper.read_weights_from_file("../PythonNeuro/mnist_model.h5",
                                                                "/model_weights/dense/dense/bias:0",
                                                                dims);

    if(!bias_dense_optional.has_value())
    {
        std::cerr << "h5 read error!" << std::endl;
        return -1;
    }

    std::vector<std::vector<float>> tensor_flatten;

    tensor_flatten.push_back(pool2.to_flatten());

    std::vector<float> result = make_matrix_mul(cl_vars_matrix_mul, tensor_flatten,
                                                filters_dense_optional.value());

    for(int i = 0; i < bias_dense_optional.value().size(); ++i) {
        result[i] += bias_dense_optional.value()[i];
    }

    for(auto& val : result) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    softmax(result);

    for(auto& val : result) {
        std::cout << val << " ";
    }

    cl_clean(cl_vars_convolution_3d);
    cl_clean(cl_vars_max_pool_3d);
    cl_clean(cl_vars_matrix_mul);
    free(cl_vars_max_pool_3d.kernel_string);
    free(cl_vars_convolution_3d.kernel_string);
    free(cl_vars_matrix_mul.kernel_string);

#else

    CLVars cl_vars_convolution_3d;
    CLVars cl_vars_max_pool_3d;
    CLVars cl_vars_matrix_mul;

    std::string input_file = "./bmp/train_932.bmp";
    int x, y;
    std::vector<float> input_image = read_image(input_file.data(), x, y);
    Tensor<float> tensor(x, y);
    tensor.add_kernel(input_image, x, y);

    auto ReLu = [](float val) -> float{
        return fmax(val, 0.0f);
    };

    size_t sz_conv = 3 * 3 * 32;
    size_t sz_bias = 32;

    std::vector<float> a(sz_conv);
    std::vector<float> b(sz_bias);

    for(auto& val : a) {
        val = 0.845 * (rand() % 10);
    }

    for(auto& val : b) {
        val = 0.845 * (rand() % 10);
    }

    Tensor filters_conv(32, 3, 3, a);

    auto conv1 = make_convolution_3d(cl_vars_convolution_3d, tensor, filters_conv, b, ReLu);
    auto pool1 = make_max_pool_3d(cl_vars_max_pool_3d, conv1);


#endif

    return 0;
}

#include "connected_libs.h"
#include "h5_helper.h"

bool h5_test() {
    std::vector<size_t> dims = {3,3,1,32};
    std::string path = "../PythonNeuro/mnist_model.h5";
    std::string layer = "/model_weights/conv2d/conv2d/kernel:0";

    auto t = read_weights_from_file(path, layer, dims);

    if(t.has_value()) {
        auto weights = get_weight_from_flatten_convolution(t.value(), std::make_pair(3,3));

        if(!weights.has_value()) {
            return false;
        }

        for(const auto& kernel : weights.value()) {
            for(int i = 0; i < kernel.size(); ++i) {
                for(int j = 0; j < kernel[i].size(); ++j) {
                    std::cout << kernel[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        return true;
    }

    return false;
}

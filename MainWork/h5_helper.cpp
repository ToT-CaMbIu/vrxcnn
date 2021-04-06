#include "h5_helper.h"

void h5_test() {
    std::string path = "../PythonNeuro/mnist_model.h5";
    std::string layer = "/model_weights/conv2d/conv2d/kernel:0";
    std::vector<float> weights(3 * 3 * 1 * 32);
    std::vector<int> dims(4);

    read_weights_from_file(path, layer, weights, dims.size());

    for (int i = 0; i < weights.size(); ++i) {
        std::cout << weights[i] << std::endl;
    }
}
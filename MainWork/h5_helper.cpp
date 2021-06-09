#include "h5_helper.h"

/*bool h5_bias() {
    std::vector<size_t> dims = {32};
    std::string path = "../PythonNeuro/mnist_model.h5";
    std::string layer = "/model_weights/conv2d/conv2d/bias:0";

    auto t = read_weights_from_file(path, layer, dims);

    if(t.has_value()) {

        std::cout << t.value().size() << std::endl;
        for(const auto& weight : t.value()) {
            std::cout << weight << " ";
        }

        return true;
    }

    return false;
}*/

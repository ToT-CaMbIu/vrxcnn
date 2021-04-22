#pragma once
#include "hdf5.h"
#include <H5Cpp.h>
#include <vector>
#include <iostream>
#include <string>
#include <optional>

bool h5_test();

template<typename T, typename U>
std::optional<std::vector<std::vector<std::vector<T>>>> get_weight_from_flatten_convolution(
                                            std::vector<T>& weights,
                                            std::pair<U,U> kernel_dims) {

    if(weights.size() % (kernel_dims.first * kernel_dims.second) != 0) {
        return std::nullopt;
    }

    size_t count_of_kernels = weights.size() / (kernel_dims.first * kernel_dims.second);

    std::vector<std::vector<std::vector<T>>> weights_formatted;

    std::vector<std::vector<T>> current_kernel(kernel_dims.first, std::vector<T>(kernel_dims.second));

    for(size_t iter = 0; iter < count_of_kernels; ++iter) {
        int iter_align = iter;

        for(size_t x = 0; x < kernel_dims.first; ++x) {
            for(size_t y = 0; y < kernel_dims.second; ++y) {
                current_kernel[x][y] = weights[iter_align];
                iter_align += count_of_kernels;
            }
        }

        weights_formatted.push_back(current_kernel);
    }

    return weights_formatted;
}

template<typename T>
std::optional<std::vector<float>> read_weights_from_file(const std::string& path,
                                                         const std::string& layer,
                                                         const std::vector<T>& dimensions) {

    const int numDims = dimensions.size();

    if(numDims <= 0) {
        return std::nullopt;
    }

    try {
        H5::H5File file(path.c_str(), H5F_ACC_RDONLY);
        H5::DataSet dataSet = file.openDataSet(layer);
        H5::DataSpace dataSpace = dataSet.getSpace();

        hsize_t dims[numDims];
        dataSpace.getSimpleExtentDims(dims, nullptr);

        H5::DataSpace memSpace(numDims, dims);

        size_t sz = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>());
        std::vector<float> weights(sz);

        dataSet.read(weights.data(), H5::PredType::NATIVE_FLOAT, memSpace, dataSpace);

        return weights;
    }
    catch (...) {
        return std::nullopt;
    }

    return std::nullopt;
}
#pragma once
#include "hdf5.h"
#include <H5Cpp.h>
#include <vector>
#include <iostream>
#include <string>
#include <optional>

template<typename T>
std::optional<std::vector<float>> read_weights_from_file(const std::string& path,
                                                         const std::string& layer,
                                                         std::vector<T>& dimensions) {

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
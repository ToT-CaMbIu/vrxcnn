#pragma once
#include "hdf5.h"
#include <H5Cpp.h>
#include <vector>
#include <iostream>
#include <string>

#define h5_debug

template<typename T>
void read_weights_from_file(const std::string& path,
                            const std::string& layer,
                            std::vector<T>& weights,
                            const size_t numDims) {
    H5::H5File file(path.c_str(), H5F_ACC_RDONLY);
    H5::DataSet dataSet = file.openDataSet(layer);
    H5::DataSpace dataSpace = dataSet.getSpace();

    hsize_t dims[numDims];
    dataSpace.getSimpleExtentDims(dims, nullptr);

    H5::DataSpace memSpace(4, dims);

    dataSet.read(weights.data(), H5::PredType::NATIVE_FLOAT, memSpace, dataSpace);
}
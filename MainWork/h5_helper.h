#pragma once
#include "connected_libs.h"
#include "models.h"
#include "hdf5.h"
#include "utils.h"
#include <H5Cpp.h>

template<typename T>
class H5Helper
{
private:

    template<typename U>
    std::optional<Tensor<T>> get_weights_from_flatten_convolution(const std::vector<T>& weights,
                                                                  const std::vector<U>& dims) {

        if(weights.size() % (dims[0] * dims[1]) != 0) {
            return std::nullopt;
        }

        size_t count_of_kernels = weights.size() / (dims[0] * dims[1]);

        Tensor<T> weights_formatted(dims[0], dims[1]);
        std::vector<std::vector<T>> current_kernel(dims[0], std::vector<T>(dims[1]));

        for (size_t iter = 0; iter < dims[3]; ++iter) {
            for(size_t row = 0; row < dims[2]; ++row) {
                size_t iter_align = iter + row * dims[3];

                for (size_t x = 0; x < dims[0]; ++x) {
                    for (size_t y = 0; y < dims[1]; ++y) {
                        current_kernel[x][y] = weights[iter_align];
                        iter_align += count_of_kernels;
                    }
                }

                weights_formatted.add_kernel(current_kernel);
            }
        }

        //print_tensor(weights_formatted.get_data(), dims[0], dims[1], weights_formatted.get_z());

        return weights_formatted;
    }

//    template<typename U>
//    std::optional<Tensor<T>> get_weights_from_flatten_convolution(const std::vector<T>& weights,
//                                                                  const std::pair<U,U> kernel_dims) {
//
//        if(weights.size() % (kernel_dims.first * kernel_dims.second) != 0) {
//            return std::nullopt;
//        }
//
//        size_t count_of_kernels = weights.size() / (kernel_dims.first * kernel_dims.second);
//
//        Tensor<T> weights_formatted(kernel_dims.first, kernel_dims.second);
//        std::vector<std::vector<T>> current_kernel(kernel_dims.first, std::vector<T>(kernel_dims.second));
//
//        for(size_t iter = 0; iter < count_of_kernels; ++iter) {
//            int iter_align = iter;
//
//            for(size_t x = 0; x < kernel_dims.first; ++x) {
//                for(size_t y = 0; y < kernel_dims.second; ++y) {
//                    current_kernel[x][y] = weights[iter_align];
//                    iter_align += count_of_kernels;
//                }
//            }
//
//            weights_formatted.add_kernel(current_kernel);
//        }
//
//        return weights_formatted;
//    }

    template<typename U>
    std::optional<std::vector<std::vector<T>>> get_weights_from_flatten_dense(const std::vector<T>& weights,
                                                                              const std::pair<U,U> dims) {

        if(weights.size() != dims.first * dims.second) {
            return std::nullopt;
        }

        std::vector<std::vector<T>> weights_formatted(dims.first, std::vector<T>(dims.second));

        size_t iter_dense = 0;
        for(size_t i = 0; i < dims.first; ++i) {
            for(size_t j = 0; j < dims.second; ++j) {
                weights_formatted[i][j] = weights[iter_dense++];
            }
        }

        return weights_formatted;
    }

public:
    template<typename U1, typename U2>
    std::optional<std::vector<T>> read_weights_from_file(U1&& path, U2&& layer,
                                                         const std::vector<size_t>& dimensions,
                                                         H5::PredType type = H5::PredType::NATIVE_FLOAT) {

        const int numDims = dimensions.size();

        if(numDims <= 0) {
            return std::nullopt;
        }

        try {
            H5::H5File file(std::forward<std::string>(path).c_str(), H5F_ACC_RDONLY);
            H5::DataSet dataSet = file.openDataSet(std::forward<std::string>(layer));
            H5::DataSpace dataSpace = dataSet.getSpace();

            hsize_t dims[numDims];
            dataSpace.getSimpleExtentDims(dims, nullptr);

            H5::DataSpace memSpace(numDims, dims);

            size_t sz = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<size_t>());
            std::vector<T> weights(sz);

            dataSet.read(weights.data(), type, memSpace, dataSpace);

            return weights;
        }
        catch (...) {
            return std::nullopt;
        }

        return std::nullopt;
    }

    template<typename U1, typename U2>
    std::optional<Tensor<T>> h5_convolution_wrapper(U1&& path, U2&& layer,
                                                    const std::vector<size_t>& dims) {

        auto t = read_weights_from_file(std::forward<U1>(path),
                                        std::forward<U2>(layer), dims);

        if(t.has_value()) {
            auto weights = get_weights_from_flatten_convolution(t.value(), dims);

            if(!weights.has_value()) {
                return std::nullopt;
            }

            return weights.value();
        }

        return std::nullopt;
    }

    template<typename U1, typename U2>
    std::optional<std::vector<std::vector<T>>> h5_dense_wrapper(U1&& path, U2&& layer,
                                                                const std::vector<size_t>& dims) {

        auto t = read_weights_from_file(std::forward<U1>(path),
                                        std::forward<U2>(layer),
                                        dims);

        if(t.has_value()) {
            auto weights = get_weights_from_flatten_dense(t.value(), std::make_pair(dims[0], dims[1]));

            if(!weights.has_value()) {
                return std::nullopt;
            }

            return weights.value();
        }

        return std::nullopt;
    }
};
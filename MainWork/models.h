#pragma once

#include "connected_libs.h"
#include "opencl_environment.h"
#include "hosts/hosts_convolution.h"
#include "hosts/hosts_matrix_mul.h"
#include "hosts/hosts_max_pool.h"
#include "hosts/hosts_test3d.h"

template<typename T>
class ConvModel
{
    using Image_type = std::vector<std::vector<T>>;
    using Tensor_type = std::vector<std::vector<std::vector<T>>>;

private:
    Tensor_type tensor;
public:

    ConvModel(const Image_type& image)
    {
        this->image = image;
    }

    ConvModel(Image_type&& image)
    {
        this->image = std::move(image);
    }

    ConvModel(const Tensor_type& tensor)
    {
        this->tensor = tensor;
    }

    ConvModel(Tensor_type&& tensor)
    {
        this->tensor = std::move(tensor);
    }

    ConvModel(const ConvModel&) = default;
    ConvModel(ConvModel&&) = default;
    ConvModel& operator=(const ConvModel&) = default;
    ConvModel& operator=(ConvModel&&) = default;
    ~ConvModel() = default;

private:
    ConvModel() = default;
};
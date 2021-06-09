#pragma once

#include "connected_libs.h"

template<typename T>
class Tensor
{
private:
    using Tensor_type = std::vector<std::vector<std::vector<T>>>;
    using Image_type = std::vector<std::vector<T>>;

    size_t z, x, y;
    std::shared_ptr<std::vector<T>> data;
public:
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;
    ~Tensor() = default;

    Tensor(size_t x, size_t y) : x(x), y(y)
    {
        z = 0;
        this->data = std::make_shared<std::vector<T>>(std::vector<T>());
    }

    Tensor(size_t z, size_t x, size_t y, const std::vector<T>& data) : z(z), x(x), y(y)
    {
        if(data.size() != z * x * y) {
            std::cerr << "Tensor constructor: size mismatch, it can lead to undefined behaviour!";
        }
        this->data = std::make_shared<std::vector<T>>(data);
    }

    Tensor(size_t z, size_t x, size_t y, std::vector<T>&& data) : z(z), x(x), y(y)
    {
        if(data.size() != z * x * y) {
            std::cerr << "Tensor constructor: size mismatch, it can lead to undefined behaviour!";
        }
        this->data = std::make_shared<std::vector<T>>(std::move(data));
    }

    size_t get_z() const {
        return z;
    }

    size_t get_x() const {
        return x;
    }

    size_t get_y() const {
        return y;
    }

    bool change_dims(size_t z, size_t x, size_t y) {
        if(1ll * this->z * this->y * this->x != 1ll * z * x * y) {
            return false;
        }
        this->z = z;
        this->x = x;
        this->y = y;

        return true;
    }

    bool add_kernel(const Image_type& kernel) {
        if(kernel.empty() || kernel.size() != x || kernel[0].size() != y) {
            return false;
        }

        for(size_t i = 0; i < x; ++i) {
            for(size_t j = 0; j < y; ++j) {
                this->data->push_back(kernel[i][j]);
            }
        }
        z++;

        return true;
    }

    bool add_kernel(const std::vector<T>& kernel, size_t x, size_t y) {
        if(this->x != x || this->y != y || kernel.size() != x * y) {
            return false;
        }

        std::copy(kernel.begin(), kernel.end(), std::back_inserter(*(this->data)));
        z++;

        return true;
    }

    std::vector<T> get_data() const {
        return *(this->data);
    }

    Image_type get_kernel(size_t ind) const {
        if(ind >= z) {
            throw std::runtime_error("Tensor get_matrix: index is out of bounds!");
        }

        Image_type kernel;
        kernel.resize(x);

        for(size_t i = 0; i < x; ++i) {
            kernel[i].resize(y);
        }

        for(size_t i = 0; i < x; ++i) {
            for(size_t j = 0; j < y; ++j) {
                kernel[i][j] = (*(this->data))[ind * x * y + i * y + j];
            }
        }

        return kernel;
    }

    std::vector<T> to_flatten() const {
        std::vector<T> flatten(z * x * y);

        size_t iter_flatten = 0;
        for(size_t i = 0; i < x; ++i) {
            for(size_t j = 0; j < y; ++j) {
                for(size_t k = 0; k < z; ++k) {
                    flatten[iter_flatten++] = (*(this->data))[k * x * y + i * y + j];
                }
            }
        }

        return flatten;
    }

    Tensor_type get_tensor() const {
        Tensor_type tensor;

        tensor.resize(z);
        for(size_t i = 0; i < z; ++i) {
            tensor[i].resize(x);
            for(size_t j = 0; j < x; ++j) {
                tensor[i][j].resize(y);
            }
        }

        for(size_t i = 0; i < z; ++i) {
            for(size_t j = 0; j < x; ++j) {
                for(size_t k = 0; k < y; ++k) {
                    tensor[i][j][k] = (*(this->data))[i * x * y + j * y + k];
                }
            }
        }

        return tensor;
    }

    std::optional<Tensor<T>> tensor_collapse(size_t skip_count,
                                             const std::vector<T>& bias,
                                             std::function<T(const T& val)> func) const
    {
        if(z % skip_count != 0) {
            return std::nullopt;
        }

        std::vector<T> maps((z / skip_count) * x * y, 0.0f);

        for(size_t i = 0; i < z; ++i) {
            size_t current_i = i / skip_count;
            for(size_t j = 0; j < x; ++j) {
                for(size_t k = 0; k < y; ++k) {
                    maps[current_i * x * y + j * y + k] += (*(this->data))[i * x * y + j * y + k];
                }
             }
        }

        for(size_t i = 0; i < z; ++i) {
            size_t current_i = i / skip_count;
            for(size_t j = 0; j < x; ++j) {
                for(size_t k = 0; k < y; ++k) {
                    maps[current_i * x * y + j * y + k] = func(maps[current_i * x * y + j * y + k]);
                    maps[current_i * x * y + j * y + k] += bias[current_i];
                }
            }
        }

        return Tensor((z / skip_count), x, y, maps);
    }
};

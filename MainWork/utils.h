#pragma once

#include <vector>
#include <iostream>
#include <optional>
#include <math.h>

float* read_image(const char *filename,
                  int* widthOut,
                  int* heightOut);

void store_image(float *imageOut,
                 const char *filename,
                 int cols,
                 const char* refFilename);

char* read_kernel_from_file(const char* kernelPath);

bool read_kernel_binary(const char* filename,
                        uint8_t** data,
                        size_t* size);

bool float_compare(float lhs,
                   float rhs,
                   float eps);

float bin_pow(float num, uint32_t power);

template<typename T>
void print_matrix(const std::vector<T>& matrix,
                  int n,
                  int m) {
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            std::cout << matrix[i * m + j] << (j == m - 1 ? '\n' : ' ');
        }
    }
}

template<typename T>
bool test_convolution_padding(int n, int m,
                              int n1, int m1,
                              const std::vector<T>& A,
                              const std::vector<T>& Filter,
                              const std::vector<T>& C,
                              float eps = 1e-7) {

    bool isPassed = true;
    for (size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            float val = 0;
            int x = i - n1 / 2;

            for(size_t i1 = 0; i1 < n1; ++i1, ++x) {
                int y = j - m1 / 2;
                for(size_t j1 = 0; j1 < m1; ++j1, ++y) {
                    if(x >= 0 && y >= 0 && x < n && y < m) {
                        val += (Filter[i1 * m1 + j1] * A[x * m + y]);
                    }
                }
            }

            isPassed &= float_compare(val, C[i * m + j], eps);
        }
    }

    if(isPassed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return isPassed;
}

template<typename T>
bool test_convolution_valid(int n, int m,
                            int n1, int m1,
                            int n2, int m2,
                            const std::vector<T>& A,
                            const std::vector<T>& Filter,
                            const std::vector<T>& C,
                            float eps = 1e-7) {

    bool isPassed = true;
    for (size_t i = 0; i < n2; ++i) {
        for(size_t j = 0; j < m2; ++j) {
            float val = 0;

            for(size_t i1 = 0; i1 < n1; ++i1) {
                for(size_t j1 = 0; j1 < m1; ++j1) {
                    val += (Filter[i1 * m1 + j1] * A[(i + i1) * m + (j + j1)]);
                }
            }

            isPassed &= float_compare(val, C[i * m2 + j], eps);
        }
    }

    if(isPassed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return isPassed;
}

template<typename T>
bool test_max_pool(int n, int m,
                   int n1, int m1,
                   const std::vector<T>& A,
                   const std::vector<T>& C,
                   float eps = 1e-7) {

    bool isPassed = true;
    for (size_t i = 0; i < n1; ++i) {
        for(size_t j = 0; j < m1; ++j) {
            float a1 = -1e9,a2 = -1e9,a3 = -1e9,a4 = -1e9;
            if(i * 2 < n && j * 2 < m) {
                a1 = A[i * 2 * m + j * 2];
            }
            if(i * 2 < n && j * 2 + 1 < m) {
                a2 = A[i * 2 * m + j * 2 + 1];
            }
            if((i * 2 + 1) < n && j * 2 < m) {
                a3 = A[(i * 2 + 1) * m + j * 2];
            }
            if((i * 2 + 1) < n && (j * 2 + 1) < m) {
                a4 = A[(i * 2 + 1) * m + (j * 2 + 1)];
            }

            a1 = std::max(a1, a2);
            a3 = std::max(a3, a4);
            a1 = std::max(a1, a3);

            isPassed &= float_compare(C[i * m1 + j], a1, eps);
        }
    }

    if(isPassed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return isPassed;
}

template<typename T>
bool test_matrix_mul(int n, int m,
                     int k,
                     const std::vector<T>& A,
                     const std::vector<T>& B,
                     const std::vector<T>& C,
                     float eps = 1e-7) {
    bool isPassed = true;
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < m; ++j) {
            float val = 0.0f;
            for(size_t p = 0; p < k; ++p) {
                val += A[i * k + p] * B[p * m + j];
            }
            isPassed &= float_compare(val, C[i * m + j], eps);
        }
    }

    if(isPassed) {
        std::cout << "Passed!" << std::endl;
    }
    else {
        std::cout << "Failed!" << std::endl;
    }

    return isPassed;
}

template<typename T>
std::optional<std::vector<T>> matrix_expand(const std::vector<T> & arr,
                              int n, int m,
                              int n1, int m1) {

    if(n > n1 || m > m1) {
        return std::nullopt;
    }

    std::vector<T> fin(n1 * m1, 0);
    size_t iter = 0;
    for(int i = 0; i < n1; ++i) {
        for(int j = 0; j < m1; ++j) {
            if(i < n && j < m) {
                fin[iter] = arr[i * m + j];
            }
            iter++;
        }
    }

    return fin;
}


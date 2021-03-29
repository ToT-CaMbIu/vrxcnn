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

void print_matrix(float *matrix,
                  int n, int m);

bool test_convolution(int n, int m, int n1, int m1,
                      float *A,
                      float *Filter,
                      float *C,
                      float eps = 1e-7);

bool test_max_pool(int nc, int mc, int n1, int m1,
                   float *A,
                   float *C,
                   float eps = 1e-7);

bool test_matrix_mul(int n, int m, int k,
                     float *A,
                     float *B,
                     float *C,
                     float eps = 1e-7);
float* read_image(const char *filename, int* widthOut, int* heightOut);

void store_image(float *imageOut, const char *filename, int rows, int cols,
                 const char* refFilename);

char* read_kernel_from_file(char* kernelPath);
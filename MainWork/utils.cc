#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

void store_image(float *imageOut,
                 const char *filename,
                 int cols,
                 const char* refFilename) {

    FILE *ifp, *ofp;
    unsigned char tmp;
    int offset;
    unsigned char *buffer;
    int i, j;

    int bytes;

    int height, width;

    ifp = fopen(refFilename, "rb");
    if(ifp == NULL) {
        perror(filename);
        exit(-1);
    }

    fseek(ifp, 10, SEEK_SET);
    fread(&offset, 4, 1, ifp);

    fseek(ifp, 18, SEEK_SET);
    fread(&width, 4, 1, ifp);
    fread(&height, 4, 1, ifp);

    fseek(ifp, 0, SEEK_SET);

    buffer = (unsigned char *)malloc(offset);
    if(buffer == NULL) {
        perror("malloc");
        exit(-1);
    }

    fread(buffer, 1, offset, ifp);

    printf("Writing output image to %s\n", filename);
    ofp = fopen(filename, "wb");
    if(ofp == NULL) {
        perror("opening output file");
        exit(-1);
    }
    bytes = fwrite(buffer, 1, offset, ofp);
    if(bytes != offset) {
        printf("error writing header!\n");
        exit(-1);
    }

    int mod = width % 4;
    if(mod != 0) {
        mod = 4 - mod;
    }

    for(i = height-1; i >= 0; i--) {
        for(j = 0; j < width; j++) {
            tmp = (unsigned char)imageOut[i*cols+j];
            fwrite(&tmp, sizeof(char), 1, ofp);
        }

        for(j = 0; j < mod; j++) {
            fwrite(&tmp, sizeof(char), 1, ofp);
        }
    }

    fclose(ofp);
    fclose(ifp);

    free(buffer);
}

float* read_image(const char *filename,
                  int* widthOut,
                  int* heightOut) {

    char* imageData;

    int height, width;
    char tmp;
    int offset;
    int i, j;

    printf("Reading input image from %s\n", filename);
    FILE *fp = fopen(filename, "rb");
    if(fp == NULL) {
        perror(filename);
        exit(-1);
    }

    fseek(fp, 10, SEEK_SET);
    fread(&offset, 4, 1, fp);

    fseek(fp, 18, SEEK_SET);
    fread(&width, 4, 1, fp);
    fread(&height, 4, 1, fp);

    printf("width = %d\n", width);
    printf("height = %d\n", height);

    *widthOut = width;
    *heightOut = height;

    imageData = (char*)malloc(width*height);
    if(imageData == NULL) {
        perror("malloc");
        exit(-1);
    }

    fseek(fp, offset, SEEK_SET);
    fflush(NULL);

    int mod = width % 4;
    if(mod != 0) {
        mod = 4 - mod;
    }

    for(i = 0; i < height; i++) {


        for(j = 0; j < width; j++) {
            fread(&tmp, sizeof(char), 1, fp);
            imageData[i*width + j] = tmp;
        }

        for(j = 0; j < mod; j++) {
            fread(&tmp, sizeof(char), 1, fp);
        }
    }

    int flipRow;
    for(i = 0; i < height/2; i++) {
        flipRow = height - (i+1);
        for(j = 0; j < width; j++) {
            tmp = imageData[i*width+j];
            imageData[i*width+j] = imageData[flipRow*width+j];
            imageData[flipRow*width+j] = tmp;
        }
    }

    fclose(fp);

    float* floatImage = NULL;
    floatImage = (float*)malloc(sizeof(float)*width*height);
    if(floatImage == NULL) {
        perror("malloc");
        exit(-1);
    }

    for(i = 0; i < height; i++) {
        for(j = 0; j < width; j++) {
            floatImage[i*width+j] = (float)imageData[i*width+j];
        }
    }

    free(imageData);
    return floatImage;
}

char* read_kernel_from_file(char* kernelPath) {

    FILE *fp;
    char *source;
    long int size;

    printf("Program file is: %s\n", kernelPath);

    fp = fopen(kernelPath, "rb");
    if(!fp) {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    bool status = fseek(fp, 0, SEEK_END);
    if(status != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if(size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char *)malloc(size + 1);

    for (int i = 0; i < size+1; i++) {
        source[i]='\0';
    }

    if(source == NULL) {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }

    fread(source, 1, size, fp);
    source[size] = '\0';

    return source;
}

bool float_compare(float lhs,
                   float rhs,
                   float eps) {
    return fabs(lhs - rhs) <= eps;
}

void print_matrix(float *matrix,
                  int n,
                  int m) {
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            printf("%f ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

bool test_convolution(int n, int m, int n1, int m1,
                      float *A,
                      float *Filter,
                      float *C,
                      float eps) {

    bool isPassed = true;
    for (int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            //printf("%f ", C[i * m + j]);

            float val = 0;
            int x = i - n1 / 2;

            for(int i1 = 0; i1 < n1; ++i1, ++x) {
                int y = j - m1 / 2;
                for(int j1 = 0; j1 < m1; ++j1, ++y) {
                    if(x >= 0 && y >= 0 && x < n && y < m) {
                        val += (Filter[i1 * m1 + j1] * A[x * m + y]);
                    }
                }
            }

            isPassed &= float_compare(val, C[i * m + j], eps);
        }
        //printf("\n");
    }

    if(isPassed) {
        printf("Passed!\n");
    }
    else {
        printf("Failed!\n");
    }

    return isPassed;
}

bool test_max_pool(int nc, int mc, int n1, int m1,
                   float *A,
                   float *C,
                   float eps) {

    bool isPassed = true;
    for (int i = 0; i < n1; ++i) {
        for(int j = 0; j < m1; ++j) {
            float a1 = -1e9,a2 = -1e9,a3 = -1e9,a4 = -1e9;
            if(i * 2 * mc + j * 2 < mc * nc) {
                a1 = A[i * 2 * mc + j * 2];
            }
            if(i * 2 * mc + j * 2 + 1 < mc * nc) {
                a2 = A[i * 2 * mc + j * 2 + 1];
            }
            if((i * 2 + 1) * mc + j * 2 < mc * nc) {
                a3 = A[(i * 2 + 1) * mc + j * 2];
            }
            if((i * 2 + 1) * mc + (j * 2 + 1) < mc * nc) {
                a4 = A[(i * 2 + 1) * mc + (j * 2 + 1)];
            }

            a1 = fmax(a1, a2);
            a3 = fmax(a3, a4);
            a1 = fmax(a1, a3);

            //printf("%f ", C[i * m1 + j]);
            isPassed &= float_compare(C[i * m1 + j], a1, eps);
        }
        //printf("\n");
    }

    if(isPassed) {
        printf("Passed!\n");
    }
    else {
        printf("Failed!\n");
    }

    return isPassed;
}

bool test_matrix_mul(int n, int m, int k,
                     float *A,
                     float *B,
                     float *C,
                     float eps) {
    bool isPassed = true;
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            float val = 0.0f;
            for(int p = 0; p < k; ++p) {
                val += A[i * k + p] * B[p * m + j];
            }
            printf("%f ", val);
            isPassed &= float_compare(val, C[i * m + j], eps);
        }
        printf("\n");
    }

    if(isPassed) {
        printf("Passed!\n");
    }
    else {
        printf("Failed!\n");
    }

    return isPassed;
}
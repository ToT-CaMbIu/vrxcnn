#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

//C
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

char* read_kernel_from_file(const char* kernelPath) {

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

bool read_kernel_binary(const char* filename,
                        uint8_t** data,
                        size_t* size) {
    if (nullptr == filename || nullptr == data || 0 == size) {
        return false;
    }

    FILE* fp = fopen(filename, "r");
    if (NULL == fp) {
        fprintf(stderr, "Failed to load kernel.");
        return -1;
    }
    fseek(fp , 0 , SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);

    *data = (uint8_t*)malloc(fsize);
    *size = fread(*data, 1, fsize, fp);

    fclose(fp);

    return true;
}

bool float_compare(float lhs,
                   float rhs,
                   float eps) {
    return fabs(lhs - rhs) <= eps;
}

float bin_pow(float num, uint32_t power) {
    if(power <= 0) {
        return 1;
    }
    if(power & 1) {
        return num * bin_pow(num, power - 1);
    }
    float result = bin_pow(num, power >> 1);
    return result * result;
}
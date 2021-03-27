#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <string>
#include <chrono>
#include <time.h>

#include "hdf5.h"
#include <H5Cpp.h>
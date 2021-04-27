#include "activation_funtions.h"

__kernel void test3d(int x,
                     int y,
                     int z,
                     const __global float* A,
                     __global float* C) {

    const int localX = get_local_id(1);
    const int localY = get_local_id(2);
    const int globalZ = get_global_id(0);
    const int globalX = get_global_id(1);
    const int globalY = get_global_id(2);

    __local float val[4];

    if(globalZ < z && globalX < x && globalY < y) {
        val[localX * 2 + localY] = A[globalZ * x * y + globalX * y + globalY];
    }
    else {
        val[localX * 2 + localY] = -1e9;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    C[globalZ * x * y + globalX * y + globalY] = val[localX * 2 + localY];
}


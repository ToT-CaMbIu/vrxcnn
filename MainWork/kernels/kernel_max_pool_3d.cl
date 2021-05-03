#include "activation_funtions.h"

__kernel void matrix_max_pool_transformation_3d(int n,
                                                int m,
                                                int n1,
                                                int m1,
                                                const __global float* A,
                                                __global float* C) {

    const int localX = get_local_id(1);
    const int localY = get_local_id(2);
    const int globalZ = get_global_id(0);
    const int globalX = get_global_id(1);
    const int globalY = get_global_id(2);

    const int tx = globalX >> 1;
    const int ty = globalY >> 1;

    __local float val[4];

    if(globalX < n1 && globalY < m1) {
        val[(localX << 1) + localY] = A[globalZ * n1 * m1 + globalX * m1 + globalY];
    }
    else {
        val[(localX << 1) + localY] = -1e9;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local float mx;
    mx = val[0];

    for(int i = 1; i < 4; ++i) {
        if(val[i] > mx) {
            mx = val[i];
        }
    }

    C[globalZ * (n >> 1) * (m >> 1) + tx * (m >> 1) + ty] = mx;
}
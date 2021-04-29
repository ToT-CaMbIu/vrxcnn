#include "activation_funtions.h"

__kernel void matrix_max_pool_transformation_3d(int n,
                                                int m,
                                                int n1,
                                                int m1,
                                                int z,
                                                const __global float* A,
                                                __global float* C) {

    const int localX = get_local_id(1);
    const int localY = get_local_id(2);
    const int globalZ = get_global_id(0);
    const int globalX = get_global_id(1);
    const int globalY = get_global_id(2);

    const int tx = globalX / 2;
    const int ty = globalY / 2;
    const int align_x = n / 2;
    const int align_y = m / 2;

    __local float val[4];

    if(globalZ < z && globalX < n1 && globalY < m1) {
        val[localX * 2 + localY] = A[globalZ * n1 * m1 + globalX * m1 + globalY];
    }
    else {
        val[localX * 2 + localY] = -1e9;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local float mx;
    mx = val[0];

    for(int i = 1; i < 4; ++i) {
        if(val[i] > mx) {
            mx = val[i];
        }
    }

    C[globalZ * align_x * align_y + tx * align_y + ty] = mx;
}
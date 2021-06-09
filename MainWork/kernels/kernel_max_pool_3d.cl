#include "activation_funtions.h"

__kernel void matrix_max_pool_transformation_3d(int z,
                                                int n,
                                                int m,
                                                int n1,
                                                int m1,
                                                int thread_skip,
                                                const __global float* A,
                                                __global float* C) {

    const int localX = get_local_id(1);
    const int localY = get_local_id(2);
    const int globalZ = get_global_id(0);
    const int globalX = get_global_id(1);
    const int globalY = get_global_id(2);

    const int tx = globalX >> 1;
    const int ty = globalY >> 1;
    const int nx = n >> 1;
    const int my = m >> 1;

    __local float val[4];

    for(int i = 0; globalZ + i * thread_skip < z; ++i) {
        int current_z = globalZ + i * thread_skip;

        if (globalX < n1 && globalY < m1) {
            val[(localX << 1) + localY] = A[current_z * n1 * m1 + globalX * m1 + globalY];
        } else {
            val[(localX << 1) + localY] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local float mx;
        mx = val[0];

        for (int i = 1; i < 4; ++i) {
            mx = max(val[i], mx);
        }

        C[current_z * nx * my + tx * my + ty] = mx;
    }
}
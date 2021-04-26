/*__kernel void matrix_mul(const int M,
                         const int N,
                         const int K,
                         const int TS_x,
                         const int TS_y,
                         const __global float* A,
                         const __global float* B,
                         __global float* C,
                         __local float* A_tile,
                         __local float* B_tile) {
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);
    if(globalRow >= M || globalCol >= N) {
        return;
    }
    float acc = 0.0f;
    for (int k=0; k<K; ++k) {
        acc += A[globalRow * K + k] * B[N * k + globalCol];
    }
    C[globalRow*N + globalCol] = acc;
}*/

#include "activation_funtions.h"

__kernel void matrix_mul(const int n,
                         const int m,
                         const int k,
                         const int k1,
                         const int ts,
                         const __global float* A,
                         const __global float* B,
                         __global float* C,
                         __local float* A_tile,
                         __local float* B_tile) {

    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    const int blockRow = get_group_id(0);
    const int blockCol = get_group_id(1);

    int row = blockRow * ts + localRow;
    int col = blockCol * ts + localCol;

    float acc = 0.0f;

    for(int i = 0; i < k / ts; ++i) {

        if(row < n && (i * ts + localCol) < k1) {
            A_tile[localRow * ts + localCol] = A[row * k1 + (i * ts + localCol)];
        }
        else {
            A_tile[localRow * ts + localCol] = 0.0;
        }

        if((i * ts + localRow) < k1 && col < m) {
            B_tile[localRow * ts + localCol] = B[(i * ts + localRow) * m + col];
        }
        else {
            B_tile[localRow * ts + localCol] = 0.0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < ts; ++j) {
            acc += A_tile[localRow * ts + j] * B_tile[j * ts + localCol];
            barrier(CLK_LOCAL_MEM_FENCE);
        }

    }

    if(row < n && col < m) {
        C[row * m + col] = acc;
    }
}
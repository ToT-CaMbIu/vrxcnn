#include "activation_funtions.h"

int toeplitz_changer(const int row,
                     const int col,
                     const int m,
                     const int m1,
                     const int m2) {

    int row_block = col / m2;
    int col_block = col % m2;
    int block_upper_left = row_block * m + col_block;
    return block_upper_left + (row / m1) * m + row % m1;
}

__kernel void matrix_convolutional_transformation(int nA, int mA,
                                                  int block_x,
                                                  int nF, int mF,
                                                  int nM, int mM, int k_aligned,
                                                  int ts,
                                                  int blocks_in_row,
                                                  int weights_per_matrix,
                                                  const __global float* A,
                                                  const __global float* Filter,
                                                  __global float* C,
                                                  __local float* Filter_tile,
                                                  __local float* A_tile) {

    const int localRow = get_local_id(1);
    const int localCol = get_local_id(2);
    const int blockRow = get_group_id(1);
    const int blockCol = get_group_id(2);

    const int globalZ = get_global_id(0);

    int row = blockRow * ts + localRow;
    int col = blockCol * ts + localCol;

    float acc = 0.0f;

    for(int i = 0; i < k_aligned / ts; ++i) {

        if(row < nF && (i * ts + localCol) < mF) {
            Filter_tile[localRow * ts + localCol] = Filter[globalZ * nF * mF + row * mF + (i * ts + localCol)];
        }
        else {
            Filter_tile[localRow * ts + localCol] = 0.0;
        }

        if((i * ts + localRow) < nM && col < mM) {
            int a_index = toeplitz_changer(i * ts + localRow, col, mA, block_x, blocks_in_row);
            A_tile[localRow * ts + localCol] = A[(globalZ / weights_per_matrix) * nA * mA + a_index];
        }
        else {
            A_tile[localRow * ts + localCol] = 0.0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int j = 0; j < ts; ++j) {
            acc += Filter_tile[localRow * ts + j] * A_tile[j * ts + localCol];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if(row < nF && col < mM) {
        C[globalZ * nF * mM + row * mM + col] = acc;
    }
}

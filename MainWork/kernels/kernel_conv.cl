/*__kernel void matrix_convolutional_transformation(int n, int m, int n1, int m1,
                                                  const __global float* A,
                                                  const __global float* Filter,
                                                  __global float* C) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if(row >= n || col >= m) {
        return;
    }

    float val = 0.0f;
    int row_shifted = row - n1 / 2;

    for(int i = 0; i < n1; ++i, ++row_shifted) {
        int col_shifted = col - m1 / 2;
        for(int j = 0; j < m1; ++j, ++col_shifted) {
            if(row_shifted >= 0 && col_shifted >= 0 && row_shifted < n && col_shifted < m) {
                val += A[row_shifted * m + col_shifted] * Filter[i * m1 + j];
            }
        }
    }

    C[row * m + col] = val;
}*/

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

__kernel void matrix_convolutional_transformation(int mA, int block_x,
                                                  int nF, int mF,
                                                  int nM, int mM, int k_aligned,
                                                  int ts,
                                                  int blocks_in_row,
                                                  const __global float* A,
                                                  const __global float* Filter,
                                                  __global float* C,
                                                  __local float* Filter_tile,
                                                  __local float* A_tile) {

    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    const int blockRow = get_group_id(0);
    const int blockCol = get_group_id(1);

    int row = blockRow * ts + localRow;
    int col = blockCol * ts + localCol;

    float acc = 0.0f;

    for(int i = 0; i < k_aligned / ts; ++i) {

        if(row < nF && (i * ts + localCol) < mF) {
            Filter_tile[localRow * ts + localCol] = Filter[row * mF + (i * ts + localCol)];
        }
        else {
            Filter_tile[localRow * ts + localCol] = 0.0;
        }

        if((i * ts + localRow) < nM && col < mM) {
            int a_index = toeplitz_changer(i * ts + localRow, col, mA, block_x, blocks_in_row);
            A_tile[localRow * ts + localCol] = A[a_index];
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
        C[row * mM + col] = acc;
    }
}

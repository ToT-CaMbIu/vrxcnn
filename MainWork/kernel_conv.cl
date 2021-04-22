__kernel void matrix_convolutional_transformation(int n, int m, int n1, int m1,
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
}

/*__kernel void matrix_convolutional_transformation(int n, int m,
                                                  int n1, int m1,
                                                  int n2, int m2,
                                                  const __global float* A,
                                                  const __global float* Filter,
                                                  __global float* C) {

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

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
}*/

__kernel void vecadd (int n, const __global *A, const __global float *B, __global float *C) {
    int gid = get_global_id(0);
    if(gid < n) {
        C[gid] = A[gid] + B[gid];
    }
}

__kernel void vecmul (int n, const __global float *A, const __global float *B, __global float *C) {
    int gid = get_global_id(0);
    if (gid < n) {
        C[gid] = A[gid] * B[gid];
    }
}

__kernel void matrix_convolutional_transformation(int n, int m, int n1, int m1,
                                                  const __global float *A,
                                                  const __global float *Filter,
                                                  __global float *C) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if(row >= n || col >= m) {
        return;
    }

    float val = 0.0;
    int row_shifted = row - n1 / 2;

    for(int i = 0; i < n1; ++i, ++row_shifted) {
        int col_shifted = col - m1 / 2;
        for(int j = 0; j < m1; ++j, ++col_shifted) {
            if(row_shifted >= 0 && col_shifted >= 0 && row_shifted < n && col_shifted < m) {
                val += A[row_shifted * n + col_shifted] * Filter[i * n1 + j];
            }
        }
    }

    C[row * n + col] = val;
}

__kernel void matrix_max_pool_transformation(int n, int m
                                            const __global float* A,
                                            const __global float* C) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if(row >= n / 2 || col >= m / 2) {
        return;
    }

    float val = 0.0;

    
}
$
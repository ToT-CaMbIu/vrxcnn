__kernel void matrix_max_pool_transformation(int n,
                                             int m,
                                             int n1,
                                             int m1,
                                             const __global float* A,
                                             __global float* C) {

    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    const int tx = globalRow / 2;
    const int ty = globalCol / 2;
    const int align = m / 2;

    __local float val[4];

    if(globalRow < n1 && globalCol < m1) {
        val[localRow * 2 + localCol] = A[globalRow * m1 + globalCol];
    }
    else {
        val[localRow * 2 + localCol] = -1e9;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local float mx;
    mx = val[0];

    for(int i = 1; i < 4; ++i) {
        if(val[i] > mx) {
            mx = val[i];
        }
    }

    C[tx * align + ty] = mx;
}


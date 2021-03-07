__kernel void matrix_max_pool_transformation(int n, int m,
                                             const __global float* A,
                                             __global float* C) {

    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = n * get_group_id(0) + row;
    const int globalCol = m * get_group_id(1) + col;

    if (globalRow >= n || globalCol >= m) {
        return;
    }

    __local float val;

    if(A[globalRow * n + globalCol] > val) {
        val = A[globalRow * n + globalCol];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    C[globalRow * (n / 2) + globalCol / 2] = 5.0;
}$
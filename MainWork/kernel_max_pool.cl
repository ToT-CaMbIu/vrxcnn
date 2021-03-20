__kernel void matrix_max_pool_transformation(int n,
                                             int m,
                                             const __global float* A,
                                             __global float* C) {

    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    if(globalRow >= n || globalCol >= m) {
        return;
    }

    __local float val[4];
    __local int pos[4];

    val[localRow * 2 + localCol] = A[globalRow * m + globalCol];
    pos[localRow * 2 + localCol] = (globalRow * m / 4) + (globalCol / 2);

    barrier(CLK_LOCAL_MEM_FENCE);

    float mx = val[0];

    for(int i = 1; i < 4; ++i) {
        if(val[i] > mx) {
            mx = val[i];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    C[pos[0]] = mx;
}
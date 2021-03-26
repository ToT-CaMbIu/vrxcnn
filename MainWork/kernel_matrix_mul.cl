/*__kernel void matrix_mul(const int M,
                         const int N,
                         const int K,
                         const int TS,
                         const __global float* A,
                         const __global float* B,
                         __global float* C,
                         __local float* A_tile,
                         __local float* B_tile) {

    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    float acc = 0.0f;

    const int numTiles = K / TS;
    for (int i = 0; i < numTiles; ++i) {

        const int tiledRow = TS * i + row;
        const int tiledCol = TS * i + col;
        A_tile[col * TS + row] = A[tiledCol * K + globalRow];
        B_tile[col * TS + row] = B[globalCol * N + tiledRow];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < TS; ++j) {
            acc += A_tile[j * TS + row] * B_tile[col * TS + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[globalRow * N + globalCol] = acc;
}*/

__kernel void matrix_mul(const int M,
                         const int N,
                         const int K,
                         const int TS,
                         const __global float* A,
                         const __global float* B,
                         __global float* C,
                         __local float* A_tile,
                         __local float* B_tile) {

    const int globalRow = get_global_id(0);
    const int globalCol = get_global_id(1);

    float acc = 0.0f;
    for (int k=0; k<K; ++k) {
        acc += A[globalRow * K + k] * B[N * k + globalCol];
    }

    C[globalRow*N + globalCol] = acc;
}
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
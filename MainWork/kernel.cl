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
}$
__kernel void MatrixMultiplication(__global float* a, __global float* b, __global float* c, const unsigned int size) {
    int tid_x = get_global_id(0);
    int tid_y = get_global_id(1);
    int loc_size_x = get_local_size(0);
    int loc_size_y = get_local_size(1);

    //if (tid_x == 0 && tid_y == 0)
       // printf("block size x = %d\nblock size y = %d\n", loc_size_x, loc_size_y);

    float res = 0;
    for (int i = 0; i < size; ++i)
        res += a[size * tid_y + i] * b[size * i + tid_x];

    c[size * tid_y + tid_x] = res;
}
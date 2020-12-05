//__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void MatrixMultiplication(__global float* a, __global float* b, __global float* c, const unsigned int size) {
    int tid_x = get_global_id(0);
    int tid_y = get_global_id(1);

    float res = 0;
    for (int i = 0; i < size; ++i)
        res += a[size * tid_y + i] * b[size * i + tid_x];

    c[size * tid_y + tid_x] = res;
}

__kernel void MatrixMultiplicationBlock(__global float* a, __global float* b, __global float* c, const unsigned int size) {
    int tid_x = get_global_id(0);
    int tid_y = get_global_id(1);
    int loc_size_x = get_local_size(0);
    int loc_size_y = get_local_size(1);
    int loc_id_x = get_local_id(0);
    int loc_id_y = get_local_id(1);
    //int global_group_id_x = get_group_id(0);
    //int global_group_id_y = get_group_id(1);

    __local float A[16 * 16];
    __local float B[16 * 16];
    float res = 0.0;

    for (int p = 0; p < size / loc_size_x; ++p) {
        //A[loc_id_y * loc_size_x + loc_id_x] = a[(global_group_id_y * loc_size_x + loc_id_y) * size + p * loc_size_x + loc_id_x];
        //B[loc_id_y * loc_size_x + loc_id_x] = b[(p * loc_size_y + loc_id_y) * size + global_group_id_x * loc_size_x + loc_id_x];
        A[loc_id_y * loc_size_x + loc_id_x] = a[tid_y * size + p * loc_size_x + loc_id_x];
        B[loc_id_y * loc_size_x + loc_id_x] = b[(p * loc_size_y + loc_id_y) * size + tid_x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < loc_size_x; ++i)
            res += A[loc_size_x * loc_id_y + i] * B[loc_size_y * i + loc_id_x];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[size * tid_y + tid_x] = res;
}

__kernel void MatrixMultiplicationImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, const unsigned int size) {
    int tid_x = get_global_id(0);
    int tid_y = get_global_id(1);
    int2 grid_id_c = (int2)(tid_x, tid_y);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    //float4 a_loc = read_imagef(a, sampler, (int2)(tid_x, tid_y));
    //printf("res - %v4hlf,\t x - %d, y - %d\n", a_loc, tid_x, tid_y);

    float4 res = (float4)(0.0f);
    for (int i = 0; i < size; ++i) {
        float4 a_loc = read_imagef(a, sampler, (int2)(i, tid_y));
        float4 b_loc = read_imagef(b, sampler, (int2)(tid_x, i));
        res += a_loc * b_loc;
    }
    write_imagef(c, grid_id_c, res);
}
__kernel void square(__global int* input, __global int* output, const unsigned int count) {
    int i = get_global_id(0);
    if (i < count)
        output[i] = input[i] * i;
}
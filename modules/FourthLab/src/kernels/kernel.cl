__kernel void JacobiEPS(__global float* A, __global float* b, __global float* x, const unsigned int size, const double eps) {
    int tid_x = get_global_id(0);
    float x_k = 1.0;
    float x_k_1 = 0.0;
    float bi = b[tid_x];
    float aii = A[tid_x * size + tid_x];
    while ((double)(fabs(x_k - x_k_1)) > eps) {
        float sum = 0.0;
        for (size_t i = 0; i < size; ++i) {
            if (i == tid_x)
                continue;
            sum += x_k * A[tid_x * size + i];
        }
        //printf("%f\n", sum);
        x_k = x_k_1;
        x_k_1 = (bi - sum) / aii;
    }
    //printf("i - %d, %f\n", tid_x, x_k_1);
    x[tid_x] = x_k_1;
}

__kernel void JacobiIter(__global float* A, __global float* b, __global float* x, const unsigned int size, const int iter) {
    int tid_x = get_global_id(0);
    float x_k = 1.0;
    float x_k_1 = 0.0;
    float bi = b[tid_x];
    float aii = A[tid_x * size + tid_x];
    for (int j = 0; j < iter; ++j) {
        float sum = 0.0;
        for (size_t i = 0; i < size; ++i) {
            if (i == tid_x)
                continue;
            sum += x_k * A[tid_x * size + i];
        }
        //printf("%f\n", sum);
        x_k = x_k_1;
        x_k_1 = (bi - sum) / aii;
    }
    //printf("%f\n", x_k_1);
    x[tid_x] = x_k_1;
}
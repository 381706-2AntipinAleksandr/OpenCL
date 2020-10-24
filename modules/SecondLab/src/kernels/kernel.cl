__kernel void saxpy(const unsigned int n, const float a, __global float* x, const int incx, __global float* y, const int incy) {
    //int b = get_group_id(0);
    //int t = get_local_id(0);
    int i = get_global_id(0);

    //if (i < 1 + (n-1)*abs(incx))
    y[i * incy] = y[i * incy] + a * x[i * incx];
}
#include <CL/cl.h>
#include <omp.h>
#include <iostream>
#include <string>
#include "../../../init.h"

void SequentialSaxpy(const unsigned int n, const float a, float* x, const int incx, float* y, const int incy) {
    for (int i = 0; i < n; ++i)
        y[i * incy] = y[i * incy] + a * x[i * incx];
}

void ParallelOpenMPSaxpy(const unsigned int n, const float a, float* x, const int incx, float* y, const int incy) {
    int num_tr = omp_get_num_threads();
    #pragma omp parallel num_threads(num_tr)
    {
        #pragma omp for schedule(static, 1)
            for (int i = 0; i < n; ++i)
                y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

int main(int argc, char** argv) {
    OCLInitialization pr(argv[0], 64);
    pr.AddKernel("saxpy");
    // input and output values
    const unsigned int size = 1000;
    float y[size];
    float x[size];
    float a = 2.0;
    for (int i = 0; i < size; ++i) {
        y[i] = i;
        x[i] = size - 1 - i;
    }
    int incx = 1, incy = 1;


    //ParallelOpenMPSaxpy(size, a, x, incx, y, incy);
    pr.AddBuffer<float>(CL_MEM_READ_WRITE, size);
    pr.AddBuffer<float>(CL_MEM_READ_ONLY, size);
    pr.WriteElementsToBuffer(0, size, y);
    pr.WriteElementsToBuffer(1, size, x);
    pr.SetKernelArg(pr.GetKernel(0), 0, &size);
    pr.SetKernelArg(pr.GetKernel(0), 1, &a);
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetBuffer(1));
    pr.SetKernelArg(pr.GetKernel(0), 3, &incx);
    pr.SetKernelArg(pr.GetKernel(0), 4, &pr.GetBuffer(0));
    pr.SetKernelArg(pr.GetKernel(0), 5, &incy);
    pr.ExecuteKernel(pr.GetKernel(0), size);
    pr.ReadElementsFromBuffer(0, size, y);

    for (int i = 0; i < size; ++i) {
        std::cout << y[i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}
#include <CL/cl.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <random>
#include "../../../init.h"

void SequentialMatrixMultiplication(const size_t n, const float* a, const float* b, float* c) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
        {
            c[i * n + j] = 0;
            for (size_t k = 0; k < n; ++k)
                c[i * n + j] += a[i * n + k] * b[k * n + j];
        }
}

void ParallelMatrixMultiplication(const size_t n, const float* a, const float* b, float* c) {
    int num_tr = omp_get_max_threads();
    num_tr = n < num_tr ? n : num_tr;
#pragma omp parallel num_threads(num_tr)
    {
#pragma omp for schedule(static, n / num_tr)
        for (long long i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                c[i * n + j] = 0;
                for (size_t k = 0; k < n; ++k)
                    c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
    }
}

void ParallelOpenCLMatrixMultiplication(const char* path, const size_t group, const unsigned int size,
                                        float* a, float* b, float* c) {
    size_t group_size[] = { group, group };
    size_t size_n[] = { size, size };
    OCLInitialization pr(path, 2, group_size, size_n);
    pr.AddKernel("MatrixMultiplication");
    pr.AddBuffer<float>(CL_MEM_READ_WRITE, size * size);
    pr.AddBuffer<float>(CL_MEM_READ_WRITE, size * size);
    pr.AddBuffer<float>(CL_MEM_WRITE_ONLY, size * size);
    pr.WriteElementsToBuffer(0, size * size, a);
    pr.WriteElementsToBuffer(1, size * size, b);
    pr.SetKernelArg(pr.GetKernel(0), 0, &pr.GetBuffer(0));
    pr.SetKernelArg(pr.GetKernel(0), 1, &pr.GetBuffer(1));
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetBuffer(2));
    pr.SetKernelArg(pr.GetKernel(0), 3, &size);
    double start = omp_get_wtime();
    pr.ExecuteKernel(pr.GetKernel(0));
    double end = omp_get_wtime();
    std::cout << "GPU float time - " << end - start << std::endl;
    pr.ReadElementsFromBuffer(2, size * size, c);
}

bool IsEqual(const float* a, const float* b, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > 0.01) {
            //std::cout << a[i] << "   " << b[i] << std::endl;
            printf("%lf   %lf\n", a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    const unsigned int size = 2048;

    // init float
    float* a_seq = new float[size * size];
    float* b_seq = new float[size * size];
    float* c_seq = new float[size * size];

    float* a_par_cpu = new float[size * size];
    float* b_par_cpu = new float[size * size];
    float* c_par_cpu = new float[size * size];

    float* a_par_gpu = new float[size * size];
    float* b_par_gpu = new float[size * size];
    float* c_par_gpu = new float[size * size];

    float lower_bound = -10.0;
    float upper_bound = 10.0;
    std::cout << "gen start" << std::endl;
    std::uniform_real_distribution<float> unif_f(lower_bound, upper_bound);
    std::default_random_engine re;
    for (int i = 0; i < size * size; ++i) {
        float num_f = unif_f(re);

        a_seq[i] = a_par_cpu[i] = a_par_gpu[i] = num_f;

        b_seq[i] = b_par_cpu[i] = b_par_gpu[i] = static_cast<float>(10.0) - num_f;
    }
    std::cout << "gen end" << std::endl;

    double start = omp_get_wtime();
    SequentialMatrixMultiplication(size, a_seq, b_seq, c_seq);
    double end = omp_get_wtime();
    std::cout << "Sequential float time - " << end - start << std::endl;
    start = omp_get_wtime();
    ParallelMatrixMultiplication(size, a_par_cpu, b_par_cpu, c_par_cpu);
    end = omp_get_wtime();
    std::cout << "OpenMP float time - " << end - start << std::endl;
    ParallelOpenCLMatrixMultiplication(argv[0], 16, size, a_par_gpu, b_par_gpu, c_par_gpu);

    std::cout << "Float compare - " << (IsEqual(c_seq, c_par_gpu, size * size) == true ? "true" : "false") << std::endl;

    delete[] a_seq;
    delete[] b_seq;
    delete[] c_seq;

    delete[] a_par_cpu;
    delete[] b_par_cpu;
    delete[] c_par_cpu;

    delete[] a_par_gpu;
    delete[] b_par_gpu;
    delete[] c_par_gpu;
    return 0;
}
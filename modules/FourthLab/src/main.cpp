#include <CL/cl.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include "../../../init.h"

float isZero(const float number) {
    if (number < 0.0000000000001)
        return 0.0;
    else
        return number;
}

bool IsEqual(const float* a, const float* b, const size_t n, const float eps) {
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > eps) {
            printf("%d - %lf   %lf\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

bool CheckResult(float* A, float* b, float* x, const size_t size, const float eps) {
    for (size_t i = 0; i < size; ++i) {
        float b_res = 0.0;
        for (size_t j = 0; j < size; ++j)
            b_res += A[i * size + j] * x[j];
        if (std::abs(b_res - b[i]) > eps) {
            std::cout << "i - " << i << ", " << b_res << " - " << b[i] << "\t";
            return false;
        }
    }
    return true;
}

void SequentialGaussMethod(float* A, float* b, float* x, const size_t size) {
    for (size_t k = 0; k < size; ++k) {
        float leaderElem = A[k * size + k];
        if (leaderElem == 0.0) {
            if (k == size - 1) {
                std::cerr << "Matrix is linearly dependent" << std::endl;
                return;
            }
            for (size_t j = k + 1; j < size; ++j) {
                if (A[j * size + k] != 0.0) {
                    std::swap(b[k], b[j]);
                    for (size_t i = 0; i < size; ++i) {
                        std::swap(A[j * size + i], A[k * size + i]);
                    }
                    leaderElem = A[k * size + k];
                    break;
                }
                if (j == size - 1) {
                    std::cerr << "Matrix is linearly dependent" << std::endl;
                    return;
                }
            }
        }
        bool moreElem = leaderElem >= 0.0 ? true : false;
        for (size_t rows = k + 1; rows < size; ++rows) {
            float leaderRow = A[rows * size + k];
            bool moreRow = leaderRow >= 0.0 ? true : false;
            if ((moreElem && moreRow) || (!moreElem && !moreRow)) {
                for (size_t cols = k; cols < size; ++cols) {
                    A[rows * size + cols] = /*isZero*/(A[rows * size + cols] -
                        A[k * size + cols] * std::abs(leaderRow / leaderElem));
                }
                b[rows] = /*isZero*/(b[rows] - b[k] * std::abs(leaderRow / leaderElem));
            }
            else {
                for (size_t cols = k; cols < size; ++cols) {
                    A[rows * size + cols] = /*isZero*/(A[rows * size + cols] +
                        A[k * size + cols] * std::abs(leaderRow / leaderElem));
                }
                b[rows] = /*isZero*/(b[rows] + b[k] * std::abs(leaderRow / leaderElem));
            }
        }
    }
    for (int i = size - 1; i >= 0; --i) {
        float res = 0.0;
        for (size_t j = size - 1; j >= i + 1; --j) {
            res = /*isZero*/(res + A[i * size + j] * x[j]);
        }
        res = /*isZero*/((b[i] - res) / A[i * size + i]);
        x[i] = res;
    }
}

void SequentialJacobiMethod(float* A, float* b, float* x, const size_t size, const int iter) {
    for (size_t i = 0; i < size; ++i) {
        float x_k = 1.0;
        float x_k_1 = 0.0;
        for (int j = 0; j < iter; ++j) {
            float sum = 0.0;
            for (size_t k = 0; k < size; ++k) {
                if (k == i)
                    continue;
                sum += x_k * A[i * size + k];
            }
            //printf("%f\n", sum);
            x_k = x_k_1;
            x_k_1 = (b[i] - sum) / A[i * size + i];
        }
        //printf("%f\n", x_k_1);
        x[i] = x_k_1;
    }
}

void ParallelOpenCLJacobiEPS(const char* path, const size_t group, const unsigned int size,
                             float* A, float* b, float* x, const double eps) {
    size_t group_size[] = { group };
    size_t size_n[] = { size };
    OCLInitialization pr(path, 1, group_size, size_n);
    pr.AddKernel("JacobiEPS");
    pr.AddBuffer<float>(CL_MEM_READ_ONLY, size * size);
    pr.AddBuffer<float>(CL_MEM_READ_ONLY, size);
    pr.AddBuffer<float>(CL_MEM_WRITE_ONLY, size);
    pr.WriteElementsToBuffer(0, size * size, A);
    pr.WriteElementsToBuffer(1, size, b);
    pr.SetKernelArg(pr.GetKernel(0), 0, &pr.GetBuffer(0));
    pr.SetKernelArg(pr.GetKernel(0), 1, &pr.GetBuffer(1));
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetBuffer(2));
    pr.SetKernelArg(pr.GetKernel(0), 3, &size);
    pr.SetKernelArg(pr.GetKernel(0), 4, &eps);
    double start = omp_get_wtime();
    pr.ExecuteKernel(pr.GetKernel(0));
    double end = omp_get_wtime();
    std::cout << "GPU Jacobi EPS - " << end - start << std::endl;
    pr.ReadElementsFromBuffer(2, size, x);
}

void ParallelOpenCLJacobiIter(const char* path, const size_t group, const unsigned int size,
                              float* A, float* b, float* x, const int iter) {
    size_t group_size[] = { group };
    size_t size_n[] = { size };
    OCLInitialization pr(path, 1, group_size, size_n);
    pr.AddKernel("JacobiIter");
    pr.AddBuffer<float>(CL_MEM_READ_ONLY, size * size);
    pr.AddBuffer<float>(CL_MEM_READ_ONLY, size);
    pr.AddBuffer<float>(CL_MEM_WRITE_ONLY, size);
    pr.WriteElementsToBuffer(0, size * size, A);
    pr.WriteElementsToBuffer(1, size, b);
    pr.SetKernelArg(pr.GetKernel(0), 0, &pr.GetBuffer(0));
    pr.SetKernelArg(pr.GetKernel(0), 1, &pr.GetBuffer(1));
    pr.SetKernelArg(pr.GetKernel(0), 2, &pr.GetBuffer(2));
    pr.SetKernelArg(pr.GetKernel(0), 3, &size);
    pr.SetKernelArg(pr.GetKernel(0), 4, &iter);
    double start = omp_get_wtime();
    pr.ExecuteKernel(pr.GetKernel(0));
    double end = omp_get_wtime();
    std::cout << "GPU Jacobi Iter - " << end - start << std::endl;
    pr.ReadElementsFromBuffer(2, size, x);
}


int main(int argc, char** argv) {
    const size_t size = 2048;

    float* A_seq_com = new float[size * size];
    float* b_seq_com = new float[size];

    float* A_seq = new float[size * size];
    float* b_seq = new float[size];
    float* x_seq = new float[size];

    float* A_par_gpu_eps = new float[size * size];
    float* b_par_gpu_eps = new float[size];
    float* x_par_gpu_eps = new float[size];

    float* A_par_gpu_iter = new float[size * size];
    float* b_par_gpu_iter = new float[size];
    float* x_par_gpu_iter = new float[size];

    float lower_bound = -1.0;
    float upper_bound = 1.0;
    std::cout << "gen start" << std::endl;
    std::uniform_real_distribution<float> unif_f(lower_bound, upper_bound);
    std::default_random_engine re;
    for (int i = 0; i < size * size; ++i) {
        float num_f = unif_f(re);
        A_seq_com[i] = A_seq[i] = A_par_gpu_eps[i] = A_par_gpu_iter[i] = num_f;
        if (i % size == i / size)
            A_seq_com[i] = A_seq[i] = A_par_gpu_eps[i] = A_par_gpu_iter[i] = static_cast<float>(size) + num_f;
    }
    for (int i = 0; i < size; ++i) {
        float num_f = unif_f(re);
        b_seq_com[i] = b_seq[i] = b_par_gpu_eps[i] = b_par_gpu_iter[i] = static_cast<float>(1.0 - num_f);
    }
    std::cout << "gen end" << std::endl;

    double start = omp_get_wtime();
    SequentialJacobiMethod(A_seq, b_seq, x_seq, size, 100);
    double end = omp_get_wtime();
    std::cout << "Sequential Gauss method time - " << end - start << std::endl;
    std::cout << "Gauss check result - " << (CheckResult(A_seq_com, b_seq_com, x_seq, size, 0.1) ? "true" : "false") << std::endl;

    ParallelOpenCLJacobiEPS(argv[0], 256, size, A_par_gpu_eps, b_par_gpu_eps, x_par_gpu_eps, 0.000001);
    std::cout << "Jacobi EPS compare with Gauss - " << (IsEqual(x_seq, x_par_gpu_eps, size, 0.001) ? "true" : "false") << std::endl;
    std::cout << "Jacobi EPS check result - " << (CheckResult(A_par_gpu_eps, b_par_gpu_eps, x_par_gpu_eps, size, 0.1) ? "true" : "false") << std::endl;

    ParallelOpenCLJacobiIter(argv[0], 256, size, A_par_gpu_iter, b_par_gpu_iter, x_par_gpu_iter, 45);
    std::cout << "Jacobi Iter compare with Gauss - " << (IsEqual(x_seq, x_par_gpu_iter, size, 0.001) ? "true" : "false") << std::endl;
    std::cout << "Jacobi Iter check result - " << (CheckResult(A_par_gpu_iter, b_par_gpu_iter, x_par_gpu_iter, size, 0.1) ? "true" : "false") << std::endl;

    delete[] A_seq_com;
    delete[] b_seq_com;

    delete[] A_seq;
    delete[] b_seq;
    delete[] x_seq;

    delete[] A_par_gpu_eps;
    delete[] b_par_gpu_eps;
    delete[] x_par_gpu_eps;

    delete[] A_par_gpu_iter;
    delete[] b_par_gpu_iter;
    delete[] x_par_gpu_iter;

    return 0;
}
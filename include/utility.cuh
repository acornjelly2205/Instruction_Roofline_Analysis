
#pragma once


#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// #include <cuda_fp16.hpp>
// ---------- Ginkgo ----------
#include <ginkgo/ginkgo.hpp>

// ---------- BSA ----------
#include "bsa_spmm.cuh"
#include "option.h"
#include "logger.h"
#include "utilities.h"
#include "matrices.h"
#include "spmm.h"
#include "reorder.h"
#include "validate.h"

// ---------- etc ----------
#include "spmm_logger.cuh"
#include "nvtx_helper.cuh"

float run_cuSparse(const CSR &lhs, const ARR &rhs, ARR &result)
{   
    int nnz = lhs.total_nonzeros;
    int m = lhs.rows;
    int n = rhs.cols;
    int k = lhs.cols;

    int *d_rowptr=nullptr, *d_colidx=nullptr;
    float *d_vals=nullptr, *d_B=nullptr, *d_C=nullptr;
    float *d_C_dummy=nullptr;

    cudaMalloc(&d_rowptr, (size_t)(m+1)*sizeof(int));
    cudaMalloc(&d_colidx, (size_t)nnz*sizeof(int));
    cudaMalloc(&d_vals,   (size_t)nnz*sizeof(float));
    cudaMalloc(&d_B,      (size_t)k*n*sizeof(float));
    cudaMalloc(&d_C,      (size_t)m*n*sizeof(float));
    cudaMalloc(&d_C_dummy, (size_t)m*n*sizeof(float));

    cudaMemcpy(d_rowptr, lhs.rowptr, (size_t)(m+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx, lhs.colidx, (size_t)nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals,   lhs.values, (size_t)nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,      rhs.mat, (size_t)k*n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, (size_t)m*n*sizeof(float));
    cudaMemset(d_C_dummy, 0, (size_t)m*n*sizeof(float));

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t A;
    cusparseCreateCsr(  &A, m, k, nnz,
                        d_rowptr, d_colidx, d_vals,
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t B, C, dummy_C;
    cusparseCreateDnMat(&B, k, n, n, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&C, m, n, n, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&dummy_C, m, n, n, d_C_dummy, CUDA_R_32F, CUSPARSE_ORDER_ROW);


    float alpha = 1.0f, beta = 0.0f;
    size_t ws_size=0; void* d_ws=nullptr;
    
    cusparseSpMM_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_size);
    
    if (ws_size>0) cudaMalloc(&d_ws, ws_size);

    
        cusparseSpMM(   handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_ws);

    

    cudaDeviceSynchronize();

    cudaMemcpy(result.mat, d_C, (size_t)m*n*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i =0; i< WARM_UP; i++)
    {
        cusparseSpMM(   handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, A, B, &beta, dummy_C, CUDA_R_32F,
                        CUSPARSE_SPMM_ALG_DEFAULT, d_ws);
    }

    cudaDeviceSynchronize();

    float time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(int i =0; i< ITERATIONS; i++)
    {
        cusparseSpMM(   handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, A, B, &beta, dummy_C, CUDA_R_32F,
                        CUSPARSE_SPMM_ALG_DEFAULT, d_ws);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_rowptr);
    cudaFree(d_colidx);
    cudaFree(d_vals);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_dummy);
    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(B);
    cusparseDestroyDnMat(C);
    cusparseDestroy(handle);    

    if(ITERATIONS == 0) return 0.0f; // Return average time per iteration
    return time / ITERATIONS;

}


float run_ginkgo(const CSR &lhs, const ARR &rhs, ARR &result)
{   
    int nnz = lhs.total_nonzeros;
    int m = lhs.rows;
    int n = rhs.cols;
    int k = lhs.cols;

    auto omp = gko::OmpExecutor::create();
    auto exec = gko::CudaExecutor::create(0, omp);  // GPU 0

    auto A_h = gko::matrix::Csr<float,int>::create(omp, gko::dim<2>(m,k), (size_t)lhs.total_nonzeros);
    std::copy_n(lhs.rowptr,  m+1,         A_h->get_row_ptrs());
    std::copy_n(lhs.colidx,  lhs.total_nonzeros, A_h->get_col_idxs());
    std::copy_n(lhs.values,  lhs.total_nonzeros, A_h->get_values());
    // 2) Device로 복사
    auto A = gko::matrix::Csr<float,int>::create(exec);
    A->copy_from(A_h.get());

    auto B_h = gko::matrix::Dense<float>::create(omp, gko::dim<2>(k,n));
    std::copy_n(rhs.mat, (size_t)k*n, B_h->get_values());
    auto B   = gko::matrix::Dense<float>::create(exec);
    B->copy_from(B_h.get());

    auto C = gko::matrix::Dense<float>::create(exec, gko::dim<2>(m, n));
    C->fill(0.0f);

    auto C_dummy = gko::matrix::Dense<float>::create(exec, gko::dim<2>(m, n));
    C_dummy->fill(0.0f);


    A->apply(B.get(), C.get());

    
    exec->synchronize();

    auto C_h = gko::matrix::Dense<float>::create(omp, gko::dim<2>(m,n));
    C_h->copy_from(C.get());
    std::copy_n(C_h->get_values(), (size_t)m*n, result.mat);
    
    for (int it=0; it< WARM_UP; ++it) A->apply(B.get(), C_dummy.get());
    exec->synchronize();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int it=0; it< ITERATIONS; ++it) A->apply(B.get(), C_dummy.get());
    exec->synchronize();
    cudaEventRecord(stop);  
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (ITERATIONS == 0) return 0.0f; // Return average time per iteration
    return elapsed_time / ITERATIONS;

}

float validate_results_with_host(const ARR &result, const CSR &lhs, const ARR &rhs)
{   
    int nnz = lhs.total_nonzeros;
    int m = lhs.rows;
    int n = rhs.cols;
    int k = lhs.cols;

    float *answer = new float[m*n];

    for(int r =0; r< m; r++)
    {
        for(int c =0; c< n; c++)
        {
            answer[r*n + c] = 0.0f;
            int start = lhs.rowptr[r];
            int end = lhs.rowptr[r+1];

            for(int k = start; k < end; k++)
            {
                int col = lhs.colidx[k];
                answer[r*n + c] += lhs.values[k] * rhs.mat[col*n + c];
            }
        }
    }

    for(int i =0; i< m; i++)
    {
        int nnz_row = lhs.rowptr[i+1] - lhs.rowptr[i];
        double epsilon = (1e-6f * nnz_row);

        for(int j =0; j<n; j++)
        {
            if (fabs(result.mat[i*n + j] - answer[i*n + j]) > epsilon)
            {
                cout << "Validation failed at (" << i << ", " << j << "): "
                     << "Expected: " << answer[i*n + j] 
                     << ", Got: " << result.mat[i*n + j] << endl;


                float diff = fabs(result.mat[i*n + j] - answer[i*n + j]);
                delete[] answer;
                return diff;
            }
        }
    }

    cout << "Validation successful!" << endl;
    delete[] answer;
    return 0.0f;
}

double gflops(float ms, long long nnz, int n)
{
    double fl = 2.0 * (double)nnz * (double)n;
    return fl / (ms*1e-3) / 1e9;
}






#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif


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
#include "utility.cuh"

// #define WARM_UP 100
// #define ITERATIONS 10

using namespace std;

#define cuSPARSE_COL palette::blue
#define GINKGO_COL palette::orange
#define KOKKOS_COL palette::green

int main(int argc, char *argv[])
{
    Option option = Option(argc, argv);

    option.n_cols = 128;
    option.output_filename = "result/cuSPARSE_results.csv";
    
    option.input_format = FileFormatType::smtx;
    // option.input_format = FileFormatType::mtx;

    option.compress_rows = false;
    option.zero_padding = true;
    option.pattern_only = false;

    CSR lhs = CSR(option);
    SpMM_LOGGER logger = SpMM_LOGGER(option);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    random_device rd;
    mt19937 e2(rd());
    uniform_real_distribution<> dist(0, 1);

    lhs.values = new DataT[lhs.total_nonzeros];

    for(int i =0; i< lhs.total_nonzeros; i++)
    {
        lhs.values[i] = dist(e2);
    } 
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    ARR rhs = ARR(lhs.original_cols, lhs.cols, option.n_cols, true);
    rhs.fill_random(option.zero_padding);
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    ARR cusparse_result = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
    ARR Ginkgo_result = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
    ARR Kokkos_kernel_result = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
    

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    logger.cusparse_time = run_cuSparse(lhs, rhs, cusparse_result);
    logger.cusparse_error = validate_results_with_host(cusparse_result, lhs, rhs);
    if(logger.cusparse_error > 0) logger.cusparse_result = RESULTS::FAILURE;
    else logger.cusparse_result = RESULTS::SUCCESS;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]



    // logger.ginkgo_time = run_ginkgo(lhs, rhs, Ginkgo_result);
    // logger.ginkgo_error = validate_results_with_host(Ginkgo_result, lhs, rhs);
    // if(logger.ginkgo_error > 0) logger.ginkgo_result = RESULTS::FAILURE;
    // else logger.ginkgo_result = RESULTS::SUCCESS;



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    // Kokkos::initialize(argc, argv);
    // {
    //     Kokkos::print_configuration(std::cout);
    //     std::cout << "DefaultExecutionSpace = " << Kokkos::DefaultExecutionSpace::name() << "\n";

    //     Kokkos::Profiling::pushRegion("Kokkos SpMM");
    //     logger.kokkos_time = run_kokkos(lhs, rhs, Kokkos_kernel_result);
    //     Kokkos::Profiling::popRegion();
    //     Kokkos::fence();
    //     cudaDeviceSynchronize();

    //     logger.kokkos_error = validate_results_with_host(Kokkos_kernel_result, lhs, rhs);
    //     if(logger.kokkos_error > 0) logger.kokkos_result = RESULTS::FAILURE;
    //     else logger.kokkos_result = RESULTS::SUCCESS;

    // }
    // Kokkos::finalize();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]

    logger.infile = option.input_filename;
    logger.outfile = option.output_filename;
    logger.repetitions = ITERATIONS;
    logger.M = lhs.rows;
    logger.N = rhs.cols;
    logger.K = lhs.cols;
    logger.NNZ = lhs.total_nonzeros;
    logger.density = static_cast<float>(lhs.total_nonzeros) / (lhs.rows * lhs.cols);

    logger.MU = (float)logger.NNZ / logger.M;
    
    int max_nnz = 0;
    for(int i =0;i < lhs.rows; i++)
    {
        if(lhs.rowptr[i+1] - lhs.rowptr[i] > max_nnz)
            max_nnz = lhs.rowptr[i+1] - lhs.rowptr[i];
    }
    logger.MAX = max_nnz;

    float diff = 0.0f;
    for(int i =0; i< lhs.rows; i++) diff += (lhs.rowptr[i+1] - lhs.rowptr[i] - logger.MU) * (lhs.rowptr[i+1] - lhs.rowptr[i] - logger.MU);
    float avg_diff = diff / lhs.rows;
    logger.STD_NNZ = sqrt(avg_diff);

    logger.MAX_MU = logger.MAX - logger.MU;

    int distance = 0;
    for(int i =0; i< lhs.rows; i++)
    {
        distance += abs(lhs.colidx[lhs.rowptr[i+1] - 1] - lhs.colidx[lhs.rowptr[i]]);
    }

    logger.AVE_BW = distance / lhs.rows;
 
    float diff_bw = 0.0f;
    for(int i =0; i< lhs.rows; i++)
    {
        diff_bw += (abs(lhs.colidx[lhs.rowptr[i+1] - 1] - lhs.colidx[lhs.rowptr[i]]) - logger.AVE_BW) * (abs(lhs.colidx[lhs.rowptr[i+1] - 1] - lhs.colidx[lhs.rowptr[i]]) - logger.AVE_BW);
    }
    float avg_diff_bw = diff_bw / lhs.rows;
    logger.STD_BW = sqrt(avg_diff_bw);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////[]


    cout << endl << "===========================" << endl;
    cout << "Input File: " << option.input_filename << endl;
    cout << "Output File: " << option.output_filename << endl;
    cout << "Repetitions: " << ITERATIONS << endl;
    cout << "M, N, K: " << lhs.rows << ", " << rhs.cols << ", " << lhs.cols << endl;
    cout << "Density: " << logger.density << endl;
    cout << "NNZ: " << lhs.total_nonzeros << endl;
    cout << "MU: " << logger.MU << endl;
    cout << "MAX: " << logger.MAX << endl;
    cout << "STD_NNZ: " << logger.STD_NNZ << endl;
    cout << "MAX_MU: " << logger.MAX_MU << endl;
    cout << "AVE_BW: " << logger.AVE_BW << endl;
    cout << "STD_BW: " << logger.STD_BW << endl;

    cout << endl << "===========================" << endl;
    cout << "cuSPARSE Time: " << logger.cusparse_time << " seconds" << endl;
    cout << "cuSPARSE Error: " << logger.cusparse_error << endl;
    cout << "cuSPARSE Result: " << (logger.cusparse_result == RESULTS::SUCCESS ? "SUCCESS" : "FAILURE") << endl;
    cout << endl << "===========================" << endl;
    cout << "Ginkgo Time: " << logger.ginkgo_time << " seconds" << endl;
    cout << "Ginkgo Error: " << logger.ginkgo_error << endl;
    cout << "Ginkgo Result: " << (logger.ginkgo_result == RESULTS::SUCCESS ? "SUCCESS" : "FAILURE") << endl;
    cout << endl << "===========================" << endl;
    cout << "Kokkos Time: " << logger.kokkos_time << " seconds" << endl;
    cout << "Kokkos Error: " << logger.kokkos_error << endl;
    cout << "Kokkos Result: " << (logger.kokkos_result == RESULTS::SUCCESS ? "SUCCESS" : "FAILURE") << endl;
    cout << endl << "===========================" << endl;


    if (option.output_filename.length())
        logger.save_logfile();
}



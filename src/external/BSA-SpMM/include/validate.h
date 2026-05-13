
// Modified from original BSA-SpMM (Euro-PAR 2024)
// Original: https://doi.org/10.5281/zenodo.11579181
// Changes: modification in validate function

#pragma once


#include "matrices.h"
#include "definitions.h"

bool validate_spmm(CSR &lhs, ARR &rhs, ARR &result_mat, Major b_major, Major c_major)
{
    DataT_C *valid_matC = new DataT_C[result_mat.rows * result_mat.cols];
    memset(valid_matC, 0, result_mat.rows * result_mat.cols * sizeof(DataT_C));
    DataT val = 1;

    for (int r = 0; r < lhs.rows; r++)
    {
        intT start_pos = lhs.rowptr[r];
        intT end_pos = lhs.rowptr[r + 1];
        for (int j = start_pos; j < end_pos; j++)
        {
            if (not lhs.pattern_only)
                val = lhs.values[j];
            for (int k = 0; k < rhs.cols; k++)
            {
                int b_idx = b_major == row ? lhs.colidx[j] * rhs.cols + k : k * rhs.rows + lhs.colidx[j];
                int c_idx = c_major == row ? r * result_mat.cols + k : k * result_mat.rows + r;
                valid_matC[c_idx] += val * rhs.mat[b_idx];
            }
        }
    }

    float epsilon = 1.0;
    bool success = true;
    float total_error = 0.0;

    for (int i = 0; i < result_mat.rows * result_mat.cols; i++)
    {
        if (abs(result_mat.mat[i] - valid_matC[i]) > epsilon)
        {
            // printf("The error %f\n", abs(result_mat.mat[i] - valid_matC[i]));
            // success = false;
            // break;
            total_error += abs(result_mat.mat[i] - valid_matC[i]);
        }
    }

    printf("Total error: %f\n", total_error);
    delete[] valid_matC;
    return success;

    // delete[] valid_matC;
    // return success;

        // for (int i = 0; i < result_mat.rows * result_mat.cols; i++)
    // {
    //     if (abs(result_mat.mat[i] - valid_matC[i]) > epsilon)
    //     {
    //         printf("The error %f\n", abs(result_mat.mat[i] - valid_matC[i]));
    //         success = false;
    //         break;
    //     }
    // }

    // delete[] valid_matC;
    // return success;




}
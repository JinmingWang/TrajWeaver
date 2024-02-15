// dllmain.cpp : Defines the entry point for the DLL application.
// #define EXPORT __declspec(dllexport)
#include <iostream>
#include <string>


using namespace std;

extern "C" {
    int* accumulateCount(int* row_indices, int* col_indices, int n_grids, int n_indices) {
        long n_elements = n_grids * n_grids;
        int* count_mat = new int[n_elements];

        for (int i = 0; i < n_elements; i++) {
            count_mat[i] = 0;
        }

        for (int ii = 0; ii < n_indices; ii++) {
            int i = row_indices[ii] * n_grids + col_indices[ii];
            count_mat[i] ++;
        }
        return count_mat;
    }

    float NDTW(float* dist_mat, float* dtw_mat, int rows, int cols) {
        // dist_mat: rows * cols
        // dtw_mat: (rows + 1) * (cols + 1)
        for (int r = 1; r < rows + 1; r++) {
            for (int c = 1; c < cols + 1; c++) {
                float point_dist = dist_mat[(r - 1) * cols + (c - 1)];
                float min_dist = min(dtw_mat[(r - 1) * (cols + 1) + c], dtw_mat[r * (cols + 1) + (c - 1)]);
                min_dist = min(min_dist, dtw_mat[(r - 1) * (cols + 1) + (c - 1)]);
                dtw_mat[r * (cols + 1) + c] = point_dist + min_dist;
            }
        }
        return dtw_mat[(rows + 1) * (cols + 1) - 1] / (float)cols;
    }
}

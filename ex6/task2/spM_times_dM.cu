#include "timer.hpp"
#include "poisson2d.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
// #include <cublas_v2.h>
// #include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#define BLOCK_SIZE 256
#define GRID_SIZE 256
// #define SEP ";"
#define TESTS 5

// #define DEBUG
#ifndef DEBUG
  #define CSV
#endif

// START--------------- CONVENIENCE FUNTIONS ------------------START // 
// template <typename T>
// void printContainer(T* container, const int size) {
//   std::cout << *container;
//   for (int i = 1; i < size; ++i) 
//     std::cout << " | " << *(container+i) ;
//   std::cout << std::endl;
// }

template <typename T>
void printContainer(T container, const int size) {
  std::cout << container[0];
  for (int i = 1; i < size; ++i) 
    std::cout << " | " << container[i] ;
  std::cout << std::endl;
}

template <typename T>
void printContainer(T container, const int size, const int only) {
  std::cout << container[0];
  for (int i = 1; i < only; ++i) 
      std::cout  << " | " << container[i];
  std::cout << " | ...";
  for (int i = size - only; i < size; ++i) 
    std::cout  << " | " << container[i];
  std::cout << std::endl;
}

template <typename T>
void printContainerStrided(T container, const int size, const int stride) {
  std::cout << container[0];
  for (int i = stride; i < size; i+=stride) 
      std::cout  << " | " << container[i];
  std::cout << std::endl;
}

void printResults(double* results, std::vector<std::string> names, int size){
  std::cout << "Results:" << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cout << names[i] << " : " << results[i] << std::endl;
  }
}

void printResults(double* results, double* ref, std::vector<std::string> names, int size){
  std::cout << "Results (with difference to reference):" << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cout << names[i] << " = " << results[i] << " ||  " << ref[i] - results[i] << std::endl;
  }
}

template <typename T>
void toCSV(std::fstream& csv, T* array, int size) {
  csv << size;
  for (int i = 0; i < size; ++i) {
    csv << ";" << array[i];
  }
  csv << std::endl;
}

double median(std::vector<double>& vec)
{
  size_t size = vec.size();
  if (size == 0)
    return 0.;

  sort(vec.begin(), vec.end());
  size_t mid = size/2;
  return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}
// END--------------- CONVENIENCE FUNCTIONS ------------------END // 

//
// START--------------- KERNELS ------------------START // 
//
// y = A * x
__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
  int *csr_colindices, double *csr_values,
  double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * x[csr_colindices[k]];
    }
    y[i] = sum;
  }
}

// Y= A * X
__global__ void A_MatMul_Xrm(int N, int K,
  int *csr_rowoffsets, int *csr_colindices, double *csr_values,
  double *X, double *Y)
{
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    int row_start = csr_rowoffsets[tid];
    int row_end = csr_rowoffsets[tid + 1];

    // for (int k = 0; k < K; ++k){
    //   double sum = 0.0;
    //   for (int i = row_start; i < row_end; i++) {
    //     sum += csr_values[i]* X[csr_colindices[i]*K + k];
    //   }
    //   Y[k + tid*K] = sum;
    // }
    for (int k = 0; k < K; ++k){
      double sum = 0.0;
      for (int i = row_start; i < row_end; i++) {
        sum += csr_values[i]* X[csr_colindices[i]*K + k];
      }
      Y[k + tid*K] = sum;
    }

    // for (int i = row_start; i < row_end; i++) {
    //   double aij = csr_values[i];
    //   int row_of_X = csr_colindices[i]*K;
    //   for (int k = 0; k < K; ++k){
    //     Y[k + tid*K] += aij * X[row_of_X + k];
    //   }
    // }
  }
}

// Y= A * X
__global__ void A_MatMul_Xcm(int N, int K,
  int *csr_rowoffsets, int *csr_colindices, double *csr_values,
  double *X, double *Y)
{
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    int row_start = csr_rowoffsets[tid];
    int row_end = csr_rowoffsets[tid + 1];

    for (int k = 0; k < K; ++k){
      double sum = 0.0;
      for (int i = row_start; i < row_end; i++) {
        sum += csr_values[i]* X[csr_colindices[i] + k*N];
      }
      Y[k + tid*K] = sum;
    }
  }
}


// Y= A * X
__global__ void A_MatMul_Xrm_ik(int N, int K,
  int *csr_rowoffsets, int *csr_colindices, double *csr_values,
  double *X, double *Y)
{
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    int row_start = csr_rowoffsets[tid];
    int row_end = csr_rowoffsets[tid + 1];

    for (int i = row_start; i < row_end; i++) {
      double aij = csr_values[i];
      int colindex = csr_colindices[i];
      for (int k = 0; k < K; ++k){
        Y[k + tid*K] += aij * X[colindex*K + k];
      }
    }
  }
}

// Y= A * X
__global__ void A_MatMul_Xcm_ik(int N, int K,
  int *csr_rowoffsets, int *csr_colindices, double *csr_values,
  double *X, double *Y)
{
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x) {
    int row_start = csr_rowoffsets[tid];
    int row_end = csr_rowoffsets[tid + 1];

    for (int i = row_start; i < row_end; i++) {
      double aij = csr_values[i];
      int colindex = csr_colindices[i];
      for (int k = 0; k < K; ++k){
        Y[k + tid*K] += aij* X[colindex + k*N];
      }
    }
  }
}

// // Y = A * X
// __global__ void A_MatMul_Xrm(int N, int K, int *csr_rowoffsets,
//   int *csr_colindices, double *csr_values,
//   double *x, double *y)
// {
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

//     for (int k = 0; k < K; k++) {
//       double sum = 0;

//       for (int jj = csr_rowoffsets[i]; jj < csr_rowoffsets[i + 1]; jj++) {
//         sum += csr_values[jj] * x[csr_colindices[jj]*K + k];
//       }

//       y[i + N*k] = sum;
//     }

//   }
// }


// // Y = A * X
// __global__ void A_MatMul_Xcm(int N, int K, int *csr_rowoffsets,
//   int *csr_colindices, double *csr_values,
//   double *x, double *y)
// {
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {

//     for (int k = 0; k < K; k++) {
//       double sum = 0;

//       for (int jj = csr_rowoffsets[i]; jj < csr_rowoffsets[i + 1]; jj++) {
//         sum += csr_values[jj] * x[csr_colindices[jj] + N*k];
//       }

//       y[i + N*k] = sum;
//     }

//   }
// }
// END--------------- KERNELS ------------------END // 
//

int main(void) {
  Timer timer;
  std::vector<int> vec_Ns{100, 1024, 10000, 100489, 1000000};
  // std::vector<int> vec_Ns{1000000};
  std::vector<int> vec_Ks{2, 4, 8, 16};
  // std::vector<int> vec_Ks{3, 5, 9, 15};
  std::vector<double> times(TESTS, 0);

#ifdef CSV
  std::fstream csv_times;
  std::string csv_times_name = "ph_data_ik.csv";
  csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);

  std::string header = "N;K;time_single;time_rm_stacked;time_cm_stacked";
    // to csv file
  csv_times << header << std::endl;
#endif

  for (int& N : vec_Ns) {
    for (int& K : vec_Ks) {
      // cublasHandle_t h;
      // cublasCreate(&h);

      //
      // allocate + init host memory:
      //
  #ifdef DEBUG
      std::cout << "Allocating host + device arrays..." << std::endl;
  #endif
      // "Vectors"
      double* X = (double *)malloc(sizeof(double) * N * K);
      double* Y = (double *)malloc(sizeof(double) * N * K);
      double* Y2 = (double *)malloc(sizeof(double) * N * K);
      // double* x = (double *)malloc(sizeof(double) * N);
      double* y = (double *)malloc(sizeof(double) * N);
      std::fill(X, X + (N*K), 1.);
      std::fill(Y, Y + (N*K), 0.);
      std::fill(Y2, Y2 + (N*K), 0.);
      // std::fill(x, x + N, 1.);

      double *cuda_X;
      double *cuda_Y;
      // double *cuda_Y2;
      // double *cuda_x;
      double *cuda_y;
      cudaMalloc(&cuda_X, sizeof(double) * N*K);
      cudaMalloc(&cuda_Y, sizeof(double) * N*K);
      // cudaMalloc(&cuda_Y2, sizeof(double) * N*K);
      // cudaMalloc(&cuda_x, sizeof(double) * N);
      cudaMalloc(&cuda_y, sizeof(double) * N);

      // Matrix
      int* csr_rowoffsets = (int* )malloc(sizeof(int) * (N+1));
      int* csr_colindices = (int* )malloc(sizeof(int) * 5*N);
      double* csr_values = (double* )malloc(sizeof(double) * 5*N);

      int* cuda_csr_rowoffsets; 
      int* cuda_csr_colindices;
      double* cuda_csr_values;
      cudaMalloc(&cuda_csr_rowoffsets, sizeof(int) * (N+1));
      cudaMalloc(&cuda_csr_colindices, sizeof(int) * 5*N);
      cudaMalloc(&cuda_csr_values, sizeof(double) * 5*N);
      //
      // Copy data to GPU
      //
  #ifdef DEBUG
      std::cout << "Copying data to GPU..." << std::endl;
  #endif
      cudaMemcpy(cuda_X, X, sizeof(double) * N*K, cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_Y, Y, sizeof(double) * N*K, cudaMemcpyHostToDevice);
      // cudaMemcpy(cuda_Y2, Y2, sizeof(double) * N*K, cudaMemcpyHostToDevice);
      // cudaMemcpy(cuda_x, X, sizeof(double) * N, cudaMemcpyHostToDevice);
  //    cudaMemcpy(cuda_y, y, sizeof(double) * N*K, cudaMemcpyHostToDevice);

  // Assemble A
  #ifdef DEBUG
      std::cout << "Generating A..." << std::endl;
  #endif
      generate_fdm_laplace(sqrt(N), csr_rowoffsets, csr_colindices, csr_values);
  #ifdef DEBUG
      std::cout << "Generating A done!" << std::endl;
  #endif
      cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(int) * (N+1), cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(int) * 5*N, cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_csr_values, csr_values, sizeof(double) * 5*N, cudaMemcpyHostToDevice);  
      
      // ------------------ TEST ---------------- //

#ifdef DEBUG
      std::cout << "N = " << N << std::endl;
      std::cout << "K = " << K << std::endl;

      std::cout << "Running per vector product kernel K times..." << std::endl;
#endif
      for (int iter = 0; iter < TESTS; iter++){
        timer.reset();
        for (int k = 0; k < K; ++k)
          cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(
            N, 
            cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values,
            cuda_X, cuda_y);
        cudaMemcpy(y, cuda_y, sizeof(double) * N, cudaMemcpyDeviceToHost);
        times[iter] = timer.get();
      } 
      double time_single = median(times);

  #ifdef DEBUG
      std::cout << "Running RowMajor stacked kernel..." << std::endl;
  #endif
      for (int iter = 0; iter < TESTS; iter++){
        timer.reset();
        A_MatMul_Xrm_ik<<<GRID_SIZE, BLOCK_SIZE>>>(
            N, K,
            cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values,
            cuda_X, cuda_Y);
        cudaMemcpy(Y, cuda_Y, sizeof(double) * N*K, cudaMemcpyDeviceToHost);
        times[iter] = timer.get();
      }
      double time_rm_stacked = median(times);

  #ifdef DEBUG
      std::cout << "Running ColumnMajor stacked kernel..." << std::endl;
  #endif
      cudaMemcpy(cuda_Y, Y2, sizeof(double) * N*K, cudaMemcpyHostToDevice);
      for (int iter = 0; iter < TESTS; iter++){
        timer.reset();
        A_MatMul_Xcm_ik<<<GRID_SIZE, BLOCK_SIZE>>>(
            N, K,
            cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values,
            cuda_X, cuda_Y);
        cudaMemcpy(Y2, cuda_Y, sizeof(double) * N*K, cudaMemcpyDeviceToHost);
        times[iter] = timer.get();
      }
      double time_cm_stacked = median(times);




      //
      // Compare results
      //
  #ifdef DEBUG
      std::cout << "DEBUG output:" << std::endl;
      // int only = 4;
      std::cout << "A (non zero entries by row)" << std::endl;
      // int csr_values_size = csr_rowoffsets[N+1];
      // printContainer(y, N);
      std::cout << "Row" << std::endl;
      int max_output = 10;
      for (int row = 0; row < min(N, max_output); row++){
        std::cout << row << ": ";
        printContainer(csr_values + csr_rowoffsets[row], min(csr_rowoffsets[row+1]-csr_rowoffsets[row], max_output));
      }

      std::cout << "y:" << std::endl;
      printContainer(y, min(N, max_output));
      std::cout << "Y_rm:" << std::endl;
      printContainerStrided(Y, min(N, max_output)*K, K);
      std::cout << "Y_cm:" << std::endl;
      printContainerStrided(Y2, min(N, max_output)*K, K);


      std::cout << "Single runtime: " << time_single << std::endl;
      std::cout << "RM Stacked runtime: " << time_rm_stacked << std::endl;
      std::cout << "CM Stacked runtime: " << time_cm_stacked << std::endl;

      //
      // Clean up:
      //
      std::cout << "----------------------------------------------------" << std::endl;
      std::cout << "Cleaning up..." << std::endl;
  #endif

  #ifdef CSV
      std::string sep = ";";
      csv_times << N << sep
                << K << sep
                << time_single << sep
                << time_rm_stacked << sep
                << time_cm_stacked 
                << std::endl;
  #endif
    free(X);
    free(Y);
    free(Y2);
    // free(x);
    free(y);
    free(csr_rowoffsets); 
    free(csr_colindices);
    free(csr_values);

    cudaFree(cuda_X);
    cudaFree(cuda_Y);
    // cudaFree(cuda_Y2);
    // cudaFree(cuda_x);
    cudaFree(cuda_y);
    cudaFree(cuda_csr_rowoffsets); 
    cudaFree(cuda_csr_colindices);
    cudaFree(cuda_csr_values);
#ifdef DEBUG
    std::cout << "Clean up done!" << std::endl;
#endif
    }
  }

#ifdef CSV
  csv_times.close();
  
  std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex6/" + csv_times_name << std::endl;
#endif
  return EXIT_SUCCESS;
}

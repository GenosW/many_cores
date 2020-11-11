#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "timer.hpp"

#define BLOCK_SIZE 256
#define GRID_SIZE 256

__global__ void xDOTy(const size_t N, double* x, 
  double* y,
  double* z)
{
  size_t tid = threadIdx.x + blockDim.x* blockIdx.x;
  const size_t stride = blockDim.x* gridDim.x;

  __shared__ double cache[BLOCK_SIZE];

  double tid_sum = 0.0;
  for (; tid < N; tid += stride)
  {
    double tmp_x = x[tid];
    tid_sum +=  tmp_x * y[tid];
  }
  tid = threadIdx.x;
  cache[tid] = tid_sum;

  __syncthreads();
  for (size_t i = blockDim.x/2; i != 0; i /=2)
  {
    __syncthreads();
    if (tid < i) //lower half does smth, rest idles
      cache[tid] += cache[tid + i]; //lower looks up by stride and sums up
  }

  if(tid == 0) // cache[0] now contains block_sum
  {
    atomicAdd(z, cache[0]);
  }
}

/** Computes 8 vector dot products of type <x,y_i> with i=1,...,8 at once

z should be a pointer to an array of size 8 to store the results.
*/
__global__ void xDOTy8(const size_t N, double* x, 
                      double* y1, double* y2,
                      double* y3, double* y4,
                      double* y5, double* y6,
                      double* y7, double* y8,
                      double* z)
{
  size_t tid = threadIdx.x + blockDim.x* blockIdx.x;
  const size_t stride = blockDim.x* gridDim.x;

  __shared__ double cache[8][BLOCK_SIZE];

  double tid_sum1 = 0.0; double tid_sum2 = 0.0;
  double tid_sum3 = 0.0; double tid_sum4 = 0.0;
  double tid_sum5 = 0.0; double tid_sum6 = 0.0;
  double tid_sum7 = 0.0; double tid_sum8 = 0.0;
  for (; tid < N; tid += stride)
  {
    double tmp_x = x[tid];
    tid_sum1 +=  tmp_x * y1[tid];
    tid_sum2 +=  tmp_x * y2[tid];
    tid_sum3 +=  tmp_x * y3[tid];
    tid_sum4 +=  tmp_x * y4[tid];
    tid_sum5 +=  tmp_x * y5[tid];
    tid_sum6 +=  tmp_x * y6[tid];
    tid_sum7 +=  tmp_x * y7[tid];
    tid_sum8 +=  tmp_x * y8[tid];
  }
  tid = threadIdx.x;
  cache[0][tid] = tid_sum1; cache[1][tid] = tid_sum2;
  cache[2][tid] = tid_sum3; cache[3][tid] = tid_sum4;
  cache[4][tid] = tid_sum5; cache[5][tid] = tid_sum6;
  cache[6][tid] = tid_sum7; cache[7][tid] = tid_sum8;

  __syncthreads();
  for (size_t i = blockDim.x/2; i != 0; i /=2)
  {
    __syncthreads();
    if (tid < i) { //lower half
      cache[0][tid] += cache[0][tid + i];
      cache[1][tid] += cache[1][tid + i];
      cache[2][tid] += cache[2][tid + i];
      cache[4][tid] += cache[4][tid + i];
    // }
    // else if (tid < i*2){
      cache[5][tid] += cache[5][tid + i];
      cache[6][tid] += cache[6][tid + i];
      cache[7][tid] += cache[7][tid + i];
      cache[8][tid] += cache[8][tid + i];
    }

  }

  if (tid==0) // cache[0] now contains block_sum
  {
    for (int i = 0; i < 8; ++i)
    atomicAdd(z+i, cache[i][0]);
  }
}

int main(void)
{
    const size_t N = 100000;
    const size_t K = 8;
    Timer timer;

    //
    // Initialize CUBLAS:
    //
    std::cout << "Init CUBLAS..." << std::endl;
    cublasHandle_t h;
    cublasCreate(&h);


    //
    // allocate host memory:
    //
    std::cout << "Allocating host arrays..." << std::endl;
    double  *x = (double*)malloc(sizeof(double) * N);
    double **y = (double**)malloc(sizeof(double*) * K);
    for (size_t i=0; i<K; ++i) {
      y[i] = (double*)malloc(sizeof(double) * N);
    }
    double *results  = (double*)malloc(sizeof(double) * K);
    double *results2 = (double*)malloc(sizeof(double) * K);
    double *results_xDOTy = (double*)malloc(sizeof(double) * K);
    double *results_xDOTy8 = (double*)malloc(sizeof(double) * K);
    std::fill(results_xDOTy, results_xDOTy+K, 0.0);
    std::fill(results_xDOTy8, results_xDOTy8+K, 0.0);


    //
    // allocate device memory
    //
    std::cout << "Allocating CUDA arrays..." << std::endl;
    double *cuda_x; cudaMalloc(&cuda_x, sizeof(double)*N);
    double **cuda_y = (double**)malloc(sizeof(double*) * K);  // storing CUDA 
    double *cuda_results; cudaMalloc(&cuda_results, sizeof(double)*K);
    for (size_t i=0; i<K; ++i) {
      cudaMalloc( (void **)(&cuda_y[i]), sizeof(double)*N);
    }

    //
    // fill host arrays with values
    //
    for (size_t j=0; j<N; ++j) {
      x[j] = 1 + j%K;
    }
    for (size_t i=0; i<K; ++i) {
      for (size_t j=0; j<N; ++j) {
        y[i][j] = 1 + rand() / (1.1 * RAND_MAX);
      }
    }

    //
    // Reference calculation on CPU:
    //
    timer.reset();
    for (size_t i=0; i<K; ++i) {
      results[i] = 0;
      results2[i] = 0;
      for (size_t j=0; j<N; ++j) {
        results[i] += x[j] * y[i][j];
      }
    }    
    double time_CPU = timer.get();
   
    //
    // Copy data to GPU
    //
    std::cout << "Copying data to GPU..." << std::endl;
    cudaMemcpy(cuda_x, x, sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_results, results_xDOTy8, sizeof(double)*K, cudaMemcpyHostToDevice);
    for (size_t i=0; i<K; ++i) {
      cudaMemcpy(cuda_y[i], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
    }


    //
    // Let CUBLAS do the work:
    //
    std::cout << "Running dot products with CUBLAS..." << std::endl;
    timer.reset();
    for (size_t i=0; i<K; ++i) {
      cublasDdot(h, N, cuda_x, 1, cuda_y[i], 1, results2 + i);
    }
    double time_cublas = timer.get();

    //
    // Let xDOTy do the work:
    //
    std::cout << "Running dot products with custom xDOTy8..." << std::endl;
    timer.reset();
    for (size_t i=0; i<K; ++i) {
      xDOTy<<<GRID_SIZE, BLOCK_SIZE>>>(N, 
            cuda_x, cuda_y[i],
            cuda_results+i);
    }
    cudaMemcpy(results_xDOTy, cuda_results, sizeof(double)*K, cudaMemcpyDeviceToHost);
    double time_xDOTy = timer.get();

    //
    // Let xDOTy8 do the work:
    //
    cudaMemcpy(cuda_results, results_xDOTy8, sizeof(double)*K, cudaMemcpyHostToDevice);
    std::cout << "Running dot products with custom xDOTy8..." << std::endl;
    timer.reset();
    for (size_t i=0; i<(int)K/8; ++i) {
      int batch_offset = (i*8);
      xDOTy8<<<GRID_SIZE, BLOCK_SIZE>>>(N, 
            cuda_x, 
            cuda_y[batch_offset], cuda_y[batch_offset+1],
            cuda_y[batch_offset+2], cuda_y[batch_offset+3],
            cuda_y[batch_offset+4], cuda_y[batch_offset+5],
            cuda_y[batch_offset+6], cuda_y[batch_offset+7],
            cuda_results+batch_offset);
    }
    cudaMemcpy(results_xDOTy8, cuda_results, sizeof(double)*K, cudaMemcpyDeviceToHost);
    double time_xDOTy8 = timer.get();

    //
    // Compare results
    //
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Copying results back to host..." << std::endl;
    for (size_t i=0; i<K; ++i) {
      std::cout << results[i] << " on CPU, " << results2[i] << " on GPU. Relative difference: " << fabs(results[i] - results2[i]) / results[i] << std::endl;
    }std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Now to compare the custom kernel xDOTy to CPU..." << std::endl;
    for (size_t i=0; i<K; ++i) {
      std::cout << results[i] << " on CPU, " << results_xDOTy[i] << " on GPU. Relative difference: " << fabs(results[i] - results_xDOTy[i]) / results[i] << std::endl;
    }
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Now to compare the custom kernel xDOTy8 to CPU..." << std::endl;
    for (size_t i=0; i<K; ++i) {
      std::cout << results[i] << " on CPU, " << results_xDOTy8[i] << " on GPU. Relative difference: " << fabs(results[i] - results_xDOTy8[i]) / results[i] << std::endl;
    }

    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "And now compare the runtime of all implementations..." << std::endl;
    std::cout << "CPU......" << time_CPU << "s >> Speedup: " << 100-time_CPU/time_CPU*100 << "%" << std::endl;
    std::cout << "CUBAS...." << time_cublas << "s >> Speedup: " << 100-time_cublas/time_CPU*100 << "%" << std::endl;
    std::cout << "xDOTy..." << time_xDOTy << "s >> Speedup: " << 100-time_xDOTy/time_CPU*100 << "%" << std::endl;
    std::cout << "xDOTy8..." << time_xDOTy8 << "s >> Speedup: " << 100-time_xDOTy8/time_CPU*100 << "%" << std::endl;

    
    
    //
    // Clean up:
    //
    std::cout << "Cleaning up..." << std::endl;
    free(x);
    cudaFree(cuda_x);

    for (size_t i=0; i<K; ++i) {
      free(y[i]);
      cudaFree(cuda_y[i]);
    }
    free(y);
    free(cuda_y);

    free(results);
    free(results2);
    free(results_xDOTy8);
 
    cublasDestroy(h);
    return 0;
}

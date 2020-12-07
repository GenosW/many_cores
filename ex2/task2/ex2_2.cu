#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"
#include <cmath>

#define BLOCK_SIZE 256

__global__ void initKernel1(double* arr, const double value, const size_t N)
{
  const int stride = blockDim.x * gridDim.x;
  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  for(; tid < N; tid += stride) 
  {
    arr[tid] = value;
  }
}

__global__ void initKernel2(double* arr, double* arr2, const size_t N)
{
  const int stride = blockDim.x * gridDim.x;
  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  for(; tid < N; tid += stride) 
  {
    arr[tid] = tid;
    arr2[tid] = N - 1 - tid;
  }
}

__global__ void initKernel3(double* arr, double* arr2, double* arr3, const size_t N)
{
  const int stride = blockDim.x * gridDim.x;
  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  for(; tid < N; tid += stride) 
  {
    arr[tid] = tid;
    arr2[tid] = N - 1 - tid;
    arr3[tid] = 0;
  }
}

__global__ void addKernel(double* x, double* y, double* res, const size_t N)
{
  const int stride = blockDim.x * gridDim.x;
  int tid = threadIdx.x +  blockIdx.x * blockDim.x;

  for(; tid < N; tid += stride) 
  {
    res[tid] = x[tid] + y[tid];
    //res[tid] += 1;
  }
}

__global__ void dot_A_1(double* x, double* y, double* block_sums, const size_t N)
{
  uint tid = threadIdx.x + blockDim.x* blockIdx.x;
  uint stride = blockDim.x* gridDim.x;

  __shared__ double cache[BLOCK_SIZE];

  double tid_sum = 0.0;
  for (; tid < N; tid += stride)
  {
    tid_sum += x[tid] * y[tid];
  }
  cache[threadIdx.x] = tid_sum;

  __syncthreads();
  for (uint i = blockDim.x/2; i != 0; i /=2)
  {
    if (threadIdx.x < i) //lower half does smth, rest idles
    {
      cache[threadIdx.x] += cache[threadIdx.x + i]; //lower looks up by stride and sums up
    }
    __syncthreads();
  }

  if(threadIdx.x == 0) // cache[0] now contains block_sum
  {
    block_sums[blockIdx.x] = cache[0];
  }
  __syncthreads();
}

__global__ void dot_A_2(double* block_sums)
{
  int tid = threadIdx.x; // only one block, so this is fine!
  for (int i = blockDim.x / 2; i >= 1; i /=2) // same principal as above
  {
    if ( tid < i )
    {
      block_sums[tid] += block_sums[tid + i];
    }
    __syncthreads();
  }
}

__global__ void dot_Atomic(double* x, double* y, double* result, const size_t N)
{
  uint tid = threadIdx.x + blockDim.x* blockIdx.x ;
  uint stride = blockDim.x* gridDim.x ;

  __shared__ double cache[BLOCK_SIZE];

  double tid_sum = 0.0;
  for (; tid < N; tid += stride)
  {
    tid_sum += x[tid] * y[tid];
  }
  tid = threadIdx.x;
  cache[tid] = tid_sum;

  __syncthreads();
  for (uint i = blockDim.x/2; i != 0; i /=2)
  {
    __syncthreads();
    if (tid < i) //lower half does smth, rest idles
    {
      cache[tid] += cache[tid + i ]; //lower looks up by stride and sums up
    }
  }

  if(tid == 0) // cache[0] now contains block_sum
  {
    atomicAdd(result, cache[0]);
  }
}

int main(void)
{
  const int num_tests = 10;
  int tests_done = num_tests;
  const int block_size = BLOCK_SIZE;
  const int mode = 1;
  const int option = 2;

  std::cout << "N;blocks;block_size;tests_done;total_time;time_per_test;check" << std::endl;
  for (size_t N = 256; N <= 1000000000; N*=10)
  {
    std::cout << N << std::endl;
    //int blocks = (int)(N+block_size-1)/block_size;
    int blocks = block_size;
    double result = 0.0;
    double result_true = N;
    double* h_block_sums = (double *)malloc(sizeof(double) * blocks);
    double* presult = (double *)malloc(sizeof(double));
    presult = &result;    
    double* pnull = (double *)malloc(sizeof(double));
    *pnull = 0.0;
    double *d_x, *d_y, *d_block_sums, *d_result;

    cudaMalloc(&d_result, sizeof(double));
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));
    cudaMalloc(&d_block_sums, blocks*sizeof(double));
    cudaDeviceSynchronize();

    int i = 0;
    Timer timer;
    double total_time = 0.0;

    initKernel1<<<blocks, block_size>>>(d_block_sums, 0.0, block_size);
    initKernel1<<<blocks, block_size>>>(d_x, 1.0, N);
    initKernel1<<<blocks, block_size>>>(d_y, 1.0, N);
    //initKernel2<<<blocks, block_size>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    timer.reset();
    for (i = 0; i<num_tests; i++) 
    {
      tests_done = i+1;
      if (option == 1)
      {
        dot_A_1<<<blocks, block_size>>>(d_x, d_y, d_block_sums,  N);
        dot_A_2<<<1, block_size>>>(d_block_sums);
        cudaDeviceSynchronize();
      }
      if (option == 2)
      {
        dot_A_1<<<blocks, block_size>>>(d_x, d_y, d_block_sums,  N);
        cudaMemcpy(h_block_sums, d_block_sums, blocks*sizeof(double), cudaMemcpyDeviceToHost);
        //std::cout << h_block_sums[0] << " =? " << h_block_sums[blocks-1] << std::endl;
        cudaDeviceSynchronize();
        result = 0.0;
        for (int j = 0; j < blocks; j+=1)
        {
          result += h_block_sums[j];
        }
      }
      if (option == 3)
      {
        cudaMemcpy(d_result, pnull, sizeof(double), cudaMemcpyHostToDevice);
        dot_Atomic<<<blocks, block_size>>>(d_x, d_y, d_result, N);
      }

      //std::cout << "(" << i+1 << ") Elapsed: " << runtime << " s" << std::endl;
    }
    total_time = timer.get();
    size_t check;
    if (option == 1)
    {
      cudaMemcpy(presult, d_block_sums, sizeof(double), cudaMemcpyDeviceToHost);
      check = result - result_true;
    }
    if (option == 2)
    {
      check = result - result_true;
    }
    if (option == 3)
    {
      cudaMemcpy(presult, d_result, sizeof(double), cudaMemcpyDeviceToHost);
      check = result - result_true;
    }
    
    if (mode == 0)
    { 
      std::cout << std::endl << "Results after " << tests_done << " tests:" << std::endl;
      std::cout << "Total runtime: " << total_time << std::endl;
      std::cout << "Average runtime; " << total_time/tests_done << std::endl;
      std::cout << "Check: " << result << " ?= " << result_true << std::endl;
      std::cout << "\n\n";
    }
    if (mode == 1)
    {
      std::cout << N << ";" 
      << blocks << ";" 
      << block_size << ";"
      << tests_done << ";" 
      << total_time << ";" 
      << total_time/tests_done << ";" 
      << check << std::endl;
    }
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_block_sums);
    cudaFree(d_result);
    // free(presult);
    // free(pnull);
    free(h_block_sums);
    if (N==256) 
    {
      N=100;
    }
  }
  cudaDeviceSynchronize();
  return EXIT_SUCCESS;
}

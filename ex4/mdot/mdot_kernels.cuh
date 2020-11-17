#pragma once
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

  __shared__ double cache1[BLOCK_SIZE];
  __shared__ double cache2[BLOCK_SIZE];
  __shared__ double cache3[BLOCK_SIZE];
  __shared__ double cache4[BLOCK_SIZE];
  __shared__ double cache5[BLOCK_SIZE];
  __shared__ double cache6[BLOCK_SIZE];
  __shared__ double cache7[BLOCK_SIZE];
  __shared__ double cache8[BLOCK_SIZE];

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
  cache1[tid] = tid_sum1; 
  cache2[tid] = tid_sum2;
  cache3[tid] = tid_sum3; 
  cache4[tid] = tid_sum4;
  cache5[tid] = tid_sum5; 
  cache6[tid] = tid_sum6;
  cache7[tid] = tid_sum7; 
  cache8[tid] = tid_sum8;

  __syncthreads();
  for (size_t i = blockDim.x/2; i != 0; i /=2)
  {
    __syncthreads();
    if (tid < i) { //lower half
      cache1[tid] += cache1[tid + i];
      cache2[tid] += cache2[tid + i];
      cache3[tid] += cache3[tid + i];
      cache4[tid] += cache4[tid + i];
    // }
    // else if (tid < i*2){
      cache5[tid] += cache5[tid + i];
      cache6[tid] += cache6[tid + i];
      cache7[tid] += cache7[tid + i];
      cache8[tid] += cache8[tid + i];
    }

  }

  if (tid==0) // cache0 now contains block_sum
  {
    atomicAdd(z, cache1[0]);
    atomicAdd(z+1, cache2[0]);
    atomicAdd(z+2, cache3[0]);
    atomicAdd(z+3, cache4[0]);
    atomicAdd(z+4, cache5[0]);
    atomicAdd(z+5, cache6[0]);
    atomicAdd(z+6, cache7[0]);
    atomicAdd(z+7, cache8[0]);
  }
}
#include <stdio.h>
#include <iostream>
#include "timer.hpp"

#define PRINT_ONLY 10


__global__
void transpose(double *A, int N)
{
  int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  int row_idx = t_idx / N;
  int col_idx = t_idx % N;
  
  if (row_idx < N && col_idx < N
    && col_idx > row_idx
    && t_idx < N*N) 
  {
    double tmp = A[row_idx * N + col_idx];
    A[row_idx * N + col_idx] = A[col_idx * N + row_idx];
    A[col_idx * N + row_idx] = tmp;
  }
}

__global__
void transpose_original(double *A, int N)
{
  int t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  int row_idx = t_idx / N;
  int col_idx = t_idx % N;
  
  if (row_idx < N && col_idx < N) 
    A[row_idx * N + col_idx] = A[col_idx * N + row_idx];
}


void print_A(double *A, int N)
{
  int to_print = min(N, PRINT_ONLY);
  for (int i = 0; i < to_print; i++) {
    for (int j = 0; j < to_print; ++j) {
      std::cout << A[i * N + j] << ", ";
    }
    std::cout << std::endl;
  }
}

int main(void)
{
  int N = 100;

  double *A, *cuda_A;
  Timer timer;

  // Allocate host memory and initialize
  A = (double*)malloc(N*N*sizeof(double));
  
  for (int i = 0; i < N*N; i++) {
    A[i] = i;
  }

  print_A(A, N);


  // Allocate device memory and copy host data over
  cudaMalloc(&cuda_A, N*N*sizeof(double)); 

  // copy data over
  cudaMemcpy(cuda_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);

  // wait for previous operations to finish, then start timings
  cudaDeviceSynchronize();
  timer.reset();

  // Perform the transpose operation
  transpose<<<(N+255)/256, 256>>>(cuda_A, N);

  // wait for kernel to finish, then print elapsed time
  cudaDeviceSynchronize();
  double elapsed = timer.get();
  std::cout << std::endl << "Time for transpose: " << elapsed << std::endl;
  std::cout << "Effective bandwidth: " << (2*N*N*sizeof(double)) / elapsed * 1e-9 << " GB/sec" << std::endl;
  std::cout << std::endl;

  // copy data back (implicit synchronization point)
  cudaMemcpy(A, cuda_A, N*N*sizeof(double), cudaMemcpyDeviceToHost);

  print_A(A, N);

  // My friend was a bit sloppy and forgot these two lines...
  free(A);
  cudaFree(cuda_A);
  // Well, happens to the best!

  cudaDeviceReset();  // for CUDA leak checker to work

  return EXIT_SUCCESS;
}


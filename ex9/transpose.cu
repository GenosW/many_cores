#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include "timer.hpp"

#define PRINT_ONLY 10

#define TILE_DIM 16
#define BLOCK_ROWS 4


// ------------- HELPERS ------------- //

void print_analysis(double elapsed, uint N)
{
  const uint mem_size = N*N*sizeof(double); // of A
  std::cout << std::endl << "Time for transpose: " << elapsed << std::endl;
  std::cout << "Effective bandwidth: " << (2*mem_size) / elapsed * 1e-9 << " GB/sec" << std::endl;
  std::cout << std::endl;
}

void print_A(double *A, uint N)
{
  const uint w = 1 + (int)std::log10(N*N);
  // std::cout << "width: " << w << std::endl;
  // << std::setfill(' ') 
  for (uint i = 0; i < N; i++) {
    for (uint j = 0; j < N; ++j) {
      std::cout << std::setw(w) << A[i * N + j] << ", ";
    }
    std::cout << std::endl;
  }
}

// ------------- KERNELS ------------- //

__global__
void transpose(double *A, uint N)
{
  uint t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  uint row_idx = t_idx / N;
  uint col_idx = t_idx % N;
  
  if (row_idx < N && col_idx < N
    && col_idx < row_idx
    && t_idx < N*N) 
  {
    double tmp = A[row_idx * N + col_idx];
    A[row_idx * N + col_idx] = A[col_idx * N + row_idx];
    A[col_idx * N + row_idx] = tmp;
  }
}

__global__
void transpose_original(double *A, uint N)
{
  uint t_idx = blockIdx.x*blockDim.x + threadIdx.x;
  uint row_idx = t_idx / N;
  uint col_idx = t_idx % N;
  
  if (row_idx < N && col_idx < N) 
    A[row_idx * N + col_idx] = A[col_idx * N + row_idx];
}

/** blocked transpose
 *
 * I used the given reference as a base and modified it to be in place
 * by adding a second shared memory block.
 *
 * Reference:
 * - https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 */
__global__
void transpose_blockwise(double *A, uint N)
{
  // Operating on a per Thread-Block basis
  __shared__ double tile_ur[TILE_DIM][TILE_DIM];
  __shared__ double tile_ll[TILE_DIM][TILE_DIM];

  uint x = blockIdx.x * TILE_DIM;
  uint y = blockIdx.y * TILE_DIM;
  uint tid = threadIdx.x;

  bool is_ondiag = (x == y) ? true : false;

  if (blockIdx.y >= blockIdx.x) {
    if (is_ondiag)
    {
      // copy to shared
      for (uint j = 0; j < TILE_DIM; j += 1) 
      {
        tile_ur[j][tid] = A[(y + tid)*N + j + x];
      }

      // sync
      __syncthreads();

      // copy from shared

      for (uint j = 0; j < TILE_DIM; j += 1)
      {
        A[(y + tid)*N + j + x] = tile_ur[threadIdx.x][j];
      }
    }
    else if (!is_ondiag)
    {
      // copy to shared
      for (uint j = 0; j < TILE_DIM; j += 1) 
      {
        tile_ur[j][tid] = A[(y + tid)*N + j + x];
        tile_ll[j][tid] = A[(x + tid)*N + j + y];
      }

      // sync
      __syncthreads();

      // copy from shared

      for (uint j = 0; j < TILE_DIM; j += 1)
      {
        A[(y + tid)*N + j + x] = tile_ll[threadIdx.x][j];
        A[(x + tid)*N + j + y] = tile_ur[threadIdx.x][j];
      }
    }
  }
}

__global__ void iptransposeCoalesced(double *data)
{
  __shared__ double tile_s[TILE_DIM][TILE_DIM+1];
  __shared__ double tile_d[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  if (blockIdx.y>blockIdx.x) { // handle off-diagonal case
    int dx = blockIdx.y * TILE_DIM + threadIdx.x;
    int dy = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      tile_s[threadIdx.y+j][threadIdx.x] = data[(y+j)*width + x];
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      tile_d[threadIdx.y+j][threadIdx.x] = data[(dy+j)*width + dx];
    __syncthreads();
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      data[(dy+j)*width + dx] = tile_s[threadIdx.x][threadIdx.y + j];
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      data[(y+j)*width + x] = tile_d[threadIdx.x][threadIdx.y + j];
  }

  else if (blockIdx.y==blockIdx.x){ // handle on-diagonal case
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      tile_s[threadIdx.y+j][threadIdx.x] = data[(y+j)*width + x];
    __syncthreads();
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
      data[(y+j)*width + x] = tile_s[threadIdx.x][threadIdx.y + j];
  }
}

int main(void)
{
  //
  // -------------- SETUP -------------- //
  //
  uint N = 64;
  //           (x        , y         , z)
  dim3 dimGrid(N/TILE_DIM, N/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, 1, 1);

  dim3 dimGrid_nv(N/TILE_DIM, N/TILE_DIM, 1);
  dim3 dimBlock_nv(TILE_DIM, 8, 1);

  double *A, *A2, *A3;
  double *cuda_A;
  Timer timer;

  // Allocate host memory and initialize
  A = (double*)malloc(N*N*sizeof(double));
  A2 = (double*)malloc(N*N*sizeof(double));
  A3 = (double*)malloc(N*N*sizeof(double));
  
  for (uint i = 0; i < N*N; i++) {
    A[i] = i;
    A2[i] = i;
    A3[i] = i;
  }

  if (N <= 32) {
    print_A(A, N);
    // print_A(A2, N); // A2 looks the same...
  }

  // Allocate device memory and copy host data over
  cudaMalloc(&cuda_A, N*N*sizeof(double)); 

  //
  // -------------- BENCHMARKS -------------- //
  //

  // -------------- Naive Transpose -------------- //
  // copy data over
  cudaMemcpy(cuda_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  // wait for previous operations to finish, then start timings
  cudaDeviceSynchronize();
  timer.reset();
  // Perform the transpose operation
  //           # blocks  , threads per block
  transpose<<<(N*N+255)/256, 256>>>(cuda_A, N);

  // wait for kernel to finish, then print elapsed time
  cudaDeviceSynchronize();
  double elapsed = timer.get();
  // copy data back (implicit synchronization point)
  cudaMemcpy(A, cuda_A, N*N*sizeof(double), cudaMemcpyDeviceToHost);

  // -------------- Block Transpose -------------- //
  // copy data over
  cudaMemcpy(cuda_A, A2, N*N*sizeof(double), cudaMemcpyHostToDevice);
  // wait for previous operations to finish, then start timings
  cudaDeviceSynchronize();
  timer.reset();
  // Perform the transpose operation
  //                    # blocks  , threads per block
  transpose_blockwise<<<dimGrid, dimBlock>>>(cuda_A, N);

  // wait for kernel to finish, then print elapsed time
  cudaDeviceSynchronize();
  double elapsed_block = timer.get();
  // copy data back (implicit synchronization point)
  cudaMemcpy(A2, cuda_A, N*N*sizeof(double), cudaMemcpyDeviceToHost);

  // -------------- NVIDIA Block Transpose -------------- //
  // copy data over
  cudaMemcpy(cuda_A, A3, N*N*sizeof(double), cudaMemcpyHostToDevice);
  // wait for previous operations to finish, then start timings
  cudaDeviceSynchronize();
  timer.reset();
  // Perform the transpose operation
  //                    # blocks  , threads per block
  iptransposeCoalesced<<<dimGrid_nv, dimBlock_nv>>>(cuda_A);

  // wait for kernel to finish, then print elapsed time
  cudaDeviceSynchronize();
  double elapsed_block_nv = timer.get();
  // copy data back (implicit synchronization point)
  cudaMemcpy(A3, cuda_A, N*N*sizeof(double), cudaMemcpyDeviceToHost);

  //
  // -------------- OUTPUT -------------- //
  //

  std::cout << "---- Naive Transpose ----" << std::endl;
  print_analysis(elapsed, N);
  std::cout << "---- Block Transpose ----" << std::endl;
  print_analysis(elapsed_block, N);
  std::cout << "---- NVIDIA Block Transpose ----" << std::endl;
  print_analysis(elapsed_block_nv, N);

  if (N <= 32) {
    std::cout << std::endl;
    std::cout << "---- Naive Transpose ----" << std::endl;
    print_A(A, N);
    std::cout << "---- Block Transpose ----" << std::endl;
    print_A(A2, N);
    std::cout << "---- NVIDIA Block Transpose ----" << std::endl;
    print_A(A3, N);
  }

  std::cout << "---- Parameters: ----" << std::endl;
  std::cout << "TILE_DIM: " << TILE_DIM << std::endl;
  std::cout << "dimGrid: (" << dimGrid.x << ", "<< dimGrid.y << ", "<< dimGrid.z << ")" << std::endl;
  std::cout << "dimBlock: (" << dimBlock.x << ", "<< dimBlock.y << ", "<< dimBlock.z << ")" << std::endl;
  std::cout << "BLOCK_ROWS: " << BLOCK_ROWS << std::endl;
  std::cout << "dimGrid_nv: (" << dimGrid_nv.x << ", "<< dimGrid_nv.y << ", "<< dimGrid_nv.z << ")" << std::endl;
  std::cout << "dimBlock_nv: (" << dimBlock_nv.x << ", "<< dimBlock_nv.y << ", "<< dimBlock_nv.z << ")" << std::endl;
  
  // My friend was a bit sloppy and forgot these two lines...
  free(A);
  free(A2);
  free(A3);
  cudaFree(cuda_A);
  // Well, happens to the best!

  cudaDeviceReset();  // for CUDA leak checker to work

  return EXIT_SUCCESS;
}


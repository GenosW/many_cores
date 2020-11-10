#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include "timer.hpp"

#define BLOCK_SIZE 256
#define GRID_SIZE 256
#define TESTS 5
#define K_LIM 64
#define N_DEF 100'000'000
#define hRES_INIT -1.
#define dRES_INIT 0.

template <typename T, typename sizeT>
__global__ void initKernel(T* arr, const sizeT N, const T val)
{
  const sizeT stride = blockDim.x * gridDim.x;
  sizeT tid = threadIdx.x +  blockIdx.x * blockDim.x;

  for(; tid < N; tid += stride) 
  {
    arr[tid] = val;
  }
}

template <typename T, typename sizeT>
__global__ void vectorAddStride(const T* x, const T* y, T* z, const sizeT N, const sizeT k) {
  const sizeT stride = blockDim.x * gridDim.x;

  for (sizeT tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N/stride; tid += stride) {
    z[tid*k] = x[tid*k] + y[tid*k];
  }
}

template <typename T, typename sizeT>
__global__ void vectorAddOffset(const T* x, const T* y, T* z, const sizeT N, const sizeT k) {
  const sizeT stride = blockDim.x * gridDim.x;

  for (sizeT tid = threadIdx.x +  blockIdx.x * blockDim.x; tid < N-k; tid += stride) {
    z[tid+k] = x[tid+k] + y[tid+k];
  }
}

template <typename T>
T median(std::vector<T>& vec)
{
  // modified taken from here: https://stackoverflow.com/questions/2114797/compute-median-of-values-stored-in-vector-c

  size_t size = vec.size();

  if (size == 0)
          return 0.;

  sort(vec.begin(), vec.end());

  size_t mid = size/2;

  return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}

int main(void)
{
  size_t N = N_DEF;
  std::string mode = "csv";
  std::vector<double> time(TESTS, -1.);
  std::vector<double> changed(TESTS, 0.);
  std::vector<double> h_res(N, hRES_INIT); // init to -1

  double *d_x, *d_y, *d_res;
  cudaMalloc(&d_x, N*sizeof(double));
  cudaMalloc(&d_y, N*sizeof(double));
  cudaMalloc(&d_res, N*sizeof(double));
  cudaDeviceSynchronize();
  initKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, N, 1.);
  initKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_y, N, 2.);
  cudaDeviceSynchronize();

  std::cout << " k | time (med) | changed # | N | Grid Size | Block Size " << std::endl;

  for (size_t k = 0; k < K_LIM; )
  {
    initKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_res, N, dRES_INIT); // init to 0
    cudaDeviceSynchronize();

    Timer timer;

    for (int i = 0; i < TESTS; ++i) 
    {
      timer.reset();

      vectorAddStride<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y, d_res, N, k);
      cudaDeviceSynchronize();

      double rt = timer.get();

      /*cudaMemcpy(h_res.data(), d_res, N*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();*/

      time[i] = rt;

      /*size_t cnt = 0;
      for (size_t j = 0; j < N; ++j) {
        if (h_res[j] == 0.) {
          std::cout << "Dev->Host copy failed: j=" << j << std::endl;
          continue;
        }
        if (h_res[j] == 3.) cnt++;
      }
      changed[i] = cnt;*/
    }

    std::cout << std::setw(2) << k << " | "
      << std::setw(10) << std::scientific << median(time) << " | "
      << std::setw(9) << std::scientific << median(changed) << " | "
      << std::setw(3) << std::scientific << N << " | "
      << std::setw(9) << std::scientific << GRID_SIZE << " | "
      << std::setw(10) << std::scientific << BLOCK_SIZE << " | "
      << std::endl;
      
  }
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_res);
  cudaDeviceSynchronize();

  return EXIT_SUCCESS;
}

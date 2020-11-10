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


__global__ void initKernel(double* arr, const size_t N, const double val)
{
  const size_t stride = blockDim.x * gridDim.x;
  size_t tid = threadIdx.x +  blockIdx.x * blockDim.x;

  for(; tid < N; tid += stride) 
    arr[tid] = val;
}

/* Task 1 a) */
__global__ void vectorAddOffset(const double* x, const double* y, double* z, const size_t N, const size_t k) {
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N-k; tid += stride)
    z[tid+k] = x[tid+k] + y[tid+k];
}

/* Task 1 b) */
__global__ void vectorAddStride(const double* x, const double* y, double* z, const size_t N, const size_t k) {
  const size_t stride = blockDim.x * gridDim.x;

  if (k != 0){
    for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N/k; tid += stride)
      z[tid*k] = x[tid*k] + y[tid*k];
  }
  else
  {
      z[0] = x[0] + y[0];
  }
}

double median(std::vector<double>& vec)
{
  // modified taken from here: https://stackoverflow.com/questions/2114797/compute-median-of-values-stored-in-vector-c

  size_t size = vec.size();

  if (size == 0)
          return 0.;

  sort(vec.begin(), vec.end());

  size_t mid = size/2;

  return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}

void printArray(double* x, const size_t N, const size_t num){
  std::string sep = " > ";
  for (int i=0; i < num; i++) std::cout << x[i] << sep;
  std::cout << std::endl;
  for (int i=0; i < num; i++) std::cout << x[N-1-i] << sep;
  std::cout << std::endl;
}

int main(void)
{
  size_t N = N_DEF;
  std::string mode = "csv";
  std::vector<double> time(TESTS, -1.);
  //std::vector<double> h_res(N, hRES_INIT); // init to -1
  double* h_res = new double[N];
  double* x = new double[N]; 
  double* y = new double[N];
  for (int i=0; i < N; ++i) h_res[i] = hRES_INIT;

  double *d_x, *d_y, *d_res;
  cudaMalloc(&d_x, N*sizeof(double));
  cudaMalloc(&d_y, N*sizeof(double));
  cudaMalloc(&d_res, N*sizeof(double));
  cudaDeviceSynchronize();
  initKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, N, 1.);
  initKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_y, N, 2.);
  cudaDeviceSynchronize();

  std::string sep = ";";
  std::cout << "k" << sep
            << "time" << sep
            << "changed" << sep
            << "N" << sep
            << "Grid Size" << sep
            << "Block Size" << std::endl;
  
  initKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_res, N, dRES_INIT); // init to 0
  cudaDeviceSynchronize();

  for (size_t k = 0; k < K_LIM; ++k)
  {
    Timer timer;

    for (int i = 0; i < TESTS; ++i) 
    {
      timer.reset();

      vectorAddStride<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y, d_res, N, k);
      //vectorAddOffset<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y, d_res, N, k);
      cudaDeviceSynchronize();

      double rt = timer.get();

      time[i] = rt;
    }

    cudaMemcpy(h_res, d_res, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);

    //std::string sep = " | ";

    std::cout << k << sep
      << std::scientific << median(time) << sep
      << N << sep
      << GRID_SIZE << sep
      << BLOCK_SIZE
      << std::endl;  
  }
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "res" << std::endl;
  printArray(h_res, N, 4);
  std::cout << "x" << std::endl;
  printArray(x, N, 4);
  std::cout << "y" << std::endl;
  printArray(y, N, 4);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_res);
  cudaDeviceSynchronize();
  delete[] h_res;

  return EXIT_SUCCESS;
}

#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"

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
  int tid = threadIdx.x +  blockIdx . x * blockDim . x;

  for(; tid < N; tid += stride) 
  {
    res[tid] = x[tid] + y[tid];
    //res[tid] += 1;
  }
}

int main(void)
{

  double *d_x, *d_y, *d_res;

  std::vector<int> N_vec{ 128, 256, 1024, 4096, 8192, 32768, 65536, 131072, int(1e6), int(1e7), int(1e8)}};
  int num_tests = 10;
  int tests_done = num_tests;
  std::string mode = "readable";

  for (int & N: N_vec)
  {
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));
    cudaMalloc(&d_res, N*sizeof(double));
    cudaDeviceSynchronize();
    //if (N > 256) break; // Break clause for testing purposes!
    //int N = 1024;
    int i = 0;
    int block_size = 256;
    int blocks = (int)(N+255)/block_size;

    Timer timer;
    double total_time = 0.0;
    double runtime = 0.0, max_runtime = 0.0, min_runtime = 100.0;
    std::vector<double> x(N, 0);
    std::vector<double> y(N, 0);
    std::vector<double> res(N, 0);

    for (i = 0; i < N; ++i)
    {
      x[i] = i;
      y[i] = N - 1 - i;
    }

    //initKernel3<<<(N+255)/256, 256>>>(d_x, d_y, d_res, N);
    initKernel3<<<blocks, block_size>>>(d_x, d_y, d_res, N);
    cudaDeviceSynchronize();

    for (i = 0; i<num_tests; i++) 
    {
      tests_done = i+1;
      timer.reset();

      //sumVectors<<<blocks, block_size>>>(d_x, d_y, d_res, N);
      addKernel<<<blocks, block_size>>>(d_x, d_y, d_res, N);
      cudaDeviceSynchronize();

      runtime = timer.get();
      //std::cout << "(" << i+1 << ") Elapsed: " << runtime << " s" << std::endl;
      total_time += runtime;

      if (runtime > max_runtime) 
      {
        max_runtime = runtime;
      }
      if (runtime < min_runtime) 
      {
        min_runtime = runtime;
      }
      if (total_time > 1.) 
      {
        break;
      }
    }

    if (mode == "readable")
    { 
      cudaMemcpy(x.data(), d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(y.data(), d_y, N*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(res.data(), d_res, N*sizeof(double), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      std::cout << std::endl << "Results after " << tests_done << " tests:" << std::endl;
      std::cout << "Total runtime: " << total_time << std::endl;
      std::cout << "Average runtime; " << total_time/tests_done << std::endl;
      std::cout << "Maximum runtime: " << max_runtime << std::endl;
      std::cout << "Minimum runtime: " << min_runtime << std::endl;
      std::cout << "\n\n";
      for (int j=0; j < 4; j++) 
      {
        std::cout << j << ": " << x[j] << " | " << y[j] << " | " << res[j] << std::endl;
      }
      for (int j=0; j < 4; j++) 
      {
        int from_end = N - 1 - j;
        std::cout << from_end << ": " << x[from_end] << " | " << y[from_end] << " | " << res[from_end] << std::endl;
      }
      std::cout << "\n\n";
    }
    if (mode == "csv")
    {
      std::cout << tests_done << ";" << N << ";" << total_time << ";" << total_time/tests_done << ";" << max_runtime << ";" << min_runtime << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_res);
    cudaDeviceSynchronize();
  }

  return EXIT_SUCCESS;
}

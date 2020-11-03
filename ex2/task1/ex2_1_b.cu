#include <iostream>
#include <string>
#include <vector>
#include "timer.hpp"

__global__ void initKernel(double* arr, double* arr2, const size_t N)
{
  const int stride = blockDim.x * gridDim.x;
  int tid = threadIdx.x + stride;

  for(; tid < N; tid += stride) 
  {
    arr[tid] = tid - 1;
    arr2[tid] = N - 1 - tid;
  }
}

int main(void)
{
  std::vector<int> N_vec{ 256, 1024, 4096, 8192, 32768, 65536, 131072};

  int option = 1;
  int num_tests = 10;
  int tests_done = num_tests;
  std::string mode = "csv";

  std::cout << "tests_done;N;total_time;average_time;max_runtime;min_runtime;hosttime" << std::endl;
  for (int & N: N_vec)
  {
    //if (N != 256) break; // Break clause for testing purposes!
    //int N = 1024;
    int i = 0;

    Timer timer;
    double total_time = 0.0;
    double runtime = 0.0, max_runtime = 0.0, min_runtime = 100.0, hosttime = -1.;
    double *d_x, *d_y;
    std::vector<double> x(N, 0);
    std::vector<double> y(N, 0);

    timer.reset();
    for (i = 0; i < N; ++i)
    {
      x[i] = i;
      y[i] = N - 1 - i;
    }
    hosttime = timer.get();

    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));

    for (i = 0; i<num_tests; i++) 
    {
      timer.reset();
      if (option == 1)
      {
        cudaMemcpy(d_x, x.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y.data(), N*sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        tests_done = i+1;
      }
      else if (option == 2)
      {
        if (N < 1025)
        {
          double* px = &x[0];
          double* py = &y[0];
          for(int k=0; k < N; k++)
          {
            cudaMemcpy(d_x+k, px+k, sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y+k, py+k, sizeof(double), cudaMemcpyHostToDevice);
          }
          cudaDeviceSynchronize();
          tests_done = i+1;
        }
      }
      else if (option == 3)
      {
        initKernel<<<(N+255)/256, 256>>>(d_x, d_y, N);
        tests_done = i+1;
      }
      else
      {
        std::cout << "No valid option selected!" << std::endl;
      }

      runtime = timer.get();
      //std::cout << "(" << i+1 << ") Elapsed: " << runtime << " s" << std::endl;
      total_time += runtime;

      cudaFree(d_x);
      cudaFree(d_y);
      cudaDeviceSynchronize();


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
      std::cout << std::endl << "Results after " << tests_done << " tests:" << std::endl;
      std::cout << "Total runtime: " << total_time << std::endl;
      std::cout << "Average runtime; " << total_time/tests_done << std::endl;
      std::cout << "Maximum runtime: " << max_runtime << std::endl;
      std::cout << "Minimum runtime: " << min_runtime << std::endl;
      std::cout << "\nTime needed for vector init on host: " << hosttime << std::endl;
      std::cout << "\n\n";
      for (int j=0; j < 4; j++) 
      {
        std::cout << j << ": " << x[j] << " | " << y[j] << std::endl;
      }
    }
    if (mode == "csv")
    {
      std::cout << tests_done << ";" << N << ";" << total_time << ";" << total_time/tests_done << ";" << max_runtime << ";" << min_runtime << ";" << hosttime << std::endl;
    }
    
  }
  return EXIT_SUCCESS;
}

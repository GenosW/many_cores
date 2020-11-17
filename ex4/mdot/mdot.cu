#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "timer.hpp"
#include "mdot_kernels.cuh"

int main(void)
{
    const size_t K = 32;
    Timer timer;

    std::vector<double> times_CPU; 
    std::vector<double> times_cublas; 
    std::vector<double> times_xDOTy; 
    std::vector<double> times_xDOTy8;
    std::vector<int> vec_ks;
    std::vector<int> vec_Ns;

    for (size_t N = 1000; N <= 1000000; N*=10) {
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
        results2[i] = 0;
      } 

      //
      // Copy data to GPU
      //
      std::cout << "Copying data to GPU..." << std::endl;
      cudaMemcpy(cuda_x, x, sizeof(double)*N, cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_results, results_xDOTy8, sizeof(double)*K, cudaMemcpyHostToDevice);
      for (size_t i=0; i<K; ++i) {
        cudaMemcpy(cuda_y[i], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
      }

      for (int k = 8; k <= K; k+=8) {
        
        //
        // Reference calculation on CPU:
        //
        timer.reset();
        for (size_t i=0; i<k; ++i) {
          results[i] = 0;
          for (size_t j=0; j<N; ++j) {
            results[i] += x[j] * y[i][j];
          }
        }    
        double time_CPU = timer.get();

        //
        // Let CUBLAS do the work:
        //
        std::cout << "Running dot products with CUBLAS..." << std::endl;
        timer.reset();
        for (size_t i=0; i<k; ++i) {
          cublasDdot(h, N, cuda_x, 1, cuda_y[i], 1, results2 + i);
        }
        double time_cublas = timer.get();

        //
        // Let xDOTy do the work:
        //
        std::cout << "Running dot products with custom xDOTy8..." << std::endl;
        timer.reset();
        for (size_t i=0; i<k; ++i) {
          xDOTy<<<GRID_SIZE, BLOCK_SIZE>>>(N, 
                cuda_x, cuda_y[i],
                cuda_results+i);
        }
        cudaMemcpy(results_xDOTy, cuda_results, sizeof(double)*k, cudaMemcpyDeviceToHost);
        double time_xDOTy = timer.get();

        //
        // Let xDOTy8 do the work:
        //
        cudaMemcpy(cuda_results, results_xDOTy8, sizeof(double)*k, cudaMemcpyHostToDevice);
        std::cout << "Running dot products with custom xDOTy8..." << std::endl;
        timer.reset();
        for (size_t i=0; i<(int)k/8; ++i) {
          int batch_offset = (i*8);
          xDOTy8<<<GRID_SIZE, BLOCK_SIZE>>>(N, 
                cuda_x, 
                cuda_y[batch_offset], cuda_y[batch_offset+1],
                cuda_y[batch_offset+2], cuda_y[batch_offset+3],
                cuda_y[batch_offset+4], cuda_y[batch_offset+5],
                cuda_y[batch_offset+6], cuda_y[batch_offset+7],
                cuda_results+batch_offset);
        }
        cudaMemcpy(results_xDOTy8, cuda_results, sizeof(double)*k, cudaMemcpyDeviceToHost);
        double time_xDOTy8 = timer.get();

        //
        // Compare results
        //
        if (k==8) {
          std::cout << "------------------------------------------------------------" << std::endl;
          std::cout << "Copying results back to host..." << std::endl;
          for (size_t i=0; i<k; ++i) {
            std::cout << results[i] << " on CPU, " << results2[i] << " on GPU. Relative difference: " << fabs(results[i] - results2[i]) / results[i] << std::endl;
          }std::cout << "------------------------------------------------------------" << std::endl;
          std::cout << "Now to compare the custom kernel xDOTy to CPU..." << std::endl;
          for (size_t i=0; i<k; ++i) {
            std::cout << results[i] << " on CPU, " << results_xDOTy[i] << " on GPU. Relative difference: " << fabs(results[i] - results_xDOTy[i]) / results[i] << std::endl;
          }
          std::cout << "------------------------------------------------------------" << std::endl;
          std::cout << "Now to compare the custom kernel xDOTy8 to CPU..." << std::endl;
          for (size_t i=0; i<k; ++i) {
            std::cout << results[i] << " on CPU, " << results_xDOTy8[i] << " on GPU. Relative difference: " << fabs(results[i] - results_xDOTy8[i]) / results[i] << std::endl;
          }
        }

        bool in_percent = false;
        auto speedup = [ref_time=time_CPU, in_percent] (double comp_time) -> double { return (in_percent) ? (ref_time/comp_time)*100 : ref_time/comp_time;};
        auto time_in_ms = [] (double time) -> double { return time*1e-3;};
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "And now compare the runtime of all implementations..." << std::endl;
        std::string s_unit = (in_percent) ? "%" : "";
        std::string t_unit = "ms";
        std::cout << "CPU.........." << time_in_ms(time_CPU) << t_unit << std::endl;
        std::cout << "CUBLAS..." << time_in_ms(time_cublas) << t_unit << " >> Speedup: " << speedup(time_cublas) << s_unit << std::endl;
        std::cout << "xDOTy......" << time_in_ms(time_xDOTy) << t_unit << " >> Speedup: " << speedup(time_xDOTy) << s_unit << std::endl;
        std::cout << "xDOTy8...." << time_in_ms(time_xDOTy8) << t_unit << " >> Speedup: " << speedup(time_xDOTy8) << s_unit << std::endl;

        times_CPU.push_back(time_CPU); 
        times_cublas.push_back(time_cublas); 
        times_xDOTy.push_back(time_xDOTy); 
        times_xDOTy8.push_back(time_xDOTy8);
        vec_ks.push_back(k);
        vec_Ns.push_back(N);
      }
      
      
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
    }

    std::cout << "--------------------------CSV--------------------------------" << std::endl;
      std::string sep = ";";
      std::cout << "N" << sep << "k" << sep << "time_CPU" << sep << "time_cublas" << sep << "time_xDOTy" << sep << "time_xDOTy8\n";
      for (int i = 0; i < vec_ks.size(); ++i ) {
        std::cout << std::setprecision("scientific") << vec_Ns[i] << sep 
          << vec_ks[i] << sep 
          << times_CPU[i] << sep 
          << times_cublas[i] << sep 
          << times_xDOTy[i] << sep 
          << times_xDOTy8[i] << "\n";
      }
      std::cout << std::endl;

    return 0;
}

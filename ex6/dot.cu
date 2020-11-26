#include "timer.hpp"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#define BLOCK_SIZE 128
#define GRID_SIZE 128
#define SEP ";"
#define CSV_NAME "data_ph.csv"

#define DEBUG
#ifndef DEBUG
  #define CSV
#endif

template <typename T>
void printContainer(T container, const int size) {
  std::cout << container[0];
  for (int i = 1; i < size; ++i) 
    std::cout << " | " << container[i] ;
  std::cout << std::endl;
}

template <typename T>
void printContainer(T container, const int size, const int only) {
  std::cout << container[0];
  for (int i = 1; i < only; ++i) 
      std::cout  << " | " << container[i];
  std::cout << " | ...";
  for (int i = size - only; i < size; ++i) 
    std::cout  << " | " << container[i];
  std::cout << std::endl;
}

void printResults(double* results, std::vector<std::string> names, int size){
  std::cout << "Results:" << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cout << names[i] << " : " << results[i] << std::endl;
  }
}

/** atomicMax for double
 * 
 * References:
 * (1) https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicmax
 * (2) https://www.micc.unifi.it/bertini/download/gpu-programming-basics/2017/gpu_cuda_5.pdf
 * (3) https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
 */
__device__ void atomicMax(double* address, double val){    
  if (val > *address) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address; 
    unsigned long long int old = *address_as_ull, assumed;
    do  {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
      // atomicCAS returns the value that is stored in address AFTER the CAS
      // atomicCAS(a, b, c) --> return c
      //
    } while (assumed != old || __double_as_longlong(val) > old);
  }
}

/** atomicMin for double
 */
__device__ void atomicMin(double* address, double val){    
  if (val < *address) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address; 
    unsigned long long int old = *address_as_ull, assumed;
    do  {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
      // atomicCAS returns the value that is stored in address AFTER the CAS
      // atomicCAS(a, b, c) --> return a
      // atomicCAS(a, a, c) --> return c
    } while (assumed != old || __double_as_longlong(val) < old);
  }
}

/** scalar = x DOT y
 */
__global__ void xDOTy(const int N, double *x, double *y, double *scalar) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int stride = blockDim.x * gridDim.x;

  __shared__ double cache[BLOCK_SIZE];

  double tid_sum = 0.0;
  for (; tid < N; tid += stride) {
    double tmp_x = x[tid];
    tid_sum += tmp_x * y[tid];
  }
  tid = threadIdx.x;
  cache[tid] = tid_sum;

  __syncthreads();
  for (int i = blockDim.x / 2; i != 0; i /= 2) {
    __syncthreads();
    if (tid < i)                    // lower half does smth, rest idles
      cache[tid] += cache[tid + i]; // lower looks up by stride and sums up
  }

  if (tid == 0) // cache[0] now contains block_sum
  {
    atomicAdd(scalar, cache[0]);
  }
}

/** analyze_x_shared
 * 
 * result[0] = sum;
 * result[1] = abs_sum;
 * result[2] = sqr_sum;
 * result[3] = mod_max;
 * result[4] = min;
 * result[5] = max;
 * result[6] = z_entries;
 */
template <int block_size=BLOCK_SIZE>
__global__ void analyze_x_shared(const int N, double *x, double *results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x; // global tid
  const int stride = blockDim.x * gridDim.x;

  __shared__ double cache[7][block_size];

  double sum = 0.0, abs_sum = 0.0, sqr_sum = 0.0;
  // double mod_max = 0.0;
  double max = x[tid];
  double min = max;
  int z_entries = 0;
  for (; tid < N; tid += stride) {
    double value = x[tid];
    sum += value;
    abs_sum += std::abs(value);
    sqr_sum += value*value;

    // mod_max = (std::abs(value) > mod_max)? value : mod_max;
    min = (value < min)? value : min; 
    max = (value > max)? value : max;
    z_entries += (value)? 0 : 1;
  }
  tid = threadIdx.x; // block tid 
  cache[0][tid] = sum;
  cache[1][tid] = abs_sum;
  cache[2][tid] = sqr_sum;
  cache[3][tid] = (std::abs(min) > std::abs(max)) ? std::abs(min) : std::abs(max);
  cache[4][tid] = min;
  cache[5][tid] = max;
  cache[6][tid] = z_entries;

  __syncthreads();
  for (int i = blockDim.x / 2; i != 0; i /= 2) {
    __syncthreads();
    if (tid < i) { // lower half does smth, rest idles
      // sums
      cache[0][tid] += cache[0][tid + i]; 
      cache[1][tid] += cache[1][tid + i]; 
      cache[2][tid] += cache[2][tid + i]; 
      // min/max values
      cache[3][tid] = (cache[3][tid + i] > cache[3][tid])? cache[3][tid + i] : cache[3][tid]; // already all values are std::abs(...)
      cache[4][tid] = (cache[4][tid + i] < cache[4][tid])? cache[4][tid + i] : cache[4][tid]; 
      cache[5][tid] = (cache[5][tid + i] > cache[5][tid])? cache[5][tid + i] : cache[5][tid]; 

      // "sum"
      cache[6][tid] += cache[6][tid + i]; 
    }
  }

  if (tid == 0) // cache[0] now contains block_sum
  {
    atomicAdd(results, cache[0][0]);
    atomicAdd(results+1, cache[1][0]);
    atomicAdd(results+2, cache[2][0]);

    // Ideally...
    atomicMax(results+3, cache[3][0]);
    atomicMin(results+4, cache[4][0]);
    atomicMax(results+5, cache[5][0]);

    atomicAdd(results+6, cache[6][0]);
  }
}

/** analyze_x_shared
 * 
 * result[0] = sum;
 * result[1] = abs_sum;
 * result[2] = sqr_sum;
 * result[3] = mod_max;
 * result[4] = min;
 * result[5] = max;
 * result[6] = z_entries;
 */
 template <int block_size=BLOCK_SIZE>
 __global__ void analyze_x_warp(const int N, double *x, double *results) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x; // global tid
  const int stride = blockDim.x * gridDim.x;

  double sum = 0.0, abs_sum = 0.0, sqr_sum = 0.0;
  // double mod_max = 0.0;
  double max = x[tid];
  double min = max;
  int z_entries = 0;
  for (; tid < N; tid += stride) {
    double value = x[tid];
    sum += value;
    abs_sum += std::abs(value);
    sqr_sum += value*value;

    // mod_max = (std::abs(value) > mod_max)? value : mod_max;
    min = (value < min)? value : min; 
    max = (value > max)? value : max;
    z_entries += (value)? 0 : 1;
  }
  tid = threadIdx.x; // block tid 
  double mod_max = (std::abs(min) > std::abs(max)) ? std::abs(min) : std::abs(max);

  __syncthreads();
  // for (int i = blockDim.x / 2; i != 0; i /= 2) {
  //   //__syncthreads();
  //   sum += __shfl_down_sync(-1, sum, i);
  //   abs_sum += __shfl_down_sync(-1, abs_sum, i);
  //   sqr_sum += __shfl_down_sync(-1, sqr_sum, i);

  //   double tmp = __shfl_down_sync(-1, mod_max, i);
  //   mod_max = (tmp > mod_max) ? tmp : mod_max;
  //   tmp = __shfl_down_sync(-1, min, i);
  //   min = (tmp < min) ? tmp : min;
  //   tmp = __shfl_down_sync(-1, max, i);
  //   max = (tmp > max) ? tmp : max;

  //   z_entries += __shfl_down_sync(-1, z_entries, i);
  // }
  for (int i = blockDim.x / 2; i != 0; i /= 2) {
    //__syncthreads();
    sum += __shfl_xor_sync(-1, sum, i);
    abs_sum += __shfl_xor_sync(-1, abs_sum, i);
    sqr_sum += __shfl_xor_sync(-1, sqr_sum, i);

    double tmp = __shfl_xor_sync(-1, mod_max, i);
    mod_max = (tmp > mod_max) ? tmp : mod_max;
    tmp = __shfl_xor_sync(-1, min, i);
    min = (tmp < min) ? tmp : min;
    tmp = __shfl_xor_sync(-1, max, i);
    max = (tmp > max) ? tmp : max;

    z_entries += __shfl_xor_sync(-1, z_entries, i);
  }

  if (tid == 0) 
  {
    atomicAdd(results, sum);
    atomicAdd(results+1, abs_sum);
    atomicAdd(results+2, sqr_sum);

    atomicMax(results+3, mod_max);
    atomicMin(results+4, min);
    atomicMax(results+5, max);

    atomicAdd(results+6, z_entries);
  }
  // if (tid == 0) 
  // {
  //   atomicCAS(results, *results, sum);
  //   atomicCAS(results+1, *results+1, abs_sum);
  //   atomicCAS(results+2, *results+2, sqr_sum);

  //   atomicCAS(results+3, *results+3, mod_max);
  //   atomicCAS(results+4, *results+4, min);
  //   atomicCAS(results+5, *results+5, max);

  //   atomicCAS(results+6, *results+6, z_entries);
  // }
 }

int main(void) {
  Timer timer;

  // std::vector<double> times_CPU;
  // std::vector<double> times_cublas;
  // std::vector<double> times_analyze_x_shared;
  // std::vector<int> vec_Ks;
  std::vector<int> vec_Ns{7, 13, 23};

#ifdef CSV
  std::fstream csv;
  csv.open("ph_data.txt", std::fstream::out | std::fstream::trunc);
#endif

  for (int& N : vec_Ns) {
    //
    // Initialize CUBLAS:
    //
#ifdef DEBUG
    std::cout << "N = " << N << std::endl;
    std::cout << "Init CUBLAS..." << std::endl;
#endif
    cublasHandle_t h;
    cublasCreate(&h);

    //
    // allocate + init host memory:
    //
#ifdef DEBUG
    std::cout << "Allocating host arrays..." << std::endl;
#endif
    double *x = (double *)malloc(sizeof(double) * N);
    double *results = (double *)malloc(sizeof(double) * 7);
    double *results2 = (double *)malloc(sizeof(double) * 7);
    std::vector<std::string> names {"sum", "abs_sum", "sqr_sum", "mod_max", "min", "max", "zero_entries"};

    std::generate_n(x, N, [n = -N/2] () mutable { return n++; });
    std::random_shuffle(x, x+N);
    // I'm placing some values here by hand, so that certain results can be forced
    // --> to test: mod_max, min, max...
    x[0] = -1.1;
    x[N/5] = 0.;
    x[N/3] = -(N-1);
    x[2*N/3] = N;

    std::fill(results, results+7, 0.0);
    results[3] = x[0];
    results[4] = x[0];
    results[5] = x[0];
    std::copy(results, results+7, results2);
    /*result[0] = sum;
    * result[1] = abs_sum;
    * result[2] = sqr_sum;
    * result[3] = mod_max;
    * result[4] = min;
    * result[5] = max;
    * result[6] = z_entries;*/

    //
    // allocate device memory
    //
#ifdef DEBUG
    std::cout << "Initialized results containers: " << std::endl;
    printContainer(results, 7);
    printContainer(results2, 7);
    std::cout << "Allocating CUDA arrays..." << std::endl;
#endif
    double *cuda_x;
    double *cuda_results;
    double *cuda_results2;
    cudaMalloc(&cuda_x, sizeof(double) * N);
    cudaMalloc(&cuda_results, sizeof(double) * 7);
    cudaMalloc(&cuda_results2, sizeof(double) * 7);
    //
    // Copy data to GPU
    //
#ifdef DEBUG
    std::cout << "Copying data to GPU..." << std::endl;
#endif
    cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_results, results, sizeof(double) * 7, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_results2, results, sizeof(double) * 7, cudaMemcpyHostToDevice);

    //
    // Let CUBLAS do the work:
    //
    // std::cout << "Running dot products with CUBLAS..." << std::endl;
    // timer.reset();
    // for (size_t i = 0; i < k; ++i) {
    //   cublasDdot(h, N, cuda_x, 1, cuda_y[i], 1, results2 + i);
    // }
    // double time_cublas = timer.get();

    //
    // Let xDOTy do the work:
    //
#ifdef DEBUG
    std::cout << "Running dot products with custom analyze_x_shared..." << std::endl;
#endif
    cudaMemcpy(cuda_results, results, sizeof(double) * 7, cudaMemcpyHostToDevice);
    timer.reset();
    analyze_x_shared<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_x, cuda_results);
    cudaMemcpy(results, cuda_results, sizeof(double) * 7, cudaMemcpyDeviceToHost);
    double time_shared = timer.get();

#ifdef DEBUG
    std::cout << "Running dot products with custom analyze_x_warp..." << std::endl;
#endif
    cudaMemcpy(cuda_results2, results2, sizeof(double) * 7, cudaMemcpyHostToDevice);
    timer.reset();
    analyze_x_warp<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_x, cuda_results2);
    cudaMemcpy(results2, cuda_results2, sizeof(double) * 7, cudaMemcpyDeviceToHost);
    double time_warp = timer.get();

    //
    // Compare results
    //
#ifdef DEBUG
      std::cout << "DEBUG output:" << std::endl;
      std::cout << "x:" << std::endl;
      printContainer(x, N);
      std::cout << ">SHARED<" << std::endl;
      printResults(results, names, names.size());
      std::cout << "GPU shared runtime: " << time_shared << std::endl;

      std::cout << ">WARP<" << std::endl;
      printResults(results2, names, names.size());
      std::cout << "GPU warp runtime: " << time_warp << std::endl;
#endif

    //
    // Clean up:
    //
#ifdef DEBUG
    std::cout << "Cleaning up..." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
#endif
    free(x);
    free(results);
    free(results2);

    cudaFree(cuda_x);
    cudaFree(cuda_results);
    cudaFree(cuda_results2);

    cublasDestroy(h);
  }

#ifdef CSV
  std::cout << "--------------------------CSV--------------------------------"
            << std::endl;
  csv.open (csv_name, std::fstream::out | std::fstream::app);
  // csv << N << SEP << M << SEP
  //   << N*M << SEP
  //   << numberOfValues << SEP
  //   << time_assemble_cpu << SEP
  //   << time_assemble_gpu << SEP
  //   << runtime << SEP
  //   << iters << SEP
  //   << residual_norm << std::endl;
  // csv.close();
  std::string sep = ";";
  // to csv file
  csv << "N" << sep << "k" << sep << "time_CPU" << sep << "time_cublas" << sep
      << "time_xDOTy" << sep << "time_xDOTy8\n";
  for (int i = 0; i < vec_ks.size(); ++i) {
    csv << std::scientific << vec_Ns[i] << sep << vec_ks[i] << sep
        << times_CPU[i] << sep << times_cublas[i] << sep << times_xDOTy[i]
        << sep << times_xDOTy8[i] << "\n";
  }
  csv << std::endl;
  csv.close();

  std::cout << "\n\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex5/" + csv_name << std::endl;
#endif
  return EXIT_SUCCESS;
}

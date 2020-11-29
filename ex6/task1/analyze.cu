#include "timer.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

#define BLOCK_SIZE 256
#define GRID_SIZE 128
#define TESTS 5
// #define SEP ";"

// #define DEBUG
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

void printResults(double* results, double* ref, std::vector<std::string> names, int size){
  std::cout << "Results (with difference to reference):" << std::endl;
  for (int i = 0; i < size; ++i) {
    std::cout << names[i] << " = " << results[i] << " ||  " << ref[i] - results[i] << std::endl;
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

// ------------------ KERNELS ---------------

/** atomicMax for double
 * 
 * References:
 * (1) https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicmax
 * (2) https://www.micc.unifi.it/bertini/download/gpu-programming-basics/2017/gpu_cuda_5.pdf
 * (3) https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
 */
__device__ void atomicMax(double* address, double val){    
  unsigned long long int* address_as_ull = (unsigned long long int*) address; 
  unsigned long long int old = *address_as_ull, assumed;
  do  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
  } while (assumed != old);
}

/** atomicMin for double
 */
__device__ void atomicMin(double* address, double val){    
  unsigned long long int* address_as_ull = (unsigned long long int*) address; 
  unsigned long long int old = *address_as_ull, assumed;
  do  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
  } while (assumed != old);
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
// template <int block_size=BLOCK_SIZE>
__global__ void analyze_x_shared(const int N, double *x, double *results) {
  if (blockDim.x * blockIdx.x < N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x; // global tid
    const int stride = blockDim.x * gridDim.x;

    __shared__ double cache[7][BLOCK_SIZE];

    double sum = 0.0, abs_sum = 0.0, sqr_sum = 0.0;
    // double mod_max = 0.0;
    double max = x[0];
    double min = max;
    double z_entries = 0;
    for (; tid < N; tid += stride) {
      double value = x[tid];
      sum += value;
      abs_sum += std::abs(value);
      sqr_sum += value*value;

      // mod_max = (std::abs(value) > mod_max)? value : mod_max;
      min = fmin(value, min); 
      max = fmax(value, max);
      z_entries += (value)? 0 : 1;
    }
    tid = threadIdx.x; // block tid 
    cache[0][tid] = sum;
    cache[1][tid] = abs_sum;
    cache[2][tid] = sqr_sum;
    cache[3][tid] = fmax(std::abs(min), max);
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
        cache[3][tid] = fmax(cache[3][tid + i], cache[3][tid]); // already all values are std::abs(...)
        cache[4][tid] = fmin(cache[4][tid + i], cache[4][tid]); 
        cache[5][tid] = fmax(cache[5][tid + i], cache[5][tid]); 

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
__global__ void analyze_x_warp(const int N, double *x, double *results) {
  if (blockDim.x * blockIdx.x < N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x; // global tid
    const int stride = blockDim.x * gridDim.x;

    double sum = 0.0, abs_sum = 0.0, sqr_sum = 0.0;
    // double mod_max = 0.0;
    double max = x[0];
    double min = max;
    int z_entries = 0;
    for (; tid < N; tid += stride) {
      double value = x[tid];
      sum += value;
      abs_sum += std::abs(value);
      sqr_sum += value*value;

      min = fmin(value, min); 
      max = fmax(value, max);
      z_entries += (value)? 0 : 1;
    }
    tid = threadIdx.x; // block tid 
    double mod_max = fmax(std::abs(min), max);

    __syncthreads();
    for (int i = warpSize / 2; i != 0; i /= 2) {
      //__syncthreads();
      sum += __shfl_down_sync(0xffffffff, sum, i);
      abs_sum += __shfl_down_sync(0xffffffff, abs_sum, i);
      sqr_sum += __shfl_down_sync(0xffffffff, sqr_sum, i);

      double tmp = __shfl_down_sync(0xffffffff, mod_max, i);
      mod_max = fmax(tmp, mod_max);
      tmp = __shfl_down_sync(0xffffffff, min, i);
      min = fmin(tmp, min);
      tmp = __shfl_down_sync(0xffffffff, max, i);
      max = fmax(tmp, max) ;

      z_entries += __shfl_down_sync(0xffffffff, z_entries, i);
    }
    // for (int i = blockDim.x / 2; i != 0; i /= 2) {
    // for (int i = warpSize / 2; i != 0; i /= 2) {
    //   //__syncthreads();
    //   sum += __shfl_xor_sync(-1, sum, i);
    //   abs_sum += __shfl_xor_sync(-1, abs_sum, i);
    //   sqr_sum += __shfl_xor_sync(-1, sqr_sum, i);

    //   double tmp = __shfl_xor_sync(-1, mod_max, i);
    //   mod_max = (tmp > mod_max) ? tmp : mod_max;
    //   tmp = __shfl_xor_sync(-1, min, i);
    //   min = (tmp < min) ? tmp : min;
    //   tmp = __shfl_xor_sync(-1, max, i);
    //   max = (tmp > max) ? tmp : max;

    //   z_entries += __shfl_xor_sync(-1, z_entries, i);
    // }

    if (tid % warpSize == 0) // a block can consist of multiple warps
    {
      atomicAdd(results, sum);
      atomicAdd(results+1, abs_sum);
      atomicAdd(results+2, sqr_sum);

      atomicMax(results+3, mod_max);
      atomicMin(results+4, min);
      atomicMax(results+5, max);

      atomicAdd(results+6, z_entries);
    }
  }
}

template <typename T>
void toCSV(std::fstream& csv, T* array, int size) {
  csv << size;
  for (int i = 0; i < size; ++i) {
    csv << ";" << array[i];
  }
  csv << std::endl;
}

int main(void) {
  Timer timer;
  std::vector<int> vec_Ns{100, 1000, 10000,  100000, 1000000, 10000000, 100000000};
  // std::vector<int> vec_Ns{100, 1000};
  std::vector<double> times(TESTS, 0.);

#ifdef CSV
  std::fstream csv_times, csv_results, csv_results2, csv_results3, csv_results_ref;
  std::string csv_times_name = "ph_data.csv";
  std::string csv_results_name = "ph_results.csv";
  std::string csv_results2_name = "ph_results2.csv";
  std::string csv_results3_name = "ph_results3.csv";
  std::string csv_results_ref_name = "ph_results_ref.csv";
  csv_times.open(csv_times_name, std::fstream::out | std::fstream::trunc);
  csv_results.open(csv_results_name, std::fstream::out | std::fstream::trunc);
  csv_results2.open(csv_results2_name, std::fstream::out | std::fstream::trunc);
  csv_results3.open(csv_results3_name, std::fstream::out | std::fstream::trunc);
  csv_results_ref.open(csv_results_ref_name, std::fstream::out | std::fstream::trunc);

  std::string header = "N;time_shared;time_warp;time_warp_adapt;time_dot;time_cpuref;time_cublas";
    // to csv file
  csv_times << header << std::endl;
  
  std::string header_results = "N;sum;abs_sum;sqr_sum;mod_max;min;max;z_entries";
  csv_results << header_results << std::endl;
  csv_results2 << header_results << std::endl;
  csv_results3 << header_results << std::endl;
  csv_results_ref << header_results << std::endl;
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
    double *results3 = (double *)malloc(sizeof(double) * 7);
    double *results_ref = (double *)malloc(sizeof(double) * 7);
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
    std::copy(results, results+7, results3);
    std::copy(results, results+7, results_ref);
    timer.reset();
    // results_ref[0] = std::accumulate(x, x+N, 0.0);
    for (int i = 0; i < N; ++i){
      double tmp = x[i];
      results_ref[0] += tmp;
      results_ref[1] += std::abs(tmp);
      results_ref[2] += tmp*tmp;
      results_ref[4] = fmin(tmp, results_ref[4]);
      results_ref[5] = fmax(tmp, results_ref[5]);
      results_ref[6] += tmp ? 0 : 1;
    }
    results_ref[3] = fmax(std::abs(results_ref[4]), results_ref[5]);
    double time_cpuref = timer.get();
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
    double *cuda_scalar;
    cudaMalloc(&cuda_x, sizeof(double) * N);
    cudaMalloc(&cuda_results, sizeof(double) * 7);
    cudaMalloc(&cuda_scalar, sizeof(double));
    //
    // Copy data to GPU
    //
#ifdef DEBUG
    std::cout << "Copying data to GPU..." << std::endl;
#endif
    cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);

    //
    // Let CUBLAS do the work:
    //
#ifdef DEBUG
    std::cout << "Running dot products with CUBLAS..." << std::endl;
#endif
    double *cublas = (double *)malloc(sizeof(double));
    for (int iter = 0; iter < TESTS; ++iter) {
      *cublas= 0.0;
      cudaMemcpy(cuda_scalar, &cublas, sizeof(double), cudaMemcpyHostToDevice);
      timer.reset();
      cublasDdot(h, N, cuda_x, 1, cuda_x, 1, cublas);
      // cublasDnrm2(h, N-1, cuda_x, 1, cuda_scalar);
      // cudaMemcpy(&cublas, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);
      times[iter] = timer.get();
    }
    double time_cublas = median(times);
#ifdef DEBUG
    std::cout << "cublas: " << *cublas << std::endl;
#endif
#ifdef DEBUG
    std::cout << "Running with analyze_x_shared..." << std::endl;
#endif
  for (int iter = 0; iter < TESTS; ++iter) {
      cudaMemcpy(cuda_results, results, sizeof(double) * 7, cudaMemcpyHostToDevice);
      timer.reset();
      analyze_x_shared<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_x, cuda_results);
      cudaMemcpy(results, cuda_results, sizeof(double) * 7, cudaMemcpyDeviceToHost);
      times[iter] = timer.get();
    }
    double time_shared = median(times);

#ifdef DEBUG
    std::cout << "Running analyze_x_warp<GS, BS>..." << std::endl;
#endif
    for (int iter = 0; iter < TESTS; ++iter) {
      cudaMemcpy(cuda_results, results2, sizeof(double) * 7, cudaMemcpyHostToDevice);
      timer.reset();
      analyze_x_warp<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_x, cuda_results);
      cudaMemcpy(results2, cuda_results, sizeof(double) * 7, cudaMemcpyDeviceToHost);
      times[iter] = timer.get();
    }
    double time_warp = median(times);

#ifdef DEBUG
    std::cout << "Running analyze_x_warp<N/BS, BS>..." << std::endl;
#endif
    for (int iter = 0; iter < TESTS; ++iter) {
      cudaMemcpy(cuda_results, results3, sizeof(double) * 7, cudaMemcpyHostToDevice);
      int adapt_gridsize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; 
      // N/BLOCK_SIZE could results in a gridsize smaller than 1.
      // also,
      timer.reset();
      analyze_x_warp<<<adapt_gridsize, BLOCK_SIZE>>>(N, cuda_x, cuda_results);
      cudaMemcpy(results3, cuda_results, sizeof(double) * 7, cudaMemcpyDeviceToHost);
      times[iter] = timer.get();
    }
    double time_warp_adapt = median(times);

#ifdef DEBUG
    std::cout << "Running dot product xDOTy..." << std::endl;
#endif
    for (int iter = 0; iter < TESTS; ++iter) {
      double dot = 0.0;
      cudaMemcpy(cuda_scalar, &dot, sizeof(double), cudaMemcpyHostToDevice);
      timer.reset();
      xDOTy<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_x, cuda_x, cuda_scalar);
      cudaMemcpy(&dot, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);
      times[iter] = timer.get();
    }
    double time_dot = median(times);

    //
    // Compare results
    //
#ifdef DEBUG
    std::cout << "DEBUG output:" << std::endl;
    std::cout << "x:" << std::endl;
    int only = 4;
    printContainer(x, N, only);

    std::cout << ">SHARED<" << std::endl;
    printResults(results, results_ref, names, names.size());

    std::cout << ">WARP<" << std::endl;
    printResults(results2, results_ref, names, names.size());

    std::cout << "GPU shared runtime: " << time_shared << std::endl;
    std::cout << "GPU warp runtime: " << time_warp << std::endl;
    std::cout << "GPU warp adaptive runtime: " << time_warp_adapt << std::endl;
    std::cout << "GPU dot runtime: " << time_dot << std::endl;
    std::cout << "CPU ref runtime: " << time_cpuref << std::endl;

    //
    // Clean up:
    //
    std::cout << "Cleaning up..." << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
#endif

#ifdef CSV
    std::string sep = ";";
    csv_times << N << sep << time_shared << sep << time_warp << sep << time_warp_adapt << sep << time_dot << sep << time_cpuref << sep << time_cublas << std::endl;

    toCSV(csv_results, results, 7);
    toCSV(csv_results2, results2, 7);
    toCSV(csv_results3, results3, 7);
    toCSV(csv_results_ref, results_ref, 7);
#endif
    free(x);
    free(results);
    free(results2);
    free(results3);
    free(results_ref);
    free(cublas);

    cudaFree(cuda_x);
    cudaFree(cuda_results);
    cudaFree(cuda_scalar);

    cublasDestroy(h);
  }

#ifdef CSV
  csv_times.close();
  csv_results.close();
  csv_results2.close();
  csv_results3.close();
  csv_results_ref.close();
  
  std::cout << "\nRuntimes in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex6/" + csv_times_name << std::endl;
  std::cout << "\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex6/" + csv_results_name << std::endl;
  std::cout << "\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex6/" + csv_results2_name << std::endl;
  std::cout << "\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex6/" + csv_results3_name << std::endl;
  std::cout << "\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex6/" + csv_results_ref_name << std::endl;
#endif
  return EXIT_SUCCESS;
}

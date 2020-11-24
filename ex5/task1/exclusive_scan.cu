#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>

# define NUM_BLOCKS 256
# define THREADS_PER_BLOCK 256
# define MAX_POW 9
#define NUM_TESTS 3


__global__ void scan_kernel_1(double const *X,
                              double *Y,
                              int N,
                              double *carries)
{
  __shared__ double shared_buffer[NUM_BLOCKS];
  double my_value;

  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;

  // run scan on each section
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;

    // inclusive scan in shared buffer:
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();

    // exclusive scan requires us to write a zero value at the beginning of each block
    my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;

    block_offset += shared_buffer[blockDim.x-1];
  }

  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;

}

__global__ void scan_kernel_1_inc(double const *X,
  double *Y,
  int N,
  double *carries)
{
__shared__ double shared_buffer[NUM_BLOCKS];
double my_value;

unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
unsigned int block_offset = 0;

// run scan on each section
for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
{
// load data:
my_value = (i < N) ? X[i] : 0;

// inclusive scan in shared buffer:
for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
{
__syncthreads();
shared_buffer[threadIdx.x] = my_value;
__syncthreads();
if (threadIdx.x >= stride)
my_value += shared_buffer[threadIdx.x - stride];
}
__syncthreads();
shared_buffer[threadIdx.x] = my_value;
__syncthreads();

// my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
my_value = shared_buffer[threadIdx.x];

// write to output array
if (i < N)
Y[i] = block_offset + my_value;

block_offset += shared_buffer[blockDim.x-1];
}

// write carry:
if (threadIdx.x == 0)
carries[blockIdx.x] = block_offset;

}

// exclusive-scan of carries
__global__ void scan_kernel_2(double *carries)
{
  __shared__ double shared_buffer[NUM_BLOCKS];

  // load data:
  double my_carry = carries[threadIdx.x];

  // exclusive scan in shared buffer:

  for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();
    if (threadIdx.x >= stride)
      my_carry += shared_buffer[threadIdx.x - stride];
  }
  __syncthreads();
  shared_buffer[threadIdx.x] = my_carry;
  __syncthreads();

  // // write to output array
  carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

  // // write to output array
  // carries[threadIdx.x] = shared_buffer[threadIdx.x];
}

__global__ void scan_kernel_3(double *Y, int N,
                              double const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ double shared_offset;

  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];

  __syncthreads();

  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}

// // Shitfts array to left (to make it inclusive scan)
// __global__ void shiftToLeft(double *Y, int N, const double *X)
//  {
//   tid = blockDim.x * blockIdx.x + threadIdx.x;
//   for (int i = tid; i < N-1; i += gridDim.x * blockDim.x) {
//     Y[i] = Y[i+1];
//   }
//   // 0,1 ,2 ,3 ,4 , ...,N-2 ,N-1
//   //  /  /  /  /  /         /
//   // 0, 1, 2, 3, 4, ..., N-1
//   if (tid == 0)
//       Y[N-1] += X[N-1];
//  }
/* Task 1 a) */
__global__ void vectorAddInPlace(double* x, const double* y, const size_t N) {
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N; tid += blockDim.x * gridDim.x)
    x[tid] += y[tid];
}


void exclusive_scan(double const * input,
                    double * output, int N)
{
  double *carries;
  cudaMalloc(&carries, sizeof(double) * NUM_BLOCKS);

  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(input, output, N, carries);

  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, NUM_BLOCKS>>>(carries);

  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(output, N, carries);

  cudaFree(carries);
}

void inclusive_scan(double const * input,
                     double * output, int N)
{
  exclusive_scan(input, output, N);

  // To make inclusive
  vectorAddInPlace<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(output, input, N);
}

void inclusive_scan2(double const * input,
                     double * output, int N)
{
  double *carries;
  cudaMalloc(&carries, sizeof(double) * NUM_BLOCKS);

  // First step: Scan within each thread group and write carries
  scan_kernel_1_inc<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(input, output, N, carries);

  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, NUM_BLOCKS>>>(carries);

  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(output, N, carries);

  cudaFree(carries);
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

int main() {
  size_t max_N = (size_t)std::pow(10,MAX_POW);
  std::vector<size_t> vec_Ns;
  std::cout << "Check if correct Ns are calculated: (10^3- 10^" << MAX_POW << ")\n";
  int cnt = 2;
  for (size_t x = 1000; x <= max_N; x*=10) {
    if (x==max_N) x/=2;
    vec_Ns.push_back(x);
    std::cout << cnt++ << " : " << x << "\n";
  }
  std::cout << std::endl;

  Timer timer;
  // Container for runtimes
  std::vector<double> times_cpu_ex;
  std::vector<double> times_cpu_in;

  std::vector<double> times_gpu_ex;
  std::vector<double> times_gpu_in;
  std::vector<double> times_gpu_in2;
  std::vector<double> tmp(NUM_TESTS);

  for (size_t& N: vec_Ns) {
    //
    // Allocate host arrays for reference
    //
    double *x = (double *)malloc(sizeof(double) * N);
    double *y = (double *)malloc(sizeof(double) * N);
    double *z = (double *)malloc(sizeof(double) * N);
    double *z2 = (double *)malloc(sizeof(double) * N);
    std::fill(x, x + N, 1);

    // reference calculation EXCLUSIVE:
    for (int tests = 0; tests < NUM_TESTS; ++tests){
      timer.reset();
      y[0] = 0;
      for (std::size_t i=1; i<N; ++i) y[i] = y[i-1] + x[i-1];
      tmp[tests]= timer.get();
    }
    times_cpu_ex.push_back(median(tmp));

    //
    // Allocate CUDA-arrays
    //
    double *cuda_x, *cuda_y;
    cudaMalloc(&cuda_x, sizeof(double) * N);
    cudaMalloc(&cuda_y, sizeof(double) * N);
    cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);


    // Perform the exclusive scan and obtain results
    for (int tests = 0; tests < NUM_TESTS; ++tests){
    timer.reset();
    exclusive_scan(cuda_x, cuda_y, N);
      tmp[tests]= timer.get();
    }
    times_gpu_ex.push_back(median(tmp));
    cudaMemcpy(z, cuda_y, sizeof(double) * N, cudaMemcpyDeviceToHost);

    //
    // Print first few entries for reference
    //
    if (N ==100) {
      std::cout << "Exclusive scan:\n";
      std::cout << "CPU y: ";
      for (int i=0; i<10; ++i) std::cout << y[i] << " ";
      std::cout << " ... ";
      for (int i=N-10; i<N; ++i) std::cout << y[i] << " ";
      std::cout << std::endl;

      std::cout << "GPU y: ";
      for (int i=0; i<10; ++i) std::cout << z[i] << " ";
      std::cout << " ... ";
      for (int i=N-10; i<N; ++i) std::cout << z[i] << " ";
      std::cout << std::endl;
    }

    //--------- INCLUSIVE SCAN ---------- //
    // reference calculation INCLUSIVE:
    for (int tests = 0; tests < NUM_TESTS; ++tests){
      timer.reset();
      y[0] = x[0];
      for (std::size_t i=1; i<N; ++i) y[i] = y[i-1] + x[i];
      tmp[tests]= timer.get();
    }
    times_cpu_in.push_back(median(tmp));

    // Perform the inclusive scans and obtain results
    for (int tests = 0; tests < NUM_TESTS; ++tests){
      timer.reset();
      inclusive_scan(cuda_x, cuda_y, N);
      tmp[tests]= timer.get();
    }
    times_gpu_in.push_back(median(tmp));
    cudaMemcpy(z, cuda_y, sizeof(double) * N, cudaMemcpyDeviceToHost);

    for (int tests = 0; tests < NUM_TESTS; ++tests){
      timer.reset();
      inclusive_scan2(cuda_x, cuda_y, N);
      tmp[tests]= timer.get();
    }
    times_gpu_in2.push_back(median(tmp));
    cudaMemcpy(z2, cuda_y, sizeof(double) * N, cudaMemcpyDeviceToHost);

    if (N == 100) {
      std::cout << "Inclusive scan:\n";
      std::cout << "CPU y_: ";
      for (int i=0; i<10; ++i) std::cout << y[i] << " ";
      std::cout << " ... ";
      for (int i=N-10; i<N; ++i) std::cout << y[i] << " ";
      std::cout << std::endl;
  
      std::cout << "GPU y1: ";
      for (int i=0; i<10; ++i) std::cout << z[i] << " ";
      std::cout << " ... ";
      for (int i=N-10; i<N; ++i) std::cout << z[i] << " ";
      std::cout << std::endl;
      std::cout << "GPU y2: ";
      for (int i=0; i<10; ++i) std::cout << z2[i] << " ";
      std::cout << " ... ";
      for (int i=N-10; i<N; ++i) std::cout << z2[i] << " ";
      std::cout << std::endl;
    }


    //
    // Clean up:
    //
    free(x);
    free(y);
    free(z);
    cudaFree(cuda_x);
    cudaFree(cuda_y);
  }
  std::cout << "\n\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex5/in_ex_data_ph.csv" << std::endl;

  // OUT PUT TO CSV FILE
  std::string header = "N;times_cpu_ex;times_gpu_ex;times_cpu_in;times_gpu_in;times_gpu_in2";
  std::string sep = ";";

  std::fstream csv;
  csv.open ("in_ex_data_ph.csv", std::fstream::out | std::fstream::trunc);
  csv << header << std::endl;
  for (int i = 0; i < vec_Ns.size(); ++i ) {
    csv << std::scientific << vec_Ns[i] << sep 
      << times_cpu_ex[i] << sep 
      << times_gpu_ex[i] << sep 
      << times_cpu_in[i] << sep 
      << times_gpu_in[i] << sep 
      << times_gpu_in2[i] << "\n";
  }
  csv << std::endl;
  csv.close();

  return EXIT_SUCCESS;
}




#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
#include "poisson2d.hpp"
#include "timer.hpp"

#define BLOCK_SIZE 256
#define GRID_SIZE 256

__global__ void dot_product(double *x, double *y, double *dot, unsigned int n)
{
	unsigned int index = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned int stride = blockDim.x*gridDim.x;

	__shared__ double cache[256];

	double temp = 0.0;
	while(index < n){
		temp += x[index]*y[index];

		index += stride;
	}

	cache[threadIdx.x] = temp;

	__syncthreads();

    for(int i = blockDim.x/2; i>0; i/=2) 
    {
        __syncthreads();
        if(threadIdx.x < i)
            cache[threadIdx.x] += cache[threadIdx.x + i];
    }

	if(threadIdx.x == 0){
		atomicAdd(dot, cache[0]);
	}
}

// Naming is motivated by BLAS/LAPACK naming scheme...though bit simplified.
__global__ void xADDay(const size_t N, double *x, double *y, double *z, const double alpha)
{
  const size_t stride = blockDim.x * gridDim.x;
  for(size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += stride)
      z[i] = x[i] + alpha * y[i];
}

__global__ void xDOTy(const size_t N, double* x, double* y, double* z)
{
  size_t tid = threadIdx.x + blockDim.x* blockIdx.x;
  size_t stride = blockDim.x* gridDim.x;

  __shared__ double cache[BLOCK_SIZE];

  double tid_sum = 0.0;
  for (; tid < N; tid += stride)
  {
    tid_sum += x[tid] * y[tid];
  }
  tid = threadIdx.x;
  cache[tid] = tid_sum;

  __syncthreads();
  for (size_t i = blockDim.x/2; i != 0; i /=2)
  {
    __syncthreads();
    if (tid < i) //lower half does smth, rest idles
      cache[tid] += cache[tid + i]; //lower looks up by stride and sums up
  }

  if(tid == 0) // cache[0] now contains block_sum
  {
    atomicAdd(z, cache[0]);
  }
}

int main() {

  int N = 256;

  double xInit = 1.;
  double alpha = 2.;
  double yInit = 2.5;

  double *x = (double*)malloc(sizeof(double) * N);
  double *y = (double*)malloc(sizeof(double) * N);
  double *z = (double*)malloc(sizeof(double) * N);
  double *Dot = (double*)malloc(sizeof(double));
  *Dot = -1.;
  std::fill(x, x + N, xInit);
  std::fill(y, y + N, yInit);
  std::fill(z, z + N, 0.0);

  double *px, *py, *pz, *pDot;
  cudaMalloc(&px, N*sizeof(double));
  cudaMalloc(&py, N*sizeof(double));
  cudaMalloc(&pz, N*sizeof(double));
  cudaMalloc(&pDot, sizeof(double));
  cudaMemcpy(px, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(py, y, N*sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(pz, z, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(pDot, Dot, sizeof(double), cudaMemcpyHostToDevice);

  xADDay<<<GRID_SIZE, BLOCK_SIZE>>>(N, px, py, pz, alpha);
  cudaDeviceSynchronize();
  xDOTy<<<GRID_SIZE, BLOCK_SIZE>>>(N, px, py, pDot);
  //dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(px, py, pDot, N);
  cudaDeviceSynchronize();

  cudaMemcpy(z, pz, N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Dot, pDot, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  std::cout << "Checking xADDay..." << std::endl;
  int cnt = 0;
  for (int i = 0; i < N; ++i)
    if (z[i] != xInit + alpha*yInit) ++cnt;

  if (cnt) 
    std::cout << "Something went wrong...let's see:" << std::endl;
  else
    std::cout << "Everything ok, see:" << std::endl;

  for (int i = 0; i < 5; ++i)
    std::cout << "z[" << i << "] = " << z[i] << std::endl;
  std::cout << "..." << std::endl;
  for (int i = N-1-5; i < N; ++i)
    std::cout << "z[" << i << "] = " << z[i] << std::endl;

  std::cout << "-----------------------------------" << std::endl;
  std::cout << "Checking xDOTy..." << std::endl;
  if (*Dot != xInit*yInit*N)
    std::cout << "NOPE: " << *Dot << " != " << xInit*yInit*N << std::endl;
  else
    std::cout << "OK: " << *Dot << " == " << xInit*yInit*N << std::endl;

  free(x);
  free(y);
  free(z);
  free(Dot);
  cudaFree(px);
  cudaFree(py);
  cudaFree(pz);
  cudaFree(pDot);


  return EXIT_SUCCESS;
}

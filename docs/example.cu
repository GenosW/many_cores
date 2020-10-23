#include <stdio.h>
#include "timer.hpp"


__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1000000;

  float *x, *y, *d_x, *d_y;
  Timer timer;

  // Allocate host memory and initialize
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Allocate device memory and copy host data over
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // wait for previous operations to finish, then start timings
  cudaDeviceSynchronize();
  timer.reset();

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  // wait for kernel to finish, then print elapsed time
  cudaDeviceSynchronize();
  printf("Elapsed: %g\n", timer.get());

  // copy data back (implicit synchronization point)
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  // Numerical error check:
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  // tidy up host and device memory
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return EXIT_SUCCESS;
}


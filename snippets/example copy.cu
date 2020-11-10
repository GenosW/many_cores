#include <iostream>


int main(void)
{

  double  *x, *y, *d_x;

  // Allocate host memory and initialize
  x = (double*)malloc(sizeof(double ));
  *x = 0.5;
  y = (double*)malloc(sizeof(double));
  *y = 1.5;
  // Allocate device memory and copy host data over
  cudaMalloc(&d_x, sizeof(double)); 

  cudaMemcpy(d_x, x, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, y, sizeof(double), cudaMemcpyHostToDevice);

  cudaMemcpy(x, d_x, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  std::cout << *x << std::endl;


  cudaFree(d_x);
  free(x);
  free(y);

  return EXIT_SUCCESS;
}


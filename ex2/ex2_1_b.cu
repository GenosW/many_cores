#include <iostream>
#include "timer.hpp"

//for (; p<=3; p++);
  //      cout << *p;

double option1() {

}

int main(void)
{
  int N = 1000;
  int num_tests = 10;
  int i = 0;

  Timer timer;
  double total_time = 0.0;
  double runtime = 0.0, max_runtime = 0.0, min_runtime = 100.0;

  for (; i<num_tests; i++) {
    double *x, *y, *d_x, *d_y, *d_x2, *d_y2;
    x = (double*)malloc(N*sizeof(double));
    y = (double*)malloc(N*sizeof(double));
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));
    
    timer.reset();
    // Actual benchmark
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    runtime = timer.get();
    std::cout << "(" << i+1 << ") Elapsed: " << runtime << std::endl;
    total_time += runtime;
    if (runtime > max_runtime) {
        max_runtime = runtime;
    }
    if (runtime < min_runtime) {
      min_runtime = runtime;
  }
    if (total_time > 1.) {
        break;
    }
  }

  std::cout << std::endl << "Results after " << i << " tests:" << std::endl;
  std::cout << "Total runtime: " << total_time << std::endl;
  std::cout << "Average runtime; " << total_time/i << std::endl;
  std::cout << "Maximum runtime: " << max_runtime << std::endl;
  std::cout << "Minimum runtime: " << min_runtime << std::endl;


  return EXIT_SUCCESS;
}


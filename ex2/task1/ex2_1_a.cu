#include <iostream>
#include "timer.hpp"

//for (; p<=3; p++);
  //      cout << *p;

int main(void)
{
  int N = 1000;
  int num_tests = 10;
  int i = 0;

  Timer timer;
  double total_time = 0.0;
  double runtime = 0.0, max_runtime = 0.0, min_runtime = 100.0;

  for (; i<num_tests; i++) {
    double *d_x;
    timer.reset();

    cudaMalloc(&d_x, N*sizeof(double));
    cudaFree(d_x); 
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
  std::cout << "num_tests;min_runtime;average_time;max_runtime" << std::endl;
  std::cout << num_tests << ";" << min_runtime << ";" << total_time/i << ";" << max_runtime << std::endl;

  return EXIT_SUCCESS;
}


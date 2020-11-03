#include <iostream>
#include <string>
#include "timer.hpp"

//for (; p<=3; p++);
  //      cout << *p;

// double option1() {

// };

int main(void)
{
  int option = 1;
  int N = 1000;
  int num_tests = 3;
  int i = 0;

  Timer timer, timer2;
  double total_time = 0.0;
  double runtime = 0.0, max_runtime = 0.0, min_runtime = 100.0, hosttime = -1.;
  double *x_check, *y_check;
  x_check = (double*)malloc(N*sizeof(double));
  y_check = (double*)malloc(N*sizeof(double));


  for (; i<num_tests; i++) {
    double *x, *y, *d_x, *d_y;
    x = (double*)malloc(N*sizeof(double));
    y = (double*)malloc(N*sizeof(double));
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_y, N*sizeof(double));
    
    timer.reset();
    if (option == 1){
      timer2.reset();
      int j = 0;
      for (double *p = x; p <= x + sizeof(x); p++)
        {
          *p = j++;
        }
      for (double *p = y; p <= y + sizeof(y); p++)
        {
          *p = j--;
        }
      hosttime = timer2.get();

      cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
    }

    runtime = timer.get();
    std::cout << "(" << i+1 << ") Elapsed (total/host): " << runtime << " s/ " << hosttime << " s" << std::endl;
    total_time += runtime;

    if (i==0){
      int cnt = 0;
      for (double *p = x_check; p <= x_check + sizeof(x_check); p++)
        {
          *p = *(x + cnt++);
        }
      cnt = 0;
        for (double *p = y_check; p <= y_check + sizeof(y_check); p++)
        {
          *p = *(y + cnt++);
        }
        
    }


    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);

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

  std::cout << "\n--- Checking arrays ---" << std::endl;
  std::cout << "   i   |   x   |   y   " << std::endl;
  for (i=0; i < 3; i++){
    std::cout << "  " << i << "   |   " << *(x_check+i) << "   |   " << *(y_check+i) << std::endl;
  }

  free(x_check);
  free(y_check);

  return EXIT_SUCCESS;
}


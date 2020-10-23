#include <iostream>
#include <chrono>
#include <vector>

// CUDA Kernel
__global__
void saxpy(int n, double a, double *x, double *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
/* ------------------------------------------------------------------------------*/

// Helper functions
template<typename T>
T convertTo(const int position, const T default, int argc, char *argv[], int debug) {
  if (argc <= position) 
	{
	if (debug)
		{
		std::cout
			<< "Conversion of argument " << position
			<< " failed, not enough parameters, using default parameter: "
			<< default << std::endl;
		}
	return default;
	}
	T arg;
	std::istringstream tmp(argv[position]);
	tmp >> arg;
	// tmp >> arg ?  (std::cout << "Conversion of argument " << position << "  successfull: " << arg)
	//               : (std::cout << "Conversion of argument " << position
	//                            << "  failed");

	return arg;
}
/* ------------------------------------------------------------------------------*/

// Benchmarks
class CudaBenchmark {
public:
  double runtime;
  std::vector<double> vec;

  CudaBenchmark(std::vector<double> vect) : vec(vect) {
    runtime = -1.0;
  };

  double run(){
    cout << " - Option " << option << endl;
    std::chrono::time_point<std::chrono::steady_clock> stop;
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();

    allocate();

    stop = std::chrono::steady_clock::now();
    runtime = stop - start;
    std::cout << "Elapsed time: " << runtime << endl;
    std::cout << "----------------------------";

    free();
    return runtime;
  };

private:
  virtual void allocate();
  virtual void free();
};
/* --------- */

class BM1 : CudaBenchmark {
private:
  void allocate() override {
    // Allocate device memory and copy host data over
    cudaMalloc(&d_x, N*sizeof(double)); 
    cudaMalloc(&d_y, N*sizeof(double));
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();    
  };

  void free() override {
    cudaFree(d_x);
    cudaFree(d_y);
    cudaDeviceSynchronize();
  };
}
/* --------- */

/* ------------------------------------------------------------------------------*/


/**
 * ./main N
 */
int main(int argc, char *argv[])
{
  assert(argc == 2);
  int N = convertTo<int>(1, 1000, argc, argv, true);
  int bm_mode = convertTo<int>(2, 0, argc, argv, true);
  int option_number = 0;
  double *x, *y, *d_x, *d_y;

  std::vector<double> runtimes{ -1., -1. , -1.};

  cout << "****************************" << endl;
  cout << "--- Basic CUDA benchmark ---" << endl;
  cout << "****************************" << endl << endl;
  
  // Allocate host memory and initialize
  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));

  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = N-1-i;
  }

  // ---- Option 1 ---- //
  option_number = 0;
  
  // ------------------ //





  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  // wait for kernel to finish, then print elapsed time
  cudaDeviceSynchronize();  

  // copy data back (implicit synchronization point)
  cudaMemcpy(y, d_y, N*sizeof(double), cudaMemcpyDeviceToHost);

  // Numerical error check:
  double maxError = 0.0f;
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


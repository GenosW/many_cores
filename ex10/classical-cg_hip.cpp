#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
// #include <stdio.h>
#include "hip/hip_runtime.h"

// DEFINES
#define EX "ex10"
#define CSV_NAME "ph_data_hip.csv"
#define N_MAX_PRINT 32
#define PRINT_ONLY 10
#define NUM_TESTS 10 // should be uneven so we dont have to copy after each iteration

#define GRID_SIZE 512
#define BLOCK_SIZE 512
#define USE_MY_ATOMIC_ADD
#define HIP_ASSERT(x) (assert((x)==hipSuccess)) // only used it once

/** atomicAdd for doubles for hip for nvcc for many cores exercise 10 for me
 * by: Peter HOLZNER feat. NVIDIA
 * 
 * - Ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 * 
 * 'Don't let your memes be dreams!'
 * - Probably Ghandi, idk
 */
__device__ double 
my_atomic_Add(double* address, double val)
{
  using ulli = unsigned long long int;
  ulli* address_as_ull =
                            (ulli*)address;
  ulli old = *address_as_ull, assumed;
  do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val +
                              __longlong_as_double(assumed)));

  } while (assumed != old);
  return __longlong_as_double(old);
};

// y = A * x
__global__ void 
hip_csr_matvec_product(int N, 
                      int *csr_rowoffsets, int *csr_colindices, 
                      double *csr_values,
                      double *x, double *y)
{
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * x[csr_colindices[k]];
    }
    y[i] = sum;
  }
}

// x <- x + alpha * y
__global__ void 
hip_vecadd(int N, double *x, double *y, double alpha)
{
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x)
    x[i] += alpha * y[i];
}

// x <- y + alpha * x
__global__ void 
hip_vecadd2(int N, double *x, double *y, double alpha)
{
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x)
    x[i] = y[i] + alpha * x[i];
}

/**result = (x, y)
 */
__global__ void 
hip_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[BLOCK_SIZE];

  double dot = 0;
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < N; i += hipBlockDim_x * hipGridDim_x) {
    dot += x[i] * y[i];
  }

  shared_mem[hipThreadIdx_x] = dot;
  for (int k = hipBlockDim_x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (hipThreadIdx_x < k) {
      shared_mem[hipThreadIdx_x] += shared_mem[hipThreadIdx_x + k];
    }
  }

  if (hipThreadIdx_x == 0) 
  {
#ifdef USE_MY_ATOMIC_ADD
    my_atomic_Add(result, shared_mem[0]);
#else
    atomicAdd(result, shared_mem[0]);
#endif
  }
}



/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse
 * matrix-vector product) are transferred to hip kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use
 * with hip. Modify as you see fit.
 *
 * Modifications:
 * - returns runtime as double
 * - iteration counter (iters) is passed as reference for logging to csv-file
 * - replaced cuda with hip (literally search-and-replaced the word...)
 * - implemented the hip-style kernel launches (although unnecessary for this
 *   exercise since we pass it to nvcc anyway :D)
 */
double conjugate_gradient(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution,
                        int& iters)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;

  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);

  // initialize work vectors:
  double alpha, beta;
  double *hip_solution, *hip_p, *hip_r, *hip_Ap, *hip_scalar;
  hipMalloc(&hip_p, sizeof(double) * N);
  hipMalloc(&hip_r, sizeof(double) * N);
  hipMalloc(&hip_Ap, sizeof(double) * N);
  hipMalloc(&hip_solution, sizeof(double) * N);
  hipMalloc(&hip_scalar, sizeof(double));

  hipMemcpy(hip_p, rhs, sizeof(double) * N, hipMemcpyHostToDevice);
  hipMemcpy(hip_r, rhs, sizeof(double) * N, hipMemcpyHostToDevice);
  hipMemcpy(hip_solution, solution, sizeof(double) * N, hipMemcpyHostToDevice);

  const double zero = 0;
  double residual_norm_squared = 0;

  hipMemcpy(hip_scalar, &zero, sizeof(double), hipMemcpyHostToDevice);

  // hip_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_r, hip_r, hip_scalar);
  hipLaunchKernelGGL(hip_dot_product, // kernel
    GRID_SIZE, BLOCK_SIZE,            // device params
    0, 0,                             // shared mem, default stream
    N, hip_r, hip_r, hip_scalar       // kernel arguments
  );

  hipMemcpy(&residual_norm_squared, hip_scalar, sizeof(double), hipMemcpyDeviceToHost);

  double initial_residual_squared = residual_norm_squared;

  iters = 0;
  hipDeviceSynchronize();
  timer.reset();
  while (1) {

    // line 4: A*p:
    // hip_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, hip_p, hip_Ap);
    hipLaunchKernelGGL(hip_csr_matvec_product, // kernel
      GRID_SIZE, BLOCK_SIZE,            // device params
      0, 0,                             // shared mem, default stream
      N, csr_rowoffsets, csr_colindices, csr_values, hip_p, hip_Ap       // kernel arguments
    );

    // lines 5,6:
    hipMemcpy(hip_scalar, &zero, sizeof(double), hipMemcpyHostToDevice);
    // hip_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_p, hip_Ap, hip_scalar);
    hipLaunchKernelGGL(hip_dot_product, // kernel
      GRID_SIZE, BLOCK_SIZE,            // device params
      0, 0,                             // shared mem, default stream
      N, hip_p, hip_Ap, hip_scalar      // kernel arguments
    );
    hipMemcpy(&alpha, hip_scalar, sizeof(double), hipMemcpyDeviceToHost);
    alpha = residual_norm_squared / alpha;

    // line 7:
    // hip_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_solution, hip_p, alpha);
    hipLaunchKernelGGL(hip_vecadd,    // kernel
      GRID_SIZE, BLOCK_SIZE,          // device params
      0, 0,                           // shared mem, default stream
      N, hip_solution, hip_p, alpha   // kernel arguments
    );

    // line 8:
    // hip_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_r, hip_Ap, -alpha);
    hipLaunchKernelGGL(hip_vecadd, // kernel
      GRID_SIZE, BLOCK_SIZE,       // device params
      0, 0,                        // shared mem, default stream
      N, hip_r, hip_Ap, -alpha     // kernel arguments
    );

    // line 9:
    beta = residual_norm_squared;
    HIP_ASSERT(hipMemcpy(hip_scalar, &zero, sizeof(double), hipMemcpyHostToDevice)); // just checking if this works properly
    // hipMemcpy(hip_scalar, &zero, sizeof(double), hipMemcpyHostToDevice);
    // hip_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_r, hip_r, hip_scalar);
    hipLaunchKernelGGL(hip_dot_product, // kernel
      GRID_SIZE, BLOCK_SIZE,            // device params
      0, 0,                             // shared mem, default stream
      N, hip_r, hip_r, hip_scalar       // kernel arguments
    );
    hipMemcpy(&residual_norm_squared, hip_scalar, sizeof(double), hipMemcpyDeviceToHost);

    // line 10:
    if (std::sqrt(residual_norm_squared / initial_residual_squared) < 1e-6) {
      break;
    }

    // line 11:
    beta = residual_norm_squared / beta;

    // line 12:
    // hip_vecadd2<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_p, hip_r, beta);
    hipLaunchKernelGGL(hip_vecadd2, // kernel
      GRID_SIZE, BLOCK_SIZE,            // device params
      0, 0,                             // shared mem, default stream
      N, hip_p, hip_r, beta      // kernel arguments
    );

    if (iters > 10000)
      break; // solver didn't converge
    ++iters;
  }
  hipMemcpy(solution, hip_solution, sizeof(double) * N, hipMemcpyDeviceToHost);

  hipDeviceSynchronize();
  double runtime = timer.get();
  std::cout << "Time elapsed: " << runtime << " (" << runtime / iters << " per iteration)" << std::endl;

  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
              << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations."
              << std::endl;

  hipFree(hip_p);
  hipFree(hip_r);
  hipFree(hip_Ap);
  hipFree(hip_solution);
  hipFree(hip_scalar);

  return runtime;
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int points_per_direction) {

  int N = points_per_direction *
          points_per_direction; // number of unknows to solve for

  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix
  // a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros
  //       per row in the system matrix, so we can allocate accordingly.
  //
  int *csr_rowoffsets = (int *)malloc(sizeof(double) * (N + 1));
  int *csr_colindices = (int *)malloc(sizeof(double) * 5 * N);
  double *csr_values = (double *)malloc(sizeof(double) * 5 * N);

  int *hip_csr_rowoffsets, *hip_csr_colindices;
  double *hip_csr_values;
  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices,
                       csr_values);

  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double *)malloc(sizeof(double) * N);
  double *rhs = (double *)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);

  //
  // Allocate hip-arrays //
  //
  hipMalloc(&hip_csr_rowoffsets, sizeof(double) * (N + 1));
  hipMalloc(&hip_csr_colindices, sizeof(double) * 5 * N);
  hipMalloc(&hip_csr_values, sizeof(double) * 5 * N);
  hipMemcpy(hip_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), hipMemcpyHostToDevice);
  hipMemcpy(hip_csr_colindices, csr_colindices, sizeof(double) * 5 * N,   hipMemcpyHostToDevice);
  hipMemcpy(hip_csr_values,     csr_values,     sizeof(double) * 5 * N,   hipMemcpyHostToDevice);

  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  int iters = 0; // pass into the CG so we can track it
  double runtime = conjugate_gradient(N, hip_csr_rowoffsets, hip_csr_colindices, hip_csr_values, rhs, solution, iters);

  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm
            << " (should be smaller than 1e-6)" << std::endl;

  // not optimal (efficient), but minimally invasive --> easy to copy
  std::ofstream csv;
  csv.open(CSV_NAME, std::fstream::out | std::fstream::app);
  csv << points_per_direction << ";" 
    << N << ";"
    << runtime << ";"
    << residual_norm << ";"
    << iters << std::endl;
  csv.close();

  for (int i = 0; i < N; i++)
    std::cout << solution[i] << std::endl;

  hipFree(hip_csr_rowoffsets);
  hipFree(hip_csr_colindices);
  hipFree(hip_csr_values);
  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}

int main() {

  std::ofstream csv;
  csv.open(CSV_NAME, std::fstream::out | std::fstream::trunc);
  csv << "p;N;runtime;residual;iterations" << std::endl;
  csv.close();

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;
  std::cout << "hip Device prop succeeded " << std::endl ;

  // std::vector<int> p_per_dir{ 10, 100, 500,1000, 1500};

  std::vector<int> p_per_dir{ 10};

  for (auto& p : p_per_dir)
  {
    std::cout << "--------------------------" << std::endl;
    solve_system(p); // solves a system with p*p unknowns
  }
  std::cout << "\nData: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;

  return EXIT_SUCCESS;
}


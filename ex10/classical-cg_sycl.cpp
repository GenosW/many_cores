#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
// #include <stdio.h>
#include <CL/sycl.hpp>

#define GRID_SIZE 512
#define BLOCK_SIZE 512

// #ifdef HIP
//   //
//   // Transformation to HIP
//   //
//   // -- runtime
//   #define DEVICE_ALLOC hipMalloc
//   #define DEVICE_FREE hipFree

//   // -- kernel language
//   #define KERNEL __global__
//   #define GLOBALMEM 
//   #define LOCALMEM __shared__
//   #define DEVICE_FUNC __device__
//   #define BARRIER __syncthreads
//   #define ATOMIC_ADD atomicAdd

//   // -- thread management
//   #define LOCALID hipThreadIdx_x
//   #define GROUPID hipBlockIdx_x
//   #define GLOBALID ((hipBlockDim_x * hipBlockIdx_x) + hipThreadIdx_x)

//   #define LOCALSIZE hipBlockDim_x
//   #define GLOBALSIZE (hipGridDim_x * hipBlockDim_x)
// #endif


#define USE_MY_ATOMIC_ADD

using ulli = unsigned long long int;
/** atomicAdd for doubles for hip for nvcc for many cores exercise 10 for me
 * by: Peter HOLZNER feat. NVIDIA
 * 
 * based on this little guy:
 * unsigned long long int atomicAdd(unsigned long long int* address,unsigned long long int val)
 * 
 * 'Don't let your memes be dreams!'
 * - Probably Ghandi, idk
 */
__device__ double 
my_atomic_Add(double *p, double val) 
{
  ulli* address_as_ul = (ulli *) p; 
  ulli old = *address_as_ul, assumed;
  ulli val_as_ul =  (ulli) val;
  do  {
    assumed = old;
    old = atomicAdd(address_as_ul, val_as_ul);
  } while (assumed != old);
  return (double) old;
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

// result = (x, y)
__global__ void 
hip_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[512];

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
 */
void conjugate_gradient(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution)
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

  int iters = 0;
  hipDeviceSynchronize();
  timer.reset();
  while (1) {

    // line 4: A*p:
    hip_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, hip_p, hip_Ap);

    // lines 5,6:
    hipMemcpy(hip_scalar, &zero, sizeof(double), hipMemcpyHostToDevice);
    hip_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_p, hip_Ap, hip_scalar);
    hipMemcpy(&alpha, hip_scalar, sizeof(double), hipMemcpyDeviceToHost);
    alpha = residual_norm_squared / alpha;

    // line 7:
    hip_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_solution, hip_p, alpha);

    // line 8:
    hip_vecadd<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_r, hip_Ap, -alpha);

    // line 9:
    beta = residual_norm_squared;
    HIP_ASSERT(hipMemcpy(hip_scalar, &zero, sizeof(double), hipMemcpyHostToDevice)); // just checking if this works properly
    hip_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_r, hip_r, hip_scalar);
    hipMemcpy(&residual_norm_squared, hip_scalar, sizeof(double), hipMemcpyDeviceToHost);

    // line 10:
    if (std::sqrt(residual_norm_squared / initial_residual_squared) < 1e-6) {
      break;
    }

    // line 11:
    beta = residual_norm_squared / beta;

    // line 12:
    hip_vecadd2<<<GRID_SIZE, BLOCK_SIZE>>>(N, hip_p, hip_r, beta);

    if (iters > 10000)
      break; // solver didn't converge
    ++iters;
  }
  hipMemcpy(solution, hip_solution, sizeof(double) * N, hipMemcpyDeviceToHost);

  hipDeviceSynchronize();
  std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

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
  conjugate_gradient(N, hip_csr_rowoffsets, hip_csr_colindices, hip_csr_values, rhs, solution);

  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  std::cout << "Relative residual norm: " << residual_norm
            << " (should be smaller than 1e-6)" << std::endl;

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

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;

  std::cout << "hip Device prop succeeded " << std::endl ;

  solve_system(10); // solves a system with 100*100 unknowns

  return EXIT_SUCCESS;
}


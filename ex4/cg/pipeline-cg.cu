#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <stdio.h>

// Block and grid size defines.
// Seperate defines are really just for future convenience...
#define BLOCK_SIZE 512
#define GRID_SIZE 512
#define SEP ";"
//#define DEBUG

// y = A * x
__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
  int *csr_colindices, double *csr_values,
  double *x, double *y)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
double sum = 0;
for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
sum += csr_values[k] * x[csr_colindices[k]];
}
y[i] = sum;
}
}

// x <- x + alpha * y
__global__ void cuda_vecadd(int N, double *x, double *y, double alpha)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
x[i] += alpha * y[i];
}

// x <- y + alpha * x
__global__ void cuda_vecadd2(int N, double *x, double *y, double alpha)
{
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
x[i] = y[i] + alpha * x[i];
}

// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
{
__shared__ double shared_mem[BLOCK_SIZE];

double dot = 0;
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
dot += x[i] * y[i];
}

shared_mem[threadIdx.x] = dot;
for (int k = blockDim.x / 2; k > 0; k /= 2) {
__syncthreads();
if (threadIdx.x < k) {
shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
}
}

if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}

__global__ void part1(int N, 
  double* x, double* r, double *p, double *Ap,
  double alpha, double beta)
{
  // lines 2 , 3 + 4
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    x[i] = x[i] + alpha * p[i];
    double r_tmp = r[i] - alpha * Ap[i];
    r[i] = r_tmp;
  //}
  // Merge these two?
  //for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    p[i] = r_tmp + beta * p[i];
  }
}

__global__ void part2(int N, 
  int *csr_rowoffsets, int *csr_colindices, double *csr_values,
  double* r, double *p, double *Ap,
  double* ApAp, double* pAp, double* rr
  )
{
  __shared__ double shared_mem_ApAp[BLOCK_SIZE];
  __shared__ double shared_mem_pAp[BLOCK_SIZE];
  __shared__ double shared_mem_rr[BLOCK_SIZE];
  // Mat-vec product
  double dot_ApAp = 0., dot_pAp = 0., dot_rr = 0.;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * p[csr_colindices[k]];
    }
    Ap[i] = sum;
    dot_ApAp += sum*sum;
    dot_pAp += p[i]*sum;
    dot_rr += r[i]*r[i];
  }
  // now :
  // Ap = Ap_i --> Line 5
  // we are ready for reductions

  shared_mem_ApAp[threadIdx.x] = dot_ApAp;
  shared_mem_pAp[threadIdx.x] = dot_pAp;
  shared_mem_rr[threadIdx.x]  = dot_rr;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem_ApAp[threadIdx.x] += shared_mem_ApAp[threadIdx.x + k];
      shared_mem_pAp[threadIdx.x] += shared_mem_pAp[threadIdx.x + k];
      shared_mem_rr[threadIdx.x] += shared_mem_rr[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0) {
    atomicAdd(ApAp, shared_mem_ApAp[0]);
    atomicAdd(pAp, shared_mem_pAp[0]);
    atomicAdd(rr, shared_mem_rr[0]);
  }
  // now:
  // ApAp, pAp, rr --> Line 6
}



/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse
 * matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use
 * with CUDA. Modify as you see fit.
 */
int conjugate_gradient(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;

  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);

  // initialize work vectors:
  double alpha, beta, pAp, ApAp, rr;
  double* cuda_pAp, *cuda_ApAp, *cuda_rr;
  double* cuda_x, *cuda_p, *cuda_r, *cuda_Ap;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_x, sizeof(double) * N);
  cudaMalloc(&cuda_pAp, sizeof(double));
  cudaMalloc(&cuda_ApAp, sizeof(double));
  cudaMalloc(&cuda_rr, sizeof(double));

  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_x, solution, sizeof(double) * N, cudaMemcpyHostToDevice);

  const double zero = 0;
  cudaMemcpy(cuda_pAp, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_ApAp, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_rr, &zero, sizeof(double), cudaMemcpyHostToDevice);

  // Initial values: i = 0
  // device
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_r, cuda_r, cuda_rr);
  cuda_csr_matvec_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_p, cuda_Ap, cuda_pAp);
  cuda_dot_product<<<GRID_SIZE, BLOCK_SIZE>>>(N, cuda_Ap, cuda_Ap, cuda_ApAp);
  cudaMemcpy(&rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // host side of things
  double initial_residual_squared = rr;

#ifdef DEBUG
  std::cout << "Initial residual norm: " << initial_residual_squared << std::endl;
#endif
  alpha = rr / pAp;
  //beta = (alpha*alpha * ApAp - rr) / rr;
  beta = alpha * alpha * ApAp / rr - 1;

  int iters = 1;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {
    part1<<<BLOCK_SIZE, GRID_SIZE>>>(N, 
      cuda_x, cuda_r, cuda_p, cuda_Ap,
      alpha, beta);

    cudaMemcpy(cuda_pAp, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_ApAp, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_rr, &zero, sizeof(double), cudaMemcpyHostToDevice);
    part2<<<BLOCK_SIZE, GRID_SIZE>>>(N, 
      csr_rowoffsets, csr_colindices, csr_values,
      cuda_r, cuda_p, cuda_Ap,
      cuda_ApAp, cuda_pAp, cuda_rr);

    cudaDeviceSynchronize();
    cudaMemcpy(&rr, cuda_rr, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&pAp, cuda_pAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ApAp, cuda_ApAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // line 10:
    double rel_norm = std::sqrt(rr / initial_residual_squared);
    if (rel_norm < 1e-6) {
      break;
    }
    alpha = rr / pAp;
    //beta = (alpha*alpha * ApAp - rr) / rr;
    beta = alpha * alpha * ApAp / rr - 1;

#ifdef DEBUG
    if (iters%100==0) {
      std::cout << "Norm after " << iters << " iterations:\n"
        << "rel. norm: " << rel_norm << "\n"
        << "abs. norm: " << std::sqrt(beta) << std::endl;
    }
#endif
    if (iters > 10000)
      break; // solver didn't converge
    ++iters;
  }
  cudaMemcpy(solution, cuda_x, sizeof(double) * N, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
#ifdef DEBUG
  std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
              << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations."
              << std::endl;
#endif
  // Vectors
  cudaFree(cuda_x);
  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  // Scalers
  cudaFree(cuda_pAp);
  cudaFree(cuda_ApAp);
  cudaFree(cuda_rr);
  return iters;
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int points_per_direction) {

  Timer timer;
  int N = points_per_direction *
          points_per_direction; // number of unknows to solve for
#ifdef DEBUG
  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;
#endif
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

  int *cuda_csr_rowoffsets, *cuda_csr_colindices;
  double *cuda_csr_values;
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
  // Allocate CUDA-arrays //
  //
  cudaMalloc(&cuda_csr_rowoffsets, sizeof(double) * (N + 1));
  cudaMalloc(&cuda_csr_colindices, sizeof(double) * 5 * N);
  cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
  cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(double) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_values,     csr_values,     sizeof(double) * 5 * N,   cudaMemcpyHostToDevice);

  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  timer.reset();
  int iters = conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);
  double runtime = timer.get();

    //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);

#ifdef DEBUG
  std::cout << "Time elapsed: " << runtime << " (" << runtime) / iters << " per iteration)" << std::endl;
  std::cout << "Relative residual norm: " << residual_norm
          << " (should be smaller than 1e-6)" << std::endl;
#endif
#ifndef DEBUG
  std::cout << points_per_direction << SEP
    << N << SEP
    << runtime << SEP
    << iters << SEP
    << residual_norm << std::endl;
#endif
  cudaFree(cuda_csr_rowoffsets);
  cudaFree(cuda_csr_colindices);
  cudaFree(cuda_csr_values);
  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}

int main() {

  std::vector<size_t> p_per_dir{ (size_t)sqrt(1e3), (size_t)sqrt(1e4), (size_t)sqrt(1e5), (size_t)sqrt(1e6), (size_t)sqrt(4e6)};
  std::cout << "p" << SEP "N" << SEP
    << "time" << SEP << "iters" << SEP<< "norm_after" << std::endl;
  for (auto& points: p_per_dir)
    solve_system(points); // solves a system with 100*100 unknowns

  return EXIT_SUCCESS;
}


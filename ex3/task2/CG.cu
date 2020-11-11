
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>
#include "poisson2d.hpp"
#include "timer.hpp"

#define BLOCK_SIZE 256
#define GRID_SIZE 256

/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
__global__ void csr_Ax(const size_t N,
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *x, double *y)
{
  const size_t stride = gridDim.x * blockDim.x;
  for (int i = threadIdx.x + blockDim.x * blockIdx.x;
    i < N;
    i += stride)
  {
    double tmp = 0.0;
    for (int j = csr_rowoffsets[i]; j < csr_rowoffsets[i+1]; ++j)
      tmp += csr_values[j] * x[csr_colindices[j]];
    y[i] = tmp;
  }
}

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

/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use with CUDA.
 *  Modify as you see fit.
 */
void conjugate_gradient(const size_t N,  // number of unknows
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *h_rhs,
                        double *h_solution,
                        const double conv_factor)
                        //, double *init_guess)   // feel free to add a nonzero initial guess as needed
{

  // clear solution vector (it may contain garbage values):
  std::fill(h_solution, h_solution + N, 0.0);

  // initialize work vectors:
  double* h_pAp = (double*)malloc(sizeof(double));
  double* h_r2 = (double*)malloc(sizeof(double));
  double* h_r22 = (double*)malloc(sizeof(double));
  double* zero = (double*)malloc(sizeof(double));
  *zero = 0.00;
  *h_pAp = 0.00;
  *h_r2 = 0.00;
  *h_r22 = 0.00;
  double* x; 
  double* p; 
  double* r; 
  double* Ap; 
  double* pAp; 
  double* r2;  

  // arrays
  const size_t arr_size = N*sizeof(double);
  cudaMalloc(&x, arr_size);
  cudaMalloc(&p, arr_size);
  cudaMalloc(&r, arr_size);
  cudaMalloc(&Ap, arr_size);
  // scalars
  cudaMalloc(&pAp, sizeof(double));
  cudaMalloc(&r2, sizeof(double));

  // line 2: initialize r and p:
  //std::copy(h_rhs, h_rhs+N, h_p);
  //std::copy(h_rhs, h_rhs+N, h_r);
  cudaMemcpy(x, h_solution, arr_size, cudaMemcpyHostToDevice);
  cudaMemcpy(r, h_rhs, arr_size, cudaMemcpyHostToDevice);
  cudaMemcpy(Ap, h_rhs, arr_size, cudaMemcpyHostToDevice);
  cudaMemcpy(p, h_rhs, arr_size, cudaMemcpyHostToDevice);

  double alpha, beta;
  int iters = 0;
  //while (1) {
  while (iters < 10000) { // will end with iter == 10'000 or earlier
    cudaMemcpy(r2, zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(pAp, zero, sizeof(double), cudaMemcpyHostToDevice);
    
    // 4: Ap = A * p
    csr_Ax<<<GRID_SIZE, BLOCK_SIZE>>>(N, csr_rowoffsets, csr_colindices, csr_values, p, Ap);
    // 5: pAp = <p,Ap>
    xDOTy<<<GRID_SIZE, BLOCK_SIZE>>>(N, p, Ap, pAp);
    // r2 = <r,r>
    xDOTy<<<GRID_SIZE, BLOCK_SIZE>>>(N, r, r, r2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_pAp, pAp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r2, r2, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // 6: alpha = <r,r>/<p,Ap>
    alpha = (*h_r2) / (*h_pAp);
    // 7: x = x_i+1 = ...
    xADDay<<<GRID_SIZE, BLOCK_SIZE>>>(N, x, p, x, alpha);
    // 8: r = r_i+1 = ...
    xADDay<<<GRID_SIZE, BLOCK_SIZE>>>(N, r, Ap, r, -alpha);

    // 9: r2 = <r,r>
    xDOTy<<<GRID_SIZE, BLOCK_SIZE>>>(N, r, r, r2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_r22, r2, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // // 10: check
    if (iters < 10 or iters > 10000 - 10)
      std::cout << "r2[" << iters << "] = " << *h_r2 << " vs " << conv_factor << std::endl;
    if (*h_r22 < conv_factor) {
      break;
    }

    // beta = beta_i = ...
    beta = (*h_r22) / (*h_r2);
    // 10: check
    // if (iters < 10 or iters > 10000 - 10)
    //   std::cout << "r2[" << iters << "] = " << beta << " vs " << conv_factor << std::endl;
    // if (beta < conv_factor or beta > 10) {
    //   break;
    // }
    // 12: p = p_i+1 = ...
    xADDay<<<GRID_SIZE, BLOCK_SIZE>>>(N, r, p, p, beta);
    cudaDeviceSynchronize();

    ++iters;
  }

  if (iters >= 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations with r^2 = " << *h_r2 << std::endl;
  else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations with r^2 = " << *h_r2 << std::endl;

  cudaMemcpy(h_solution, x, arr_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(x);
  cudaFree(p);
  cudaFree(r);
  cudaFree(Ap);
  cudaFree(pAp);
  cudaFree(r2);
  free(h_pAp);
  free(h_r2);
  free(h_r22);
}



/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void solve_system(size_t points_per_direction) {

  size_t N = points_per_direction * points_per_direction; // number of unknows to solve for

  std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
  //
  const size_t size_row = sizeof(int) * (N+1);
  const size_t size_col = sizeof(int) * 5 * N;
  const size_t size_val = sizeof(double) * 5 * N;
  int *h_csr_rowoffsets =    (int*)malloc(size_row);
  int *h_csr_colindices =    (int*)malloc(size_col);
  double *h_csr_values  = (double*)malloc(size_val);

  int* csr_rowoffsets;
  int* csr_colindices;
  double* csr_values;
  cudaMalloc(&csr_rowoffsets, size_row);
  cudaMalloc(&csr_colindices, size_col);
  cudaMalloc(&csr_values, size_val);

  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(points_per_direction, h_csr_rowoffsets, h_csr_colindices, h_csr_values);

  cudaMemcpy(csr_rowoffsets, h_csr_rowoffsets, size_row, cudaMemcpyHostToDevice);
  cudaMemcpy(csr_colindices, h_csr_colindices, size_col, cudaMemcpyHostToDevice);
  cudaMemcpy(csr_values, h_csr_values, size_val, cudaMemcpyHostToDevice);

  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double*)malloc(sizeof(double) * N);
  double *rhs      = (double*)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);

  //
  // Call Conjugate Gradient implementation (CPU arrays passed here; modify to use GPU arrays)
  // CSR Matrix is passed as GPU arrays already.
  // rhs and solution are CPU arrays.
  // This isn't a nice setup obviously...but it's little more than a 1 file "script", so I think that's fine for now.
  //
  double conv_factor = 1e-6; //1e-6
  conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution, conv_factor);

  //
  // Check for convergence:
  //
  double residual_norm = relative_residual(N, h_csr_rowoffsets, h_csr_colindices, h_csr_values, rhs, solution);
  std::string check = "OK";
  if (residual_norm > conv_factor) check = "FAIL";
  std::cout << "Relative residual norm: " << residual_norm << " (should be smaller than 1e-6): " << check << std::endl;

  cudaFree(csr_rowoffsets);
  cudaFree(csr_colindices);
  cudaFree(csr_values);
  free(solution);
  free(rhs);
  free(h_csr_rowoffsets);
  free(h_csr_colindices);
  free(h_csr_values);
}


int main() {

  solve_system(10); // solves a system with 100*100 unknowns

  return EXIT_SUCCESS;
}

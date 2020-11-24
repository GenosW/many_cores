#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdio.h>

// Block and grid size defines.
// Seperate defines are really just for future convenience...
#define CSV_NAME "fd_data_ph.csv"
#define BLOCK_SIZE 256
#define GRID_SIZE 256
#define SEP ";"
#define DEBUG
#define MAX_POW 9
#define NUM_TESTS 3

/* ----------------------- FROM CG --------------------- */
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

/* ----------------------- FROM CG --------------------- */

/* ----------------------- FROM EX5.1 --------------------- */

__global__ void scan_kernel_1(int const *X,
                              int *Y,
                              int N,
                              int *carries)
{
  __shared__ int shared_buffer[GRID_SIZE];
  int my_value;

  int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  int block_offset = 0;

  // run scan on each section
  for (int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;

    // inclusive scan in shared buffer:
    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();

    // exclusive scan requires us to write a zero value at the beginning of each block
    my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;

    block_offset += shared_buffer[blockDim.x-1];
  }

  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;

}

__global__ void scan_kernel_1_inc(int const *X,
  int *Y,
  int N,
  int *carries)
{
__shared__ int shared_buffer[GRID_SIZE];
int my_value;

int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
int block_start = work_per_thread * blockDim.x *  blockIdx.x;
int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
int block_offset = 0;

// run scan on each section
for (int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
{
// load data:
my_value = (i < N) ? X[i] : 0;

// inclusive scan in shared buffer:
for(int stride = 1; stride < blockDim.x; stride *= 2)
{
__syncthreads();
shared_buffer[threadIdx.x] = my_value;
__syncthreads();
if (threadIdx.x >= stride)
my_value += shared_buffer[threadIdx.x - stride];
}
__syncthreads();
shared_buffer[threadIdx.x] = my_value;
__syncthreads();

// my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
my_value = shared_buffer[threadIdx.x];

// write to output array
if (i < N)
Y[i] = block_offset + my_value;

block_offset += shared_buffer[blockDim.x-1];
}

// write carry:
if (threadIdx.x == 0)
carries[blockIdx.x] = block_offset;

}

// exclusive-scan of carries
__global__ void scan_kernel_2(int *carries)
{
  __shared__ int shared_buffer[GRID_SIZE];

  // load data:
  int my_carry = carries[threadIdx.x];

  // exclusive scan in shared buffer:

  for(int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();
    if (threadIdx.x >= stride)
      my_carry += shared_buffer[threadIdx.x - stride];
  }
  __syncthreads();
  shared_buffer[threadIdx.x] = my_carry;
  __syncthreads();

  // // write to output array
  carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

  // // write to output array
  // carries[threadIdx.x] = shared_buffer[threadIdx.x];
}

__global__ void scan_kernel_3(int *Y, int N,
                              int const *carries)
{
  int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ int shared_offset;

  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];

  __syncthreads();

  // add offset to each element in the block:
  for (int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}

/* vectorAddInPlace

A helper function. Also helps with Ex5.1 in adapting the given exclusive scan to an inclusive one.*/
__global__ void vectorAddInPlace(int* x, const int* y, const size_t N) {
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N; tid += blockDim.x * gridDim.x)
    x[tid] += y[tid];
}

/** exclusive_scan

The given one.*/
void exclusive_scan(int const * input,
                    int * output, int N)
{
  int *carries;
  cudaMalloc(&carries, sizeof(int) * GRID_SIZE);

  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N, carries);

  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, GRID_SIZE>>>(carries);

  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<GRID_SIZE, BLOCK_SIZE>>>(output, N, carries);

  cudaFree(carries);
}

/** inclusive_scan

The worse one.*/
void inclusive_scan(int const * input,
                     int * output, int N)
{
  exclusive_scan(input, output, N);

  // To make inclusive
  vectorAddInPlace<<<GRID_SIZE, BLOCK_SIZE>>>(output, input, N);
}

/** inclusive_scan2

The good one.*/
void inclusive_scan2(int const * input,
                     int * output, int N)
{
  int *carries;
  cudaMalloc(&carries, sizeof(int) * GRID_SIZE);

  // First step: Scan within each thread group and write carries
  scan_kernel_1_inc<<<GRID_SIZE, BLOCK_SIZE>>>(input, output, N, carries);

  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, GRID_SIZE>>>(carries);

  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<GRID_SIZE, BLOCK_SIZE>>>(output, N, carries);

  cudaFree(carries);
}
/* ----------------------- FROM EX5.1 --------------------- */

// The above is taken from my source for task 1 and modified to work with uints instead, since the row offsets are ints

/** calc_nz_entries

Adaption of scheme given in lecture 5.*/
__global__ void calc_nz_entries (int* row_offsets, int N, int M) {
  for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < N * M; row += gridDim.x * blockDim.x) {
      int nz_for_this_row = 1;
      int i = row / N;
      int j = row % N;

      if(i > 0) 
        nz_for_this_row += 1;
      if(j > 0) 
        nz_for_this_row += 1;
      if(i < N-1) 
        nz_for_this_row += 1;
      if(j < M-1) 
        nz_for_this_row += 1;
      
      row_offsets[row] = nz_for_this_row;
  }
}

/** generate_values

Adaption of the assembly template given in lecture 5 */
__global__ void generate_values(double* values, int* csr_cols, int* row_offsets, int N, int M) {
  for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < N*M; row += gridDim.x * blockDim.x) 
  {
    int i = row / N;
    int j = row % N;
    int cnt = 0;
    
    if (i > 0) {
        values[row_offsets[row] + cnt] = -1.;
        csr_cols[row_offsets[row] + cnt] = (i-1)*N + j;
        ++cnt;
    }

    if (j > 0) {
      values[row_offsets[row] + cnt] = -1.;
      csr_cols[row_offsets[row] + cnt] = i*N + (j-1);
      ++cnt;
    }

    // diag always has value here
    values[row_offsets[row] + cnt] = 4.;
    csr_cols[row_offsets[row] + cnt] = i*N + j;
    ++cnt;

    if (i < N-1) {
        values[row_offsets[row] + cnt] = -1.;
        csr_cols[row_offsets[row] + cnt] = (i+1)*N + j;
        ++cnt;
    }
    
    if (j < M-1) {
        values[row_offsets[row] + cnt] = -1.;
        csr_cols[row_offsets[row] + cnt] = i*N + (j+1);
        ++cnt;
    }
  }
}

void assembleA_gpu(
    const int N, const int M, int& unknowns,
    int* row_offsets, int* columns, double* values) {

  int numberOfValues;
  int* row_offsets2;
  int size = (N*M+1);
  cudaMalloc(&row_offsets, sizeof(int) * (size));
  cudaMalloc(&row_offsets2, sizeof(int) * (size));

  // Perform the calculations
  calc_nz_entries<<<GRID_SIZE, BLOCK_SIZE>>>(row_offsets2, N, M);
  // inclusive_scan(row_offsets2, row_offsets, size);
  exclusive_scan(row_offsets2, row_offsets, size);

#ifdef DEBUG

  int* check = (int*)malloc(sizeof(int) * size); 
  int *check2 = (int*)malloc(sizeof(int) * size);
  cudaMemcpy(check, row_offsets, sizeof(int)* (size), cudaMemcpyDeviceToHost);
  cudaMemcpy(check2, row_offsets2, sizeof(int)* (size), cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) 
      std::cout << check[i] << " | ";
  std::cout << std::endl;
  for (int i = 0; i < size; ++i) 
      std::cout << check2[i] << " | ";
  std::cout << std::endl;

#endif
  // Now we can malloc values and cols
  cudaMemcpy(&numberOfValues, row_offsets+(size-1), sizeof(int), cudaMemcpyDeviceToHost);
  numberOfValues += 2;
#ifdef DEBUG
  if (unknowns != numberOfValues) {
    std::cout << "Something wrong with unknowns! : " << unknowns << " vs " << numberOfValues << std::endl;
    unknowns = numberOfValues;
    std::cout << "Replaced unknowns with: " << unknowns << std::endl;
  }
#endif
  cudaFree(row_offsets2); // memory efficiency!
  cudaMalloc(&columns, sizeof(int) * numberOfValues);
  cudaMalloc(&values, sizeof(double) * numberOfValues);

  // populate values and columns
  generate_values<<<GRID_SIZE, BLOCK_SIZE>>>(values, columns, row_offsets, N, M);
#ifdef DEBUG 
  std::cout << "Done with assembly on GPU!" << std::endl;
#endif
}


/** median

Calculates median of a vector.*/
template <typename T>
double median(std::vector<T>& vec)
{
  // modified taken from here: https://stackoverflow.com/questions/2114797/compute-median-of-values-stored-in-vector-c

  size_t size = vec.size();

  if (size == 0)
    return 0.;

  std::sort(vec.begin(), vec.end());

  size_t mid = size/2;

  return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
}


/** solve_system

Solve a system with `N * M` unknowns.

As provided for Ex4 + adaptions done during Ex4.
EDIT: Properly allocate index arrows as int now.
 */
 void solve_system(int N, int M) {

  Timer timer;
  const int size = N*M * N*M; 
  int unknowns = N*M*3 - 2;// number of unknows to solve for --> to check
#ifdef DEBUG
  std::cout << "Solving Ax=b with NxM= " << N << "x" << N << std::endl;
#endif
  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix
  // a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros
  //       per row in the system matrix, so we can allocate accordingly.
  //


  // This may be a bad coding sytle - declaring pointers here and allocating in the function assembleA_gpu below - but it's convenient here.
  int *cuda_csr_rowoffsets, *cuda_csr_colindices;
  double *cuda_csr_values;

  timer.reset();
  assembleA_gpu(N, M, unknowns, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values); 
  double time_assemble_gpu = timer.get();

#ifdef DEBUG
  std::cout << "Unknowns: " << unknowns << std::endl;
#endif
  
  //
  // fill CSR matrix with values
  //
  // We measure the assembly + copy times, since we want to run a CG on the GPU.
  // The filled temps are immediatly discarded and only the ones generated on the GPU are used for the CG --> check if they are correct!

  int* cuda_rowtemps;
  int* cuda_coltemps;
  double* cuda_valtemps;
  cudaMalloc(&cuda_rowtemps, sizeof(int) * (size + 1));
  cudaMalloc(&cuda_coltemps, sizeof(int) * unknowns);
  cudaMalloc(&cuda_valtemps, sizeof(double) * unknowns);

  int *csr_rowoffsets = (int *)malloc(sizeof(int) * (size + 1));
  int *csr_colindices = (int *)malloc(sizeof(int) * unknowns);
  double *csr_values = (double *)malloc(sizeof(double) * unknowns);

#ifdef DEBUG
  std::cout << "generate CPU "<<std::endl;
#endif
  timer.reset();
  generate_fdm_laplace(unknowns, csr_rowoffsets, csr_colindices,
                      csr_values);
  cudaMemcpy(cuda_rowtemps, csr_rowoffsets, sizeof(int) * (size + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_coltemps, csr_colindices, sizeof(int) * unknowns,   cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_valtemps, csr_values, sizeof(double) * unknowns,   cudaMemcpyHostToDevice);
  double time_assemble_cpu = timer.get();

  cudaFree(cuda_rowtemps);
  cudaFree(cuda_coltemps);
  cudaFree(cuda_valtemps);
  #ifdef DEBUG
  std::cout << "fill solution "<<std::endl;
#endif

  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double *)malloc(sizeof(double) * size);
  double *rhs = (double *)malloc(sizeof(double) * size);
  std::fill(rhs, rhs + unknowns, 1);

  
  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  #ifdef DEBUG
  std::cout << "CG "<<std::endl;
#endif
  timer.reset();
  int iters = conjugate_gradient(size, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);
  double runtime = timer.get();

    //
  // Check for convergence:
  //
#ifdef DEBUG
  std::cout << "check convergence "<<std::endl;
#endif
  double residual_norm = relative_residual(size, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);

#ifdef DEBUG
  std::cout << "Time elapsed: " << runtime << " (" << runtime / iters << " per iteration)" << std::endl;
  std::cout << "Relative residual norm: " << residual_norm
          << " (should be smaller than 1e-6)" << std::endl;
#endif
#ifndef DEBUG
  std::fstream csv;
  std::string csv_name = CSV_NAME;
  csv.open (csv_name, std::fstream::out | std::fstream::app);
  csv << N << SEP << M << SEP
    << unknowns << SEP
    << time_assemble_cpu << SEP
    << time_assemble_gpu << SEP
    << runtime << SEP
    << iters << SEP
    << residual_norm << std::endl;
  csv.close();
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

int main() 
{
  std::string csv_name = CSV_NAME;
  std::cout << "\n\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex5/" + csv_name << std::endl;
  std::vector<size_t> p_per_dir{(size_t)5}; //{ (size_t)sqrt(1e3), (size_t)sqrt(1e4), (size_t)sqrt(1e5), (size_t)sqrt(1e6), (size_t)sqrt(4e6)};

  std::string header = "N;M;unknowns;times_assemble_cpu;times_assemble_gpu;times_cg;iters;norm_after";
  std::fstream csv;
  csv.open (csv_name, std::fstream::out | std::fstream::app);
  csv << header << std::endl;
  csv.close();

  for (auto& points: p_per_dir)
    solve_system(points, points); // solves a system with 100*100 unknowns

  std::cout << "\n\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex5/" + csv_name << std::endl;
  return EXIT_SUCCESS;
}


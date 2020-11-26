#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
 
 // Block and grid size defines.
// Seperate defines are really just for future convenience...
#define CSV_NAME "fd_data_ph.csv"
#define BLOCK_SIZE 256
#define GRID_SIZE 256
#define SEP ";"
// #define DEBUG
 
template <typename T>
void printContainer(T container, const int size, const int only) {
  if (only){
    for (int i = 0; i < only; ++i) 
        std::cout << container[i] << " | ";
    std::cout << " ... ";
    for (int i = size - only; i < size; ++i) 
      std::cout << container[i] << " | ";
  }
  else {
    for (int i = 0; i < size; ++i) 
        std::cout << container[i] << " | ";
  }
  std::cout << std::endl;
}
 
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
 
__global__ void scan_kernel_1(int const *X,
                              int *Y,
                              int N,
                              int *carries)
{
  __shared__ int shared_buffer[BLOCK_SIZE];
  int my_value;
 
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;
 
  // run scan on each section
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;
 
    // inclusive scan in shared buffer:
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
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
 
// exclusive-scan of carries
__global__ void scan_kernel_2(int *carries)
{
  __shared__ int shared_buffer[BLOCK_SIZE];
 
  // load data:
  int my_carry = carries[threadIdx.x];
 
  // exclusive scan in shared buffer:
 
  for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
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
 
  // write to output array
  carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
}
 
__global__ void scan_kernel_3(int *Y, int N,
                              int const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
 
  __shared__ int shared_offset;
 
  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];
 
  __syncthreads();
 
  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}

__global__ void count_nz(int* row_offsets, int N, int M) {
    for(int row = blockDim.x * blockIdx.x + threadIdx.x; row < N * M; row += gridDim.x * blockDim.x) {
        int nz_for_this_row = 1;
        int i = row / N;
        int j = row % N;

        if(i > 0) nz_for_this_row += 1;
        if(j > 0) nz_for_this_row += 1;
        if(i < N-1) nz_for_this_row += 1;
        if(j < M-1) nz_for_this_row += 1;
        
        row_offsets[row] = nz_for_this_row;
    }
}


__global__ void assembleA(double* values, int* columns, int* row_offsets, int N, int M) {
    for(int row = blockDim.x * blockIdx.x + threadIdx.x; row < N*M; row += gridDim.x * blockDim.x) {
        int i = row / N;
        int j = row % N;
        int counter = 0;

        if ( i > 0) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = (i-1)*N+j;
            counter++;
        }
        
        if ( j > 0) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = i*N+(j-1);
            counter++;
        }

        values[(int)row_offsets[row] + counter] = 4;
        columns[(int)row_offsets[row] + counter] = i*N+j;

        counter++;

        if ( j < M-1) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = i*N+(j+1);
            counter++;
        }
        if ( i < N-1) {
            values[(int)row_offsets[row] + counter] = -1;
            columns[(int)row_offsets[row] + counter] = (i+1)*N+j;
            counter++;
        }
    }
}

 
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
#endif
  if (iters > 10000)
    std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
    << std::endl;
    else
    std::cout << "Conjugate Gradient converged in " << iters << " iterations."
    << std::endl;

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
 
 
int main() {

  std::string csv_name = CSV_NAME;
  std::cout << "\n\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex5/" + csv_name << std::endl;
  std::string header = "N;M;unknowns;nz_found;times_assemble_cpu;times_assemble_gpu;times_cg;iters;norm_after";

  std::fstream csv;
  csv.open(csv_name, std::fstream::out | std::fstream::trunc);
  csv << header << std::endl;
  csv.close();

  Timer timer;


  std::vector<int> N_vec;
  for (int i = 2048; i <= 10000; i *= 2) {
    N_vec.push_back(i);
  }

  for (int& N: N_vec) {
    std::cout << "N = M = " << N << std::endl;
    int M = N;
    //
    // Allocate host arrays for reference
    //
    int *row_offsets = (int *)malloc(sizeof(int) * (N*M+1));
    
  
    //
    // Allocate CUDA-arrays
    //
    int *cuda_row_offsets;
    int *cuda_row_offsets_2;
    double *cuda_values;
    int *cuda_columns;

    cudaMalloc(&cuda_row_offsets, sizeof(int) * (N*M+1));
    cudaMalloc(&cuda_row_offsets_2, sizeof(int) * (N*M+1));
  
  
    // Perform the calculations
    int numberOfValues;
    timer.reset();
    count_nnz<<<GRID_SIZE, BLOCK_SIZE>>>(cuda_row_offsets_2, N, M);
    exclusive_scan(cuda_row_offsets_2, cuda_row_offsets, N*M+1);
    cudaMemcpy(row_offsets, cuda_row_offsets, sizeof(int) * (N*M+1), cudaMemcpyDeviceToHost);
    numberOfValues = row_offsets[N*M];

  #ifdef DEBUG
    printContainer(row_offsets, N*M+1, 4);
    std::cout << std::endl;
  #endif

    double *values = (double *)malloc(sizeof(double) * numberOfValues);
    int *columns = (int *)malloc(sizeof(int) * numberOfValues);
    cudaMalloc(&cuda_columns, sizeof(int) * numberOfValues);
    cudaMalloc(&cuda_values, sizeof(double) * numberOfValues);

    assembleA<<<GRID_SIZE, BLOCK_SIZE>>>(cuda_values, cuda_columns, cuda_row_offsets, N, M);
    double time_assemble_gpu = timer.get();

    cudaMemcpy(values, cuda_values, sizeof(double) * numberOfValues, cudaMemcpyDeviceToHost);
    cudaMemcpy(columns, cuda_columns, sizeof(int) * numberOfValues, cudaMemcpyDeviceToHost);

#ifdef DEBUG
    printContainer(values, numberOfValues, 4);
    std::cout << std::endl;
    printContainer(columns, numberOfValues, 4);
#endif
/* -------- CPU -----------*/
    int *csr_rowoffsets = (int *)malloc(sizeof(int) * (N*M+1));
    int *csr_colindices = (int *)malloc(sizeof(int) * (N*M+1)*5);
    double *csr_values = (double *)malloc(sizeof(double) * (N*M+1)*5);
  
  #ifdef DEBUG
    std::cout << "generate CPU "<<std::endl;
  #endif
    timer.reset();
    generate_fdm_laplace(N, csr_rowoffsets, csr_colindices, csr_values);
    double time_assemble_cpu = timer.get();

/* -------- CPU -----------*/

      //
    // Allocate solution vector and right hand side:
    //
    double *solution = (double *)malloc(sizeof(double) * N*M);
    double *rhs = (double *)malloc(sizeof(double) * N*M);
    std::fill(rhs, rhs + N*M, 1.);

    timer.reset();
    int iters = conjugate_gradient(N*M, cuda_row_offsets, cuda_columns, cuda_values, rhs, solution);
    double runtime = timer.get();
  #ifdef DEBUG
    std::cout << "runtime: " << runtime << std::endl;
  #endif
    double residual_norm = relative_residual(N*M, row_offsets, columns, values, rhs, solution);

  #ifndef DEBUG

    csv.open (csv_name, std::fstream::out | std::fstream::app);
    csv << N << SEP << M << SEP
      << N*M << SEP
      << numberOfValues << SEP
      << time_assemble_cpu << SEP
      << time_assemble_gpu << SEP
      << runtime << SEP
      << iters << SEP
      << residual_norm << std::endl;
    csv.close();
  #endif
  
    //
    // Clean up:
    //
    free(row_offsets);
    free(values);
    free(columns);
    free(csr_rowoffsets);
    free(csr_colindices);
    free(csr_values);
    cudaFree(cuda_row_offsets);
    cudaFree(cuda_row_offsets_2);
    cudaFree(cuda_values);
    cudaFree(cuda_columns);
  }

  std::cout << "\n\nResults in csv form can be found here\nhttps://gtx1080.360252.org/2020/ex5/" + csv_name << std::endl;
  return EXIT_SUCCESS;
}
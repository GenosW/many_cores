
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

  int cg(void)
  {
    ...

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

  ....


  }
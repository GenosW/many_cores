/** atomicMax for double
 * 
 * References:
 * (1) https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomicmax
 * (2) https://www.micc.unifi.it/bertini/download/gpu-programming-basics/2017/gpu_cuda_5.pdf
 * (3) https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
 */
 __device__ void atomicMax(double* address, double val){    
    unsigned long long int* address_as_ull = (unsigned long long int*) address; 
    unsigned long long int old = *address_as_ull, assumed;
    do  {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
  }
  
  /** atomicMin for double
   */
  __device__ void atomicMin(double* address, double val){    
    unsigned long long int* address_as_ull = (unsigned long long int*) address; 
    unsigned long long int old = *address_as_ull, assumed;
    do  {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
  }

/** analyze_x_shared
 * 
 * result[0] = sum;
 * result[1] = abs_sum;
 * result[2] = sqr_sum;
 * result[3] = mod_max;
 * result[4] = min;
 * result[5] = max;
 * result[6] = z_entries;
 */
// template <int block_size=BLOCK_SIZE>
__global__ void analyze_x_shared(const int N, double *x, double *results) {
    if (blockDim.x * blockIdx.x < N) {
      int tid = threadIdx.x + blockDim.x * blockIdx.x; // global tid
      const int stride = blockDim.x * gridDim.x;
  
      __shared__ double cache[7][BLOCK_SIZE];
  
      double sum = 0.0, abs_sum = 0.0, sqr_sum = 0.0;
      // double mod_max = 0.0;
      double max = x[0];
      double min = max;
      double z_entries = 0;
      for (; tid < N; tid += stride) {
        double value = x[tid];
        sum += value;
        abs_sum += std::abs(value);
        sqr_sum += value*value;
  
        // mod_max = (std::abs(value) > mod_max)? value : mod_max;
        min = fmin(value, min); 
        max = fmax(value, max);
        z_entries += (value)? 0 : 1;
      }
      tid = threadIdx.x; // block tid 
      cache[0][tid] = sum;
      cache[1][tid] = abs_sum;
      cache[2][tid] = sqr_sum;
      cache[3][tid] = fmax(std::abs(min), max);
      cache[4][tid] = min;
      cache[5][tid] = max;
      cache[6][tid] = z_entries;
  
      __syncthreads();
      for (int i = blockDim.x / 2; i != 0; i /= 2) {
        __syncthreads();
        if (tid < i) { // lower half does smth, rest idles
          // sums
          cache[0][tid] += cache[0][tid + i]; 
          cache[1][tid] += cache[1][tid + i]; 
          cache[2][tid] += cache[2][tid + i]; 
          // min/max values
          cache[3][tid] = fmax(cache[3][tid + i], cache[3][tid]); // already all values are std::abs(...)
          cache[4][tid] = fmin(cache[4][tid + i], cache[4][tid]); 
          cache[5][tid] = fmax(cache[5][tid + i], cache[5][tid]); 
  
          // "sum"
          cache[6][tid] += cache[6][tid + i]; 
        }
      }
  
      if (tid == 0) // cache[0] now contains block_sum
      {
        atomicAdd(results, cache[0][0]);
        atomicAdd(results+1, cache[1][0]);
        atomicAdd(results+2, cache[2][0]);
  
        // Ideally...
        atomicMax(results+3, cache[3][0]);
        atomicMin(results+4, cache[4][0]);
        atomicMax(results+5, cache[5][0]);
  
        atomicAdd(results+6, cache[6][0]);
      }
    }
  }
  
  /** analyze_x_shared
   * 
   * result[0] = sum;
   * result[1] = abs_sum;
   * result[2] = sqr_sum;
   * result[3] = mod_max;
   * result[4] = min;
   * result[5] = max;
   * result[6] = z_entries;
   */
  __global__ void analyze_x_warp(const int N, double *x, double *results) {
    if (blockDim.x * blockIdx.x < N) {
      int tid = threadIdx.x + blockDim.x * blockIdx.x; // global tid
      const int stride = blockDim.x * gridDim.x;
  
      double sum = 0.0, abs_sum = 0.0, sqr_sum = 0.0;
      // double mod_max = 0.0;
      double max = x[0];
      double min = max;
      int z_entries = 0;
      for (; tid < N; tid += stride) {
        double value = x[tid];
        sum += value;
        abs_sum += std::abs(value);
        sqr_sum += value*value;
  
        min = fmin(value, min); 
        max = fmax(value, max);
        z_entries += (value)? 0 : 1;
      }
      tid = threadIdx.x; // block tid 
      double mod_max = fmax(std::abs(min), max);
  
      __syncthreads();
      for (int i = warpSize / 2; i != 0; i /= 2) {
        //__syncthreads();
        sum += __shfl_down_sync(0xffffffff, sum, i);
        abs_sum += __shfl_down_sync(0xffffffff, abs_sum, i);
        sqr_sum += __shfl_down_sync(0xffffffff, sqr_sum, i);
  
        double tmp = __shfl_down_sync(0xffffffff, mod_max, i);
        mod_max = fmax(tmp, mod_max);
        tmp = __shfl_down_sync(0xffffffff, min, i);
        min = fmin(tmp, min);
        tmp = __shfl_down_sync(0xffffffff, max, i);
        max = fmax(tmp, max) ;
  
        z_entries += __shfl_down_sync(0xffffffff, z_entries, i);
      }
      // for (int i = blockDim.x / 2; i != 0; i /= 2) {
      // for (int i = warpSize / 2; i != 0; i /= 2) {
      //   //__syncthreads();
      //   sum += __shfl_xor_sync(-1, sum, i);
      //   abs_sum += __shfl_xor_sync(-1, abs_sum, i);
      //   sqr_sum += __shfl_xor_sync(-1, sqr_sum, i);
  
      //   double tmp = __shfl_xor_sync(-1, mod_max, i);
      //   mod_max = (tmp > mod_max) ? tmp : mod_max;
      //   tmp = __shfl_xor_sync(-1, min, i);
      //   min = (tmp < min) ? tmp : min;
      //   tmp = __shfl_xor_sync(-1, max, i);
      //   max = (tmp > max) ? tmp : max;
  
      //   z_entries += __shfl_xor_sync(-1, z_entries, i);
      // }
  
      if (tid % warpSize == 0) // a block can consist of multiple warps
      {
        atomicAdd(results, sum);
        atomicAdd(results+1, abs_sum);
        atomicAdd(results+2, sqr_sum);
  
        atomicMax(results+3, mod_max);
        atomicMin(results+4, min);
        atomicMax(results+5, max);
  
        atomicAdd(results+6, z_entries);
      }
    }
  }
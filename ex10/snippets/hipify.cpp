/** I figured I could extend the STRINGIFY macro approach to 
 * also work for hip. I also simplified the define names a bit.
 * In the end, I didn't test it because I feared it would take too
 * much additional time - finite time is a b*tch - and just used
 * it as a list of keywords to change to hipify by hand. */

#define HIP
// #define CUDA
// #define OCL

#ifdef HIP
  #define HIP_ASSERT(x) (assert((x)==hipSuccess))
  #undef CUDA
  #undef OCL
#endif

#ifdef CUDA
  #undef HIP
  #undef OCL
#endif

#ifdef OCL
  #undef CUDA
  #undef HIP
#endif

#ifdef HIP
  //
  // Transformation to HIP
  //
  // -- runtime
  #define DEVICE_ALLOC hipMalloc
  #define DEVICE_FREE hipFree

  // -- kernel language
  #define KERNEL __global__
  #define GLOBALMEM 
  #define LOCALMEM __shared__
  #define DEVICE_FUNC __device__
  #define BARRIER __syncthreads
  #define ATOMIC_ADD atomicAdd

  // -- thread management
  #define LOCALID hipThreadIdx_x
  #define GROUPID hipBlockIdx_x
  #define GLOBALID ((hipBlockDim_x * hipBlockIdx_x) + hipThreadIdx_x)

  #define LOCALSIZE hipBlockDim_x
  #define GLOBALSIZE (hipGridDim_x * hipBlockDim_x)
#endif

#ifdef CUDA
  //
  // Transformation to CUDA
  //
  // -- runtime
  #define DEVICE_ALLOC cudaMalloc
  #define DEVICE_FREE cudaFree

  // -- kernel language
  #define KERNEL __global__
  #define GLOBALMEM 
  #define LOCALMEM __shared__
  #define DEVICE_FUNC __device__
  #define BARRIER __syncthreads
  #define ATOMIC_ADD atomicAdd

  // -- thread management
  #define LOCALID threadIdx.x
  #define GROUPID blockIdx.x
  #define GLOBALID ((blockDim.x * blockIdx.x) + threadIdx.x)

  #define LOCALSIZE blockDim.x
  #define GLOBALSIZE (gridDim.x * blockDim.x)
#endif

// entry point, but need to account for multiple arguments AND need to actually force replacement before applying the macro
#define STRINGIFY(...) mFn2(__VA_ARGS__)
#define mFn2(ANS) #ANS
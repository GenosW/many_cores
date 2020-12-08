#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
 
void my_atomic_add(volatile __global double *p, double val) {
  volatile __global ulong* address_as_ul = (volatile __global ulong *) p; 
  volatile ulong old = *address_as_ul, assumed;
  ulong val_as_ul =  (ulong) val;
  do  {
    assumed = old;
    old = atomic_add(address_as_ul, val_as_ul)
  } while (assumed != old);
};

__kernel void vec_add(__global double *x, __global double *y, unsigned int N) {
  for (unsigned int i = get_global_id(0); i < N; i += get_global_size(0))
    x[i] += y[i];
};

__kernel void xDOTy(__global double *result, 
                    __global double *x,
                    __global double *y, 
                    __local cache, uint N) {
  uint gid = get_global_id(0);
  uint lid = get_local_id(0);
  uint stride = get_global_size(0);
  __local double cache[128];
  double tmp = 0.0;
  for (uint i = gid; i < N; i += stride)
    tmp += x[i] * y[i];
  cache[lid] = tmp;
  
  for (int i = get_local_size(0) / 2; i > 0; i /= 2) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < i)
      cache[lid] += cache[lid + i];
  }
  if (lid==0)
    result[get_group_id(0)] = cache[lid];
};



// __kernel void xDOTy(__global atomic_double *result, __global double *x,
//                     __global double *y, uint N) {
//   uint gid = get_global_id(0);
//   uint lid = get_local_id(0);
//   uint stride = get_global_size(0);
//   double dot = 0.0;
//   for (uint i = tid; i < N; i += stride)
//     dot += x[i] * y[i];

//   // need to add to signature: __local double *cache,
//   // for (int i = get_local_size(0) / 2; i > 0; i /= 2) {
//   //   barrier(CLK_LOCAL_MEM_FENCE);
//   //   if (lid < i)
//   //     cache[lid] += cache[lid + i];
//   // }
//   double val = work_group_reduce_add(gid < N ? dot : 0.);

//   if (lid == 0) {
//     atomic_add(result, val);
//   }
//   // // atomic flag version
//   // // need to add: __global volatile atomic_flag *lock,
//   // if (lid == 0) {
//   //   while (!atomic_flag_test_and_set(lock)){}
//   //   *result += cache[0];
//   //   atomic_flag_clear(lock;)
//   // }
// };

// __kernel void xDOTy(__global atomic_double *result, __global double *x,
//                     __global double *y, __local double *cache,
//                     __global bool *lock, uint N) {
//   uint gid = get_global_id(0);
//   uint lid = get_local_id(0);
//   uint stride = get_global_size(0);
//   double dot = 0.0;
//   for (uint i = tid; i < N; i += stride)
//     dot += x[i] * y[i];
//   #cache[lid] = dot;

//   // for (int i = get_local_size(0) / 2; i > 0; i /= 2) {
//   //   barrier(CLK_LOCAL_MEM_FENCE);
//   //   if (lid < i)
//   //     cache[lid] += cache[lid + i];
//   // }
//   double val = work_group_reduce_max( gid < length ? input[globalId] : -INFINITY);

//   if (lid == 0) {
//     while (*lock) {}

//     *result += cache[0];
//   }
// };

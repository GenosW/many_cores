#pragma OPENCL EXTENSION cl_khr_fp64 : enable
 
__kernel void ocl_csr_matvec(uint N,
                        __global int *csr_rowoffsets,
                        __global int *csr_colindices, 
                        __global double *csr_values,
                        __global double const *x, __global double *y)
{
  uint gid = get_global_id(0);
  uint stride = get_global_size(0);
  for (size_t i=gid; i<N; i+=stride) {
    double value = 0;
    for (size_t j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; ++j)
      value += csr_values[j] * x[csr_colindices[j]];
    y[i] = value;
  }
};
 
__kernel void ocl_csr_matvec_fast(uint N,
                        __global int *csr_rowoffsets,
                        __global int *csr_colindices, 
                        __global double *csr_values,
                        __global double const *x, __global double *y)
{
  __local double cache[128];
  uint gid = get_group_id(0);
  uint lid = get_local_id(0);
  uint stride = get_num_groups(0);
  uint j_stride = get_global_size(0);
  for (size_t i=gid; i<N; i+=stride) {
    double value = 0;
    for (size_t j=csr_rowoffsets[i]; j<csr_rowoffsets[i+1]; j+=j_stride)
      value += csr_values[j] * x[csr_colindices[j]];
    cache[lid] = value;
    for (int i = get_local_size(0) / 2; i > 0; i /= 2) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid < i)
        cache[lid] += cache[lid + i];
    }
    if (lid==0)
      y[gid] = cache[0];
  }
};;
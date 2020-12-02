#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// required to enable 'double' inside OpenCL programs

__kernel void vec_add(__global double *x,
                      __global double *y,
                      unsigned int N)
{
  for (unsigned int i  = get_global_id(0);
                    i  < N;
                    i += get_global_size(0))
    x[i] += y[i];
};
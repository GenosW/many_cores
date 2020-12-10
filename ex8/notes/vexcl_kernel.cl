#if defined(cl_khr_fp64)
# pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
# pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif

double SUM_double
(
double prm1,
double prm2
)
{
return prm1 + prm2;
}
kernel void vexcl_reductor_kernel
(
ulong n,
global double * prm_1,
global double * prm_2,
global double * prm_3,
global double * prm_4,
global double * g_odata,
local double * smem
)
{
double mySum = 0;
for(ulong idx = get_global_id(0); idx < n; idx += get_global_size(0))
{
mySum = SUM_double(mySum, ( ( prm_1[idx] + prm_2[idx] ) * ( prm_3[idx] - prm_4[idx] ) ));
}
local double * sdata = smem;
size_t tid = get_local_id(0);
size_t block_size = get_local_size(0);
sdata[tid] = mySum;
barrier(CLK_LOCAL_MEM_FENCE);
if (block_size >= 1024)
{
if (tid < 512) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 512]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 512)
{
if (tid < 256) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 256]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 256)
{
if (tid < 128) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 128]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 128)
{
if (tid < 64) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 64]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 64)
{
if (tid < 32) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 32]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 32)
{
if (tid < 16) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 16]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 16)
{
if (tid < 8) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 8]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 8)
{
if (tid < 4) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 4]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 4)
{
if (tid < 2) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 2]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (block_size >= 2)
{
if (tid < 1) { sdata[tid] = mySum = SUM_double(mySum, sdata[tid + 1]); }
barrier(CLK_LOCAL_MEM_FENCE);
}
if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}
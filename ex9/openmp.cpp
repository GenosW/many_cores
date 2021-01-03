#include <omp.h>
#include <cmath>
#include <iostream>

int main()
{
   int N = 10000000;
   double *data_array = (double*)malloc(sizeof(double) * N);

   for (size_t i=0; i<N; ++i) data_array[i] = i;

   double red = 0;

   #pragma omp target teams distribute parallel for map(to: data_array[0:N]) map(tofrom: red) reduction(+:red)
   for (int idx = 0; idx < N; ++idx)
   {
       red += data_array[idx];
   }   

   std::cout << "Reduction result: " << red << std::endl;
   std::cout << "Expected result: " << (N * (N+1.0)) / 2.0 << std::endl;

   return EXIT_SUCCESS;
}

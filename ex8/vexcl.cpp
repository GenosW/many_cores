#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "timer.hpp"
// VexCL
#include <stdexcept>
// #define VEXCL_SHOW_KERNELS
// #define VEXCL_BACKEND_OPENCL // default
#define VEXCL_BACKEND_COMPUTE
// #define VEXCL_BACKEND_CUDA
#include <vexcl/vexcl.hpp>

// DEFINES
#define EX "ex8"
#ifdef VEXCL_BACKEND_OPENCL
    #define CSV_NAME "ph_data_vexcl_ocl.csv"
#endif
#ifdef VEXCL_BACKEND_COMPUTE
    #define CSV_NAME "ph_data_vexcl_ocl2.csv"
#endif
#ifdef VEXCL_BACKEND_CUDA
    #define CSV_NAME "ph_data_vexcl_ocl3.csv"
#endif

#define COUT
#define NUM_TEST 5
#define N_MIN 10
#define N_MAX 10000000 //1e8
//
//----------- Helper functions
//
template <template <typename, typename> class Container,
          typename ValueType,
          typename Allocator = std::allocator<ValueType>>
double median(Container<ValueType, Allocator> data)
{
    size_t size = data.size();
    if (size == 0)
        return 0.;
    sort(data.begin(), data.end());
    size_t mid = size / 2;

    return size % 2 == 0 ? (data[mid] + data[mid - 1]) / 2 : data[mid];
};

template <typename T>
double median(T *array, size_t size)
{
    if (size == 0)
        return 0.;
    sort(array, array + size);
    size_t mid = size / 2;

    return size % 2 == 0 ? (array[mid] + array[mid - 1]) / 2 : array[mid];
};
//
//----------- functions for this program
//

double benchmark(vex::Context ctx, size_t N, double x_init, double y_init, std::vector<double>& results)
{
    Timer timer;
    timer.reset();
    std::vector<double> x(N, x_init);
    std::vector<double> y(N, y_init);
    vex::vector<double> X(ctx, x);
    vex::vector<double> Y(ctx, y);

    vex::Reductor<double, vex::SUM> DOT(ctx);
    results[0] = timer.get();

    double dot;

    std::vector<double> tmp(NUM_TEST, 0.0);
    for (int iter = 0; iter < NUM_TEST; iter++) {
        timer.reset(); 
        dot = DOT( (X+Y)*(X-Y) );
        tmp[iter] = timer.get();
    }
    results[1] = median(tmp);

    double true_dot = (x_init + y_init) * (x_init - y_init) * N;


#ifdef COUT
    std::cout << "(x+y, x-y) = " << dot << " ?= " << true_dot << std::endl;
    std::cout << "Computation took " << results[1] << "s" << std::endl;
#endif
    timer.reset();
    results[3] = dot;
    results[2] = timer.get();

    return dot;
}

int main(int argc, char const *argv[])
{
     vex::Context ctx(vex::Filter::GPU&&vex::Filter::DoublePrecision);
    std::cout << ctx << std::endl; // print list of selected devices

    double x_init = 1., y_init = 2.;
    std::vector<double> results(4, 0.0);

    std::ofstream csv;
    std::string sep = ";";
    std::string header = "N;vec_init_time;dot_time;memcpy_time;dot_result";
    auto to_csv = [&csv, &sep] (double x) { csv << sep << x;};

    csv.open(CSV_NAME, std::fstream::out | std::fstream::trunc);
    csv << header << std::endl;
    for (size_t N = N_MIN; N < 1+N_MAX; N*=10){
#ifdef COUT
        std::cout << "N: " <<  N << std::endl;
#endif
        benchmark(ctx, N, x_init, y_init, results);
        csv << N;
        std::for_each(results.begin(), results.end(), to_csv);
        csv << std::endl;
    }

    std::cout << "Data: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;

    return EXIT_SUCCESS;
}

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "timer.hpp"
// ViennaCL
#define VIENNACL_WITH_CUDA
// #define VIENNACL_WITH_OPENCL
#include "viennacl/vector.hpp"
#include "viennacl/linalg/inner_prod.hpp"

// DEFINES
#define EX "ex8"
#define HOST_DOT
#ifdef VIENNACL_WITH_CUDA
    #define CSV_NAME "ph_data_viennacl_cuda.csv"
#endif
#ifdef VIENNACL_WITH_OPENCL
    #define CSV_NAME "ph_data_viennacl_ocl.csv"
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

double viennacl_benchmark(size_t N, double x_init, double y_init, std::vector<double>& results)
{
    Timer timer;
    timer.reset();
    viennacl::vector<double> x = viennacl::scalar_vector<double>(N, x_init);
    viennacl::vector<double> y = viennacl::scalar_vector<double>(N, y_init);
    results[0] = timer.get();

#ifndef HOST_DOT
    viennacl::scalar<double> dot;
#endif
#ifdef HOST_DOT
    double dot ;
#endif

    std::vector<double> tmp(NUM_TEST, 0.0);
    for (int iter = 0; iter < NUM_TEST; iter++) {
        timer.reset(); 
        dot = viennacl::linalg::inner_prod(x+y, x-y);
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
        viennacl_benchmark(N, x_init, y_init, results);
        csv << N;
        std::for_each(results.begin(), results.end(), to_csv);
        csv << std::endl;
    }

    std::cout << "Data: https://gtx1080.360252.org/2020/" << EX << "/" << CSV_NAME;

    return EXIT_SUCCESS;
}

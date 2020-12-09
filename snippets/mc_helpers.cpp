#include <memory> // for std::allocator
#include <vector>

template <typename T>
void printContainer(T container, const int size) {
  std::cout << container[0];
  for (int i = 1; i < size; ++i) 
    std::cout << " | " << container[i] ;
  std::cout << std::endl;
}

template <typename T>
void printContainer(T container, const int size, const int only) {
  std::cout << container[0];
  for (int i = 1; i < only; ++i) 
      std::cout  << " | " << container[i];
  std::cout << " | ...";
  for (int i = size - only; i < size; ++i) 
    std::cout  << " | " << container[i];
  std::cout << std::endl;
}

template <template <typename, typename> class Container, 
          typename ValueType,
          typename Allocator=std::allocator<ValueType> >
double median(Container<ValueType, Allocator> data)
{
  size_t size = data.size();
  if (start == end)
          return 0.;
  sort(data.begin(), data.end());
  size_t mid = size/2;

  return size % 2 == 0 ? (data[mid] + data[mid-1]) / 2 : data[mid];
};

template <typename T>
double median(T* vec, size_t size)
{
  size_t size = vec.size();
  if (size == 0)
          return 0.;
  sort(vec.begin(), vec.end());
  size_t mid = size/2;

  return size % 2 == 0 ? (vec[mid] + vec[mid-1]) / 2 : vec[mid];
};

bool check(const double* test, const double* ref, const size_t N) {
  for (size_t i = 0; i < N; ++i){
    if (test[i] != ref[i])
      return false;
  }
  return true;
}

double diff_norm(const double* test, const double* ref, const size_t N) {
  double norm = 0.0;
  for (size_t i = 0; i < N; ++i){
    norm += test[i] != ref[i];
  }
  return sqrt(norm);
}
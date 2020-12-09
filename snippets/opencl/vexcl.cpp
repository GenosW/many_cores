#include <iostream>
#include <stdexcept>
#include <vexcl/vexcl.hpp>

int main() {
 vex::Context ctx(vex::Filter::GPU&&vex::Filter::DoublePrecision);
 
 std::cout << ctx << std::endl; // print list of selected devices

 size_t N = 1000;
 std::vector<double> a(N, 1.0), b(N, 2.0);

 vex::vector<double> A(ctx, a);
 vex::vector<double> B(ctx, b);

 vex::vector<double> C = A + B;

 std::cout << C[0] << ", " << C[1] << ", ..." << std::endl;
 return 0; }

cmake_minimum_required(VERSION 3.0.0)
set( CMAKE_VERBOSE_MAKEFILE on )
project(ex7.2 VERSION 0.1.0)

message("Supported features = ${CMAKE_CXX_COMPILE_FEATURES}")
add_compile_options("-std=c++17" "-Wall" "-Wextra" "-Werror" "-v")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options("-ferror-limit=1")
else()
    add_compile_options("-fmax-errors=1")
endif()
include(CTest)
enable_testing()

# # Add the following lines, if you want to use libc++ instead:
# # (not sure if it works with g++)
# # https://releases.llvm.org/4.0.0/projects/libcxx/docs/UsingLibcxx.html
# add_compile_options("-stdlib=libc++")
# add_link_options("-v")
# add_link_options("-stdlib=libc++" "-lc++abi")
# add_executable(test_algorithms test_algorithms.cpp) # this line is for EX2.1
# add_executable(test_distribution test_distribution.cpp) # this line is for EX2.2
# add_executable(test_timer test_timer.cpp) # this line is for EX2.3

add_executable(ex7.2 matvec.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

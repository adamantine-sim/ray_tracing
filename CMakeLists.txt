cmake_minimum_required(VERSION 3.15)

project(ray_tracing LANGUAGES CXX VERSION 0.1.0)

find_package(Boost 1.67.0 REQUIRED COMPONENTS program_options)
find_package(Kokkos 3.1 REQUIRED)
find_package(ArborX 1.2 REQUIRED)

add_executable(ray_tracing ray_tracing.cc)
target_link_libraries(ray_tracing ArborX::ArborX Boost::program_options)

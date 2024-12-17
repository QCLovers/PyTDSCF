# CMakeLists.txt for scikit-build

cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Clang does not support std::hermite in <cmath>.
# In apple silicon device, gcc and g++ are aliased to clang and clang++!!!
set(CMAKE_C_COMPILER "gcc-14")
set(CMAKE_CXX_COMPILER "g++-14")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-O3 -Wall)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(basis._primints MODULE pytdscf/basis/_primints.cpp)
install(TARGETS basis._primints DESTINATION ${SKBUILD_PROJECT_NAME})

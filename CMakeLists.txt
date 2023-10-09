cmake_minimum_required(VERSION 3.16)


project(realrl)


find_package(Eigen3 3.3 REQUIRED)
find_package(Franka 0.8 REQUIRED)


add_subdirectory(third_party/pybind11)
add_subdirectory(third_party/ruckig)
add_subdirectory(third_party/affx)
# add_subdirectory(third_party/libfranka)


pybind11_add_module(_realrl src/python.cpp)
target_compile_features(_realrl PUBLIC cxx_std_17)
target_include_directories(_realrl PUBLIC include)
target_link_libraries(_realrl PUBLIC Franka::Franka ruckig::ruckig affx::affx Eigen3::Eigen)

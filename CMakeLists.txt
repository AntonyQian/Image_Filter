cmake_minimum_required(VERSION 3.20)
project(Image_Filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(imgfilter
    src/main.cpp
    src/io.cpp
)
target_include_directories(imgfilter PRIVATE third_party src)
target_link_libraries(imgfilter PRIVATE OpenMP::OpenMP_CXX)
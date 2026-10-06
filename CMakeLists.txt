cmake_minimum_required(VERSION 3.20)
project(Image_Filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release so a benchmark never accidentally runs unoptimized
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(imgfilter
    src/main.cpp
    src/io.cpp
    # add new .cpp files here
)
target_include_directories(imgfilter PRIVATE third_party src)
target_link_libraries(imgfilter PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(imgfilter PRIVATE -Wall -Wextra)   # GCC warnings
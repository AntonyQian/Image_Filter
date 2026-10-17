cmake_minimum_required(VERSION 3.20)
project(Image_Filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release so a benchmark never accidentally runs unoptimized
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The CUDA Backends. Off by default, so the laptop build (Windows, GCC) never
# needs nvcc. A CUDA build is meant for Linux with GCC as nvcc's host
# compiler; set the GPU's architecture too, for example for a T4:
#   cmake -S . -B build -DIMGFILTER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
option(IMGFILTER_CUDA "Build the cuda-naive and cuda-tiled Backends (needs nvcc)" OFF)
if(IMGFILTER_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
endif()

# Do NOT add -ffast-math, -Ofast (which turns fast-math on) or -march=native.
# Every Backend must produce the same Destination bit for bit. Fast-math lets
# the compiler reorder float additions, and -march=native lets it fuse a
# multiply and an add into one FMA instruction that rounds differently and
# depends on the CPU the build ran on. Either could make two builds, or two
# loops in the same build, disagree in the last bit.
#
# -ffp-contract=off goes further and forbids that fusion outright, so the
# Reference is plain multiply-then-add even on a CPU whose base instruction
# set has FMA (ARM, say). The GPU side gets the same rule: --fmad=false.
set(cpu_fp_flags -ffp-contract=off)
set(cuda_fp_flags --fmad=false)
find_package(OpenMP REQUIRED)

# Everything except main() lives in this static library, so the command-line
# tool and the test executable link the exact same code.
add_library(imgfilter_core STATIC
    src/benchmark.cpp
    src/cli.cpp
    src/gaussian_blur.cpp
    src/io.cpp
    src/weights.cpp
    # add new .cpp files here
)
target_include_directories(imgfilter_core PUBLIC src)
# SYSTEM: third_party is not our code, so its warnings are not reported.
target_include_directories(imgfilter_core SYSTEM PUBLIC third_party)
target_link_libraries(imgfilter_core PUBLIC OpenMP::OpenMP_CXX)
# GCC options only: nvcc would not understand them. The generator expression
# applies them to C++ files and not to the .cu file.
# (Quoted, with ; between the options, so CMake keeps it as one expression.)
target_compile_options(imgfilter_core PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;${cpu_fp_flags}>")

# Exactly one implementation of cuda_blur.h is compiled: the real GPU module,
# or the stand-in that rejects the CUDA Backends.
if(IMGFILTER_CUDA)
    target_sources(imgfilter_core PRIVATE src/cuda_blur.cu)
    # No --use_fast_math, ever: it would also turn fusion back on and swap in
    # less exact division and square roots.
    target_compile_options(imgfilter_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_fp_flags}>)
    target_link_libraries(imgfilter_core PUBLIC CUDA::cudart)
    # PUBLIC, so the tests know to compile their CUDA cases.
    target_compile_definitions(imgfilter_core PUBLIC IMGFILTER_HAS_CUDA)
else()
    target_sources(imgfilter_core PRIVATE src/cuda_blur_stub.cpp)
endif()

# bench prints the build type and compile flags above its results, so they are
# handed to cli.cpp (the only file that prints them) as two string macros.
# The flags are put together the way CMake puts them on the compile line:
# CMAKE_CXX_FLAGS, then the build type's own flags (CMAKE_CXX_FLAGS_RELEASE is
# "-O3 -DNDEBUG"), then the language standard, OpenMP and the warnings above.
string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type_upper)
if(CMAKE_CXX_EXTENSIONS OR NOT DEFINED CMAKE_CXX_EXTENSIONS)
    set(standard_flag "${CMAKE_CXX17_EXTENSION_COMPILE_OPTION}")   # -std=gnu++17
else()
    set(standard_flag "${CMAKE_CXX17_STANDARD_COMPILE_OPTION}")    # -std=c++17
endif()
string(JOIN " " compile_flags
    ${CMAKE_CXX_FLAGS}
    ${CMAKE_CXX_FLAGS_${build_type_upper}}
    ${standard_flag}
    ${OpenMP_CXX_FLAGS}
    -Wall -Wextra
    ${cpu_fp_flags})
set_property(SOURCE src/cli.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    "IMGFILTER_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\""
    "IMGFILTER_COMPILE_FLAGS=\"${compile_flags}\"")

# The same for the CUDA side: nvcc's version, its flags in the order CMake
# uses (CMAKE_CUDA_FLAGS, the build type's, the standard, then ours), and the
# architectures CMake turns into --generate-code options.
if(IMGFILTER_CUDA)
    string(JOIN " " cuda_compile_flags
        ${CMAKE_CUDA_FLAGS}
        ${CMAKE_CUDA_FLAGS_${build_type_upper}}
        -std=c++17
        ${cuda_fp_flags})
    string(JOIN "," cuda_architectures ${CMAKE_CUDA_ARCHITECTURES})
    set_property(SOURCE src/cli.cpp APPEND PROPERTY COMPILE_DEFINITIONS
        "IMGFILTER_CUDA_COMPILER=\"${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}\""
        "IMGFILTER_CUDA_FLAGS=\"${cuda_compile_flags}\""
        "IMGFILTER_CUDA_ARCHITECTURES=\"${cuda_architectures}\"")
endif()

add_executable(imgfilter src/main.cpp)
target_link_libraries(imgfilter PRIVATE imgfilter_core)
target_compile_options(imgfilter PRIVATE -Wall -Wextra ${cpu_fp_flags})

# ---- Tests ----
enable_testing()

# One executable per file in tests/: tests/test_io.cpp builds test_io, and so
# on. Shared helpers live in tests/test_utilities.hpp.
set(TEST_NAMES
    test_backends
    test_blur
    # add new test files here
)

foreach(test_name IN LISTS TEST_NAMES)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE imgfilter_core)
    target_compile_options(${test_name} PRIVATE -Wall -Wextra ${cpu_fp_flags})
endforeach()

# ctest does not build anything by itself. This "test" rebuilds every test
# executable, and the fixture makes it run before the real tests, so one
# command does both:
#   ctest --test-dir build --output-on-failure
add_test(NAME build_tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${TEST_NAMES})
set_tests_properties(build_tests PROPERTIES FIXTURES_SETUP tests_built)

foreach(test_name IN LISTS TEST_NAMES)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED tests_built)
endforeach()

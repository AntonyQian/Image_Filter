cmake_minimum_required(VERSION 3.20)
project(Image_Filter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release so a benchmark never accidentally runs unoptimized
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

# Everything except main() lives in this static library, so the command-line
# tool and the test executable link the exact same code.
add_library(imgfilter_core STATIC
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
target_compile_options(imgfilter_core PRIVATE -Wall -Wextra)   # GCC warnings

add_executable(imgfilter src/main.cpp)
target_link_libraries(imgfilter PRIVATE imgfilter_core)
target_compile_options(imgfilter PRIVATE -Wall -Wextra)

# ---- Tests ----
enable_testing()

add_executable(test_cli tests/test_cli.cpp)
target_link_libraries(test_cli PRIVATE imgfilter_core)
target_compile_options(test_cli PRIVATE -Wall -Wextra)

# ctest does not build anything by itself. This "test" rebuilds test_cli, and
# the fixture makes it run before the real test, so one command does both:
#   ctest --test-dir build --output-on-failure
add_test(NAME build_test_cli
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_cli)
set_tests_properties(build_test_cli PROPERTIES FIXTURES_SETUP test_cli_built)

add_test(NAME cli COMMAND test_cli)
set_tests_properties(cli PROPERTIES FIXTURES_REQUIRED test_cli_built)

#include <iostream>
#include <string>
#include <vector>

#include "cli.h"

// main is deliberately thin: all behaviour lives in run() so the tests can
// call it directly with their own argument list and streams.
int main(int argc, char* argv[]) {
    // argv[0] is the program name, so the real arguments start at argv + 1.
    const std::vector<std::string> args(argv + 1, argv + argc);
    return run(args, std::cout, std::cerr);
}

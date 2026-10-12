#include <iostream>
#include <string>
#include <vector>

#include "cli.h"

int main(int argc, char* argv[]) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    return run(args, std::cout, std::cerr);
}

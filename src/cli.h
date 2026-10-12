#pragma once

#include <iosfwd>
#include <string>
#include <vector>

// The whole command-line tool as a function, so tests can call it directly.
//
// args: the arguments without the program name:
//         input output sigma [--backend serial|openmp] [--threads N]
//       Default is the openmp Backend on omp_get_max_threads() threads.
// out, err: where normal output and error messages go.
//
// Returns the exit code: 0 on success, non-zero on failure. Never throws.
int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

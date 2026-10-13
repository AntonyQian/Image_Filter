#pragma once

#include <iosfwd>
#include <string>
#include <vector>

// The whole command-line tool as a function, so tests can call it directly.
//
// args: the arguments without the program name, in one of two forms:
//         input output sigma [--backend serial|openmp] [--threads N]
//           blurs one image. Default is the openmp Backend on
//           omp_get_max_threads() threads.
//         bench [--sizes N,N,...] [--sigmas S,S,...] [--threads N,N,...]
//               [--runs N] [--csv <results.csv>]
//           runs the benchmark on synthetic Images and prints a table.
//           Defaults: --sizes 4096, --sigmas 3, --threads every count from 1
//           to omp_get_max_threads(), --runs 15, and no CSV file. --csv also
//           writes the table's rows to that file, replacing it if it exists.
//           A Source file literally named "bench" needs a path prefix
//           (./bench).
// out, err: where normal output and error messages go.
//
// Returns the exit code: 0 on success, non-zero on failure. Never throws.
int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

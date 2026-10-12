#pragma once

#include <iosfwd>
#include <string>
#include <vector>

// The whole command-line tool as an ordinary function, so tests can call it
// without starting a process. main() only forwards to this.
//
// args: the command-line arguments WITHOUT the program name:  input output sigma
// out:  normal output (main passes std::cout)
// err:  error messages (main passes std::cerr)
//
// Returns the process exit code: 0 on success, non-zero on any failure.
// Never throws; every failure becomes a message on err plus a non-zero code.
int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

#include "cli.h"

#include <ostream>

#include "image.h"
#include "io.h"

int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    (void)out;
    (void)err;

    const std::string& input_path = args[0];
    const std::string& output_path = args[1];

    const Image image = load_image(input_path);
    save_image(output_path, image);
    return 0;
}

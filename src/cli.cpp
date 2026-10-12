#include "cli.h"

#include <exception>
#include <ostream>

#include "image.h"
#include "io.h"

int run(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    // Check the argument count before touching args[0] or args[1]:
    // reading past the end of a vector is undefined behaviour.
    if (args.size() != 2) {
        err << "Usage: imgfilter <input image> <output.png>\n";
        return 1;
    }

    const std::string& input_path = args[0];
    const std::string& output_path = args[1];

    // load_image and save_image report failure by throwing. Catching here, in
    // one place, turns any failure into a message plus a non-zero exit code,
    // so an exception never escapes to main and aborts the program.
    try {
        const Image image = load_image(input_path);
        save_image(output_path, image);

        out << "Saved " << image.width << "x" << image.height
            << " grayscale image to " << output_path << "\n";
    } catch (const std::exception& e) {
        err << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

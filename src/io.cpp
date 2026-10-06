// The stb implementation must be switched on in exactly ONE .cpp file.
// This is that file. Do not define these macros anywhere else.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "io.h"

#include <stdexcept>

Image load_image(const std::string& path) {


    return Image{};   // placeholder so this compiles; remove once implemented
}

bool save_image(const std::string& path, const Image& img) {


    (void)path;       // placeholders so there are no unused warnings;
    (void)img;        // remove these once implemented
    return false;
}
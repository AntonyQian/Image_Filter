// The stb implementation must be switched on in exactly ONE .cpp file.
// This is that file. Do not define these macros anywhere else.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include "io.h"

#include <cstddef>
#include <stdexcept>

Image load_image(const std::string& path) {
    int width {}; // width and height used in stbi_load
    int height {};
    int channels_in_file {}; // what the file had; we do not use it

    // The last argument (1) asks stb to convert whatever the file holds to a
    // single grayscale channel, so the returned buffer is width * height bytes.
    unsigned char* image_pointer = stbi_load(path.c_str(), &width, &height, &channels_in_file, 1);

    if (image_pointer == nullptr) {
        throw std::runtime_error("Failed to load image: " + path); // thrown if stbi_load fails
    }

    // size_t, not int: width * height can overflow int on very large images.
    const std::size_t total_elements = static_cast<std::size_t>(width) * height;

    Image image;
    image.width = width;
    image.height = height;
    image.channels = 1;
    image.data.assign(image_pointer, image_pointer + total_elements); // copies all pixel bytes

    stbi_image_free(image_pointer); // stb allocated the buffer, so stb frees it

    return image;
}

void save_image(const std::string& path, const Image& img) {
    // Stride is the number of bytes from the start of one row to the next.
    // Our rows are packed with no gaps, so it is width * channels.
    const int stride_in_bytes = img.width * img.channels;

    // stbi_write_png returns 0 on failure (for example the folder does not exist).
    const int ok = stbi_write_png(path.c_str(), img.width, img.height, img.channels,
                                  img.data.data(), stride_in_bytes);

    if (ok == 0) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

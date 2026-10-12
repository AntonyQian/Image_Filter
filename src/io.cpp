// The stb implementation macros must be defined in exactly one .cpp file.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include "io.h"

#include <cstddef>
#include <memory>
#include <stdexcept>

Image load_image(const std::string& path) {
    int width {}; // width and height used in stbi_load
    int height {};
    int channels_in_file {}; // what the file had; we do not use it

    // The 1 asks stb for a single grayscale channel. The unique_ptr frees
    // the stb buffer on every exit path, including exceptions.
    const std::unique_ptr<unsigned char, void (*)(void*)> buffer(
        stbi_load(path.c_str(), &width, &height, &channels_in_file, 1), stbi_image_free);
    const unsigned char* image_pointer = buffer.get();

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

    return image;
}

void save_image(const std::string& path, const Image& img) {
    // Rows are packed with no gaps.
    const int stride_in_bytes = img.width * img.channels;

    const int ok = stbi_write_png(path.c_str(), img.width, img.height, img.channels,
                                  img.data.data(), stride_in_bytes);

    if (ok == 0) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

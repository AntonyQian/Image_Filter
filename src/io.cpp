// The stb implementation must be switched on in exactly ONE .cpp file.
// This is that file. Do not define these macros anywhere else.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"

#include "io.h"

#include <stdexcept>
#include <iostream>

Image load_image(const std::string& path) {
    int width {}; // width and height used in stbi_load
    int height {};
    int channels_in_file {};
    int total_elements {};

    unsigned char* image_pointer = stbi_load(path.c_str(), &width, &height, &channels_in_file, 1); // saves pixel pointer to datait 

    if (image_pointer == nullptr){
        throw std::runtime_error("Failed to load image" + path); // thrown if stbi_load fails
    }

    total_elements = width * height;

    Image image = {width, height, 1};
    image.data.assign(image_pointer, image_pointer + total_elements); // assigns data with all pixel bytes

    stbi_image_free(image_pointer); // frees pointer

    return image;
}

bool save_image(const std::string& path, const Image& img) {


    (void)path;       // placeholders so there are no unused warnings;
    (void)img;        // remove these once implemented
    return false;
}
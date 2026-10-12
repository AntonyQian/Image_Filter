#pragma once

#include <string>
#include "image.h"

// Loads an image file (PNG, JPG, ...) and returns it as 8-bit grayscale
// (channels == 1), whatever the file itself contains.
// Throws std::runtime_error, with the path in the message, if the file
// cannot be read.
Image load_image(const std::string& path);

// Saves an image as a PNG.
// Throws std::runtime_error, with the path in the message, if the file
// cannot be written.
void save_image(const std::string& path, const Image& img);

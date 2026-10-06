#pragma once

#include <string>
#include "image.h"

// Loads an image file (PNG, JPG, ...) and returns it as 8-bit grayscale.
// Suggestion: throw std::runtime_error if the file cannot be read.
// Whatever you choose, write it in this comment so callers know.
Image load_image(const std::string& path);

// Saves a grayscale image as a PNG. Returns true on success.
bool save_image(const std::string& path, const Image& img);
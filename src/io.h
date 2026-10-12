#pragma once

#include <string>
#include "image.h"

// Loads an image file as 8-bit grayscale, whatever the file contains.
// Throws std::runtime_error if the file can't be read.
Image load_image(const std::string& path);

// Saves an image as a PNG.
// Throws std::runtime_error if the file can't be written.
void save_image(const std::string& path, const Image& img);

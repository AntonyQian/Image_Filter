#include "weights.h"

#include <cmath>
#include <limits>
#include <stdexcept>

int radius_for_sigma(double sigma) {
    // In double: float could round 3 * sigma just past a whole number and
    // add one to the radius.
    const double radius = std::ceil(3.0 * sigma);

    // Compare before converting: an out-of-range double to int is undefined.
    // The limit leaves room for side() = 2 * radius + 1, and also catches an
    // infinite radius.
    const int largest_radius = (std::numeric_limits<int>::max() - 1) / 2;
    if (radius > static_cast<double>(largest_radius)) {
        throw std::invalid_argument("sigma is too large: the blur radius does not fit in an int");
    }

    return static_cast<int>(radius);
}

Weights make_weights(double sigma) {
    Weights weights;
    weights.radius = radius_for_sigma(sigma);
    const int radius = weights.radius;
    const int side = weights.side();

    // Allocate first so a huge sigma fails here, before the loops below.
    weights.values.resize(static_cast<std::size_t>(side) * side);

    // 1D Gaussian. The 1 / (sigma * sqrt(2 pi)) factor is skipped since
    // normalising cancels it.
    //
    // The exponent is -0.5 * t * t with t = d / sigma rather than
    // -d^2 / (2 sigma^2): d * d overflows int past radius 46340, and
    // sigma * sigma underflows to 0 for a tiny sigma, giving 0 / 0.
    std::vector<double> gaussian_1d(static_cast<std::size_t>(side));
    for (int d = -radius; d <= radius; ++d) {
        const double t = static_cast<double>(d) / sigma;
        gaussian_1d[static_cast<std::size_t>(d + radius)] = std::exp(-0.5 * t * t);
    }

    // The 2D Gaussian is the product of two 1D ones. Sum them to normalise.
    double sum = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            sum += gaussian_1d[static_cast<std::size_t>(dy + radius)] *
                   gaussian_1d[static_cast<std::size_t>(dx + radius)];
        }
    }

    // Normalise in double and narrow to float once, at the end.
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const double product = gaussian_1d[static_cast<std::size_t>(dy + radius)] *
                                   gaussian_1d[static_cast<std::size_t>(dx + radius)];
            weights.values[weights.index(dx, dy)] = static_cast<float>(product / sum);
        }
    }

    return weights;
}

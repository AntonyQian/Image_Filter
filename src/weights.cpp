#include "weights.h"

#include <cmath>

int radius_for_sigma(double sigma) {
    // Done in double on purpose. 3.0 * 1.0 is exactly 3.0 in double, so
    // sigma 1 gives radius 3. In float, a sigma that is not exactly
    // representable could land just above a whole number and round up to one
    // more than intended.
    return static_cast<int>(std::ceil(3.0 * sigma));
}

Weights make_weights(double sigma) {
    Weights weights;
    weights.radius = radius_for_sigma(sigma);
    const int radius = weights.radius;
    const int side = weights.side();

    // Step 1: the 1D Gaussian, exp(-d^2 / (2 sigma^2)), for d = -radius..radius.
    // The usual 1 / (sigma * sqrt(2 pi)) factor in front is left out: it is
    // the same for every entry, so it cancels when we divide by the sum below.
    std::vector<double> gaussian_1d(static_cast<std::size_t>(side));
    for (int d = -radius; d <= radius; ++d) {
        gaussian_1d[static_cast<std::size_t>(d + radius)] =
            std::exp(-(d * d) / (2.0 * sigma * sigma));
    }

    // Step 2: the 2D Gaussian is the product of two 1D ones, because
    // exp(-(dx^2 + dy^2) / k) = exp(-dx^2 / k) * exp(-dy^2 / k).
    // Add up every product so we know what to divide by.
    double sum = 0.0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            sum += gaussian_1d[static_cast<std::size_t>(dy + radius)] *
                   gaussian_1d[static_cast<std::size_t>(dx + radius)];
        }
    }

    // Step 3: divide each product by the sum so the Weights total 1. If they
    // totalled more or less than 1 the whole image would get brighter or
    // darker. The division is done in double and only the finished weight is
    // narrowed to float, so each stored weight is as accurate as a float can
    // be (rounding once, at the end, instead of at every step).
    weights.values.resize(static_cast<std::size_t>(side) * side);
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const double product = gaussian_1d[static_cast<std::size_t>(dy + radius)] *
                                   gaussian_1d[static_cast<std::size_t>(dx + radius)];
            weights.values[weights.index(dx, dy)] = static_cast<float>(product / sum);
        }
    }

    return weights;
}

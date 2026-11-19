// nn_tb.cpp - testbench of the standalone network: raw 70x70 image -> 3 class scores
//
// Runs the network on real held-out validation images plus an all-white, an all-black and a noise image (run-length
// encoded in nn_calibration.h) and compares the three Q16.16 scores BIT-EXACTLY with the Python integer reference
// (train_model/export_hls.py). Prints the class accuracy on the real images (highest score = predicted class).
// Returns 0 only if every vector matches.
#include <stdint.h>
#include <cstdio>
#include <cstring>

#include "nn_calibration.h"

// TEST_VECTOR_LIMIT < CALIBRATION_COUNT runs an evenly spaced subset (fast RTL co-simulation).
#ifndef TEST_VECTOR_LIMIT
#define TEST_VECTOR_LIMIT CALIBRATION_COUNT
#endif

extern "C" uint8_t shape_accel(const uint32_t image_gray8[1225], int32_t scores_q16[CALIBRATION_OUTPUTS]);

static const int IMAGE_BYTES = 70 * 70;

int main()
{
    int mismatch_count = 0, correct_count = 0, real_image_count = 0, tested_count = 0;

    for (int test_number = 0; test_number < TEST_VECTOR_LIMIT; test_number++) {
        const int vector_index = TEST_VECTOR_LIMIT >= CALIBRATION_COUNT ? test_number : (test_number * CALIBRATION_COUNT) / TEST_VECTOR_LIMIT;
        ++tested_count;

        // -------- DECODE THE IMAGE (run-length encoded) --------
        uint8_t image[IMAGE_BYTES];
        int position = 0;
        for (uint32_t run = calibration_run_starts[vector_index]; run < calibration_run_starts[vector_index + 1]; run++) {
            std::memset(image + position, calibration_run_values[run], calibration_run_lengths[run]);
            position += calibration_run_lengths[run];
        }
        uint32_t image_words[1225];                      // little-endian byte packing, same as the PS buffer
        std::memcpy(image_words, image, IMAGE_BYTES);

        // -------- RUN THE NETWORK --------
        int32_t scores_q16[CALIBRATION_OUTPUTS] = {0, 0, 0};
        const uint8_t class_result = shape_accel(image_words, scores_q16);

        // -------- COMPARE --------
        bool identical = true;
        for (int output = 0; output < CALIBRATION_OUTPUTS; output++)
            identical = identical && scores_q16[output] == calibration_scores[vector_index][output];
        int highest_class = 0;                           // the class result the controller receives: the highest score, first class on a tie
        for (int class_index = 1; class_index < 3; class_index++)
            if (scores_q16[class_index] > scores_q16[highest_class]) highest_class = class_index;
        identical = identical && class_result == highest_class;
        if (!identical) {
            ++mismatch_count;
            std::printf("MISMATCH %d:", vector_index);
            for (int output = 0; output < CALIBRATION_OUTPUTS; output++)
                std::printf(" got %d expected %d;", scores_q16[output], calibration_scores[vector_index][output]);
            std::printf("\n");
        }

        // -------- CLASS ACCURACY on the real images (label < 0 marks the synthetic extremes) --------
        if (calibration_labels[vector_index] >= 0) {
            ++real_image_count;
            correct_count += highest_class == calibration_labels[vector_index];
        }
    }

    std::printf("vectors %d, bit-exact mismatches %d, task metric on real images %d/%d\n",
                tested_count, mismatch_count, correct_count, real_image_count);
    return mismatch_count ? 1 : 0;
}

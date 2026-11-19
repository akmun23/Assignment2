// nn_tb.cpp - testbench of a cascade head: the selector's integer feature map -> 1 score
//
// Runs the head on the selector's feature maps of real held-out validation images of its pair (plus an all-white, an all-black
// and a noise image) and compares the Q16.16 score BIT-EXACTLY with the Python integer reference (train_model/export_hls.py).
// Prints how often the sign of the score picks the right class of the pair (score >= 0 means the lower-index class).
// Returns 0 only if every vector matches.
#include <stdint.h>
#include <cstdio>
#include <cstring>

#include "nn_calibration.h"

// TEST_VECTOR_LIMIT < CALIBRATION_COUNT runs an evenly spaced subset (fast RTL co-simulation).
#ifndef TEST_VECTOR_LIMIT
#define TEST_VECTOR_LIMIT CALIBRATION_COUNT
#endif

extern "C" uint8_t shape_accel(const int16_t feature_map[CALIBRATION_FEATURE_COUNT], int32_t scores_q16[CALIBRATION_OUTPUTS]);

int main()
{
    int mismatch_count = 0, correct_count = 0, real_image_count = 0, tested_count = 0;

    for (int test_number = 0; test_number < TEST_VECTOR_LIMIT; test_number++) {
        const int vector_index = TEST_VECTOR_LIMIT >= CALIBRATION_COUNT ? test_number : (test_number * CALIBRATION_COUNT) / TEST_VECTOR_LIMIT;
        ++tested_count;

        // -------- RUN THE HEAD on the selector's integer feature map --------
        int32_t scores_q16[CALIBRATION_OUTPUTS] = {0};
        const uint8_t class_result = shape_accel(calibration_features[vector_index], scores_q16);

        // -------- COMPARE --------
        if (scores_q16[0] != calibration_scores[vector_index][0] || class_result != (scores_q16[0] >= 0 ? 0 : 1)) {
            ++mismatch_count;
            std::printf("MISMATCH %d: got %d expected %d\n", vector_index, scores_q16[0], calibration_scores[vector_index][0]);
        }

        // -------- PAIR DECISION on the real images (label < 0 marks the synthetic extremes; label 1 = lower-index class) --------
        if (calibration_labels[vector_index] >= 0) {
            ++real_image_count;
            correct_count += (class_result == 0) == (calibration_labels[vector_index] == 1);
        }
    }

    std::printf("vectors %d, bit-exact mismatches %d, task metric on real images %d/%d\n",
                tested_count, mismatch_count, correct_count, real_image_count);
    return mismatch_count ? 1 : 0;
}

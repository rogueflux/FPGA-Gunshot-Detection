#ifndef TEST_AUDIO_SAMPLES_H
#define TEST_AUDIO_SAMPLES_H

#include <stdint.h>

// Test audio samples for FPGA verification
#define TEST_SAMPLE_RATE 16000
#define TEST_SAMPLE_COUNT 1024

// Gunshot test sample (synthetic impulse)
const int16_t gunshot_test_samples[TEST_SAMPLE_COUNT] = {
    #include "gunshot_samples.dat"
};

// Background noise test sample (white noise)
const int16_t background_test_samples[TEST_SAMPLE_COUNT] = {
    #include "background_samples.dat"
};

// Speech test sample (synthetic sine wave)
const int16_t speech_test_samples[TEST_SAMPLE_COUNT] = {
    #include "speech_samples.dat"
};

// Impulse noise test sample
const int16_t impulse_test_samples[TEST_SAMPLE_COUNT] = {
    #include "impulse_samples.dat"
};

// Multi-channel test data (3 channels)
const int16_t multichannel_test_samples[3][TEST_SAMPLE_COUNT] = {
    #include "multichannel_samples.dat"
};

// Feature vectors for neural network testing
const float test_features_gunshot[2052] = {
    #include "features_gunshot.dat"
};

const float test_features_background[2052] = {
    #include "features_background.dat"
};

// Expected results for test vectors
typedef struct {
    const char* test_name;
    const int16_t* audio_data;
    uint32_t sample_count;
    uint8_t expected_detection;
    uint8_t expected_confidence;
    float expected_position_x;
    float expected_position_y;
} TestVector;

const TestVector test_vectors[] = {
    {
        "gunshot_impulse",
        gunshot_test_samples,
        TEST_SAMPLE_COUNT,
        1,  // Expected detection: YES
        200, // Expected confidence: ~78%
        2.5,  // Expected X position (meters)
        1.2   // Expected Y position (meters)
    },
    {
        "background_noise",
        background_test_samples,
        TEST_SAMPLE_COUNT,
        0,  // Expected detection: NO
        50,  // Expected confidence: ~20%
        0.0,
        0.0
    },
    {
        "speech_audio",
        speech_test_samples,
        TEST_SAMPLE_COUNT,
        0,  // Expected detection: NO
        30,  // Expected confidence: ~12%
        0.0,
        0.0
    },
    {
        "impulse_noise",
        impulse_test_samples,
        TEST_SAMPLE_COUNT,
        0,  // Expected detection: NO (or could be 1 for false positive)
        120, // Expected confidence: ~47%
        0.0,
        0.0
    }
};

#define NUM_TEST_VECTORS (sizeof(test_vectors) / sizeof(test_vectors[0]))

#endif // TEST_AUDIO_SAMPLES_H

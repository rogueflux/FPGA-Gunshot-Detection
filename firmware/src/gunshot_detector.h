#ifndef GUNSHOT_DETECTOR_H
#define GUNSHOT_DETECTOR_H

#include <stdint.h>
#include "platform_config.h"

// Detection parameters
#define DETECTION_THRESHOLD 0.65f
#define MAX_FEATURE_SIZE 2052
#define MODEL_INPUT_SIZE 1024

// Hardware accelerator addresses
#define DETECTOR_BASE_ADDR 0x43C00000

// Detection result structure
typedef struct {
    uint8_t is_detected;
    uint8_t confidence;
    uint32_t timestamp;
    uint16_t sample_index;
} DetectionResult;

// Function prototypes
void gunshot_detector_init(uint32_t base_addr);
void gunshot_detector_predict(const float* features, uint8_t* confidence, uint8_t* is_gunshot);
void gunshot_detector_extract_features(const int16_t* audio_data, uint32_t num_samples, float* features);
void gunshot_detector_reset(void);
uint32_t gunshot_detector_get_version(void);

// Hardware accelerator interface
void hw_accel_write_features(const float* features);
void hw_accel_start_inference(void);
uint8_t hw_accel_get_result(void);

#endif // GUNSHOT_DETECTOR_H

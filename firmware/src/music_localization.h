#ifndef MUSIC_LOCALIZATION_H
#define MUSIC_LOCALIZATION_H

#include <stdint.h>
#include "platform_config.h"

// Localization parameters
#define NUM_MICROPHONES 6
#define SPEED_OF_SOUND 343.0f  // m/s
#define SAMPLE_RATE 16000
#define SEARCH_RADIUS 10.0f    // meters
#define GRID_RESOLUTION 0.5f   // meters

// Microphone array geometry (hexagonal, 10cm radius)
typedef struct {
    float x;
    float y;
} MicPosition;

// Localization result
typedef struct {
    float x;           // X position in meters
    float y;           // Y position in meters
    float confidence;  // 0.0 to 1.0
    float azimuth;     // Angle in degrees
    float elevation;   // Angle in degrees (if 3D)
} LocalizationResult;

// Function prototypes
void music_init(void);
void music_localize(const int16_t audio[NUM_MICROPHONES][32000],
                   LocalizationResult* result);
void music_compute_covariance(const int16_t audio[NUM_MICROPHONES][1024],
                             float covariance[NUM_MICROPHONES][NUM_MICROPHONES]);
void music_compute_steering_vector(float x, float y, float steering[NUM_MICROPHONES]);
float music_compute_pseudospectrum(const float covariance[NUM_MICROPHONES][NUM_MICROPHONES],
                                  const float steering[NUM_MICROPHONES]);
void music_find_peak(const float spectrum[41][41], float* x, float* y, float* confidence);

// Simplified version for SPI ADC with 3 microphones
void music_localize_spi(const int16_t* audio_data, uint32_t num_samples,
                       int16_t* x_pos, int16_t* y_pos, uint8_t* confidence);

#endif // MUSIC_LOCALIZATION_H

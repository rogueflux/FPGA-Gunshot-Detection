#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <stdint.h>
#include "platform_config.h"

// Audio processing parameters
#define FFT_SIZE 256
#define NUM_MEL_BANDS 40
#define NUM_MFCC_COEFFS 13
#define SAMPLE_RATE 16000

// Function prototypes
void audio_processor_init(void);
void extract_features(const int16_t* audio_data, int num_samples, float* features);
void process_multi_channel(const int16_t audio[NUM_CHANNELS][AUDIO_BUFFER_SIZE],
                          float* features);
void apply_preemphasis(float* audio, int num_samples);
void apply_window(float* audio);
void compute_spectrum(const float* audio, float* spectrum);
void compute_mel_spectrum(const float* spectrum, float* mel_spectrum);
void compute_mfcc(const float* mel_spectrum, float* mfcc);

// Utility functions
void normalize_audio(float* audio, int num_samples);
float compute_rms(const float* audio, int num_samples);
float compute_zero_crossing_rate(const float* audio, int num_samples);
void compute_spectral_centroid(const float* spectrum, float* centroid);

#endif // AUDIO_PROCESSOR_H

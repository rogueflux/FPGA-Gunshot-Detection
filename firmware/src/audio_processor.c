#include "audio_processor.h"
#include <math.h>
#include <string.h>
#include "xil_printf.h"

// Pre-emphasis filter coefficient
#define PREEMPHASIS_COEFF 0.97f

// Windowing function (Hanning window)
static float window[FFT_SIZE];
static int window_initialized = 0;

// Initialize audio processor
void audio_processor_init(void) {
    if (!window_initialized) {
        // Generate Hanning window
        for (int i = 0; i < FFT_SIZE; i++) {
            window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (FFT_SIZE - 1)));
        }
        window_initialized = 1;
    }
}

// Apply pre-emphasis filter
void apply_preemphasis(float* audio, int num_samples) {
    if (num_samples < 2) return;
    
    float prev = audio[0];
    for (int i = 1; i < num_samples; i++) {
        float current = audio[i];
        audio[i] = current - PREEMPHASIS_COEFF * prev;
        prev = current;
    }
    audio[0] = audio[1]; // First sample approximation
}

// Apply windowing
void apply_window(float* audio) {
    for (int i = 0; i < FFT_SIZE; i++) {
        audio[i] *= window[i];
    }
}

// Compute FFT magnitude spectrum (simplified)
void compute_spectrum(const float* audio, float* spectrum) {
    // Simple DFT for demonstration
    // In real implementation, use hardware FFT IP
    
    for (int k = 0; k < FFT_SIZE/2; k++) {
        float real = 0.0f;
        float imag = 0.0f;
        
        for (int n = 0; n < FFT_SIZE; n++) {
            float angle = 2.0f * M_PI * k * n / FFT_SIZE;
            real += audio[n] * cosf(angle);
            imag -= audio[n] * sinf(angle);
        }
        
        spectrum[k] = sqrtf(real*real + imag*imag) / FFT_SIZE;
    }
}

// Compute Mel-frequency spectrum
void compute_mel_spectrum(const float* spectrum, float* mel_spectrum) {
    // Simple mel filter bank (simplified)
    // In real implementation, use proper mel filter bank
    
    for (int m = 0; m < NUM_MEL_BANDS; m++) {
        mel_spectrum[m] = 0.0f;
        
        // Simple triangular filters (simplified)
        for (int k = 0; k < FFT_SIZE/2; k++) {
            float freq = (float)k * SAMPLE_RATE / FFT_SIZE;
            float mel_freq = 2595.0f * log10f(1.0f + freq / 700.0f);
            
            // Simplified filter response
            float filter_val = 0.0f;
            if (mel_freq >= m*100 && mel_freq <= (m+1)*100) {
                filter_val = 1.0f - fabsf(mel_freq - (m+0.5f)*100) / 50.0f;
            }
            
            mel_spectrum[m] += spectrum[k] * filter_val;
        }
        
        // Log compression
        if (mel_spectrum[m] < 1e-6f) mel_spectrum[m] = 1e-6f;
        mel_spectrum[m] = log10f(mel_spectrum[m]);
    }
}

// Compute MFCC coefficients
void compute_mfcc(const float* mel_spectrum, float* mfcc) {
    // Simple DCT (simplified)
    // In real implementation, use proper DCT
    
    for (int c = 0; c < NUM_MFCC_COEFFS; c++) {
        mfcc[c] = 0.0f;
        
        for (int m = 0; m < NUM_MEL_BANDS; m++) {
            mfcc[c] += mel_spectrum[m] * cosf(M_PI * c * (m + 0.5f) / NUM_MEL_BANDS);
        }
        
        // Scale factor
        if (c == 0) {
            mfcc[c] *= sqrtf(1.0f / NUM_MEL_BANDS);
        } else {
            mfcc[c] *= sqrtf(2.0f / NUM_MEL_BANDS);
        }
    }
}

// Extract features from audio data
void extract_features(const int16_t* audio_data, int num_samples, float* features) {
    float processed[FFT_SIZE];
    
    // 1. Convert to float and normalize
    for (int i = 0; i < FFT_SIZE; i++) {
        if (i < num_samples) {
            processed[i] = audio_data[i] / 32768.0f;
        } else {
            processed[i] = 0.0f;
        }
    }
    
    // 2. Apply pre-emphasis
    apply_preemphasis(processed, FFT_SIZE);
    
    // 3. Apply window
    apply_window(processed);
    
    // 4. Compute spectrum
    float spectrum[FFT_SIZE/2];
    compute_spectrum(processed, spectrum);
    
    // 5. Compute MFCCs
    float mel_spectrum[NUM_MEL_BANDS];
    compute_mel_spectrum(spectrum, mel_spectrum);
    
    float mfcc[NUM_MFCC_COEFFS];
    compute_mfcc(mel_spectrum, mfcc);
    
    // 6. Copy features to output
    for (int i = 0; i < NUM_MFCC_COEFFS; i++) {
        features[i] = mfcc[i];
    }
    
    // 7. Add spectral features
    float energy = 0.0f;
    for (int i = 0; i < FFT_SIZE; i++) {
        energy += processed[i] * processed[i];
    }
    features[NUM_MFCC_COEFFS] = energy;
    
    // 8. Add zero-crossing rate
    int zero_crossings = 0;
    for (int i = 1; i < FFT_SIZE; i++) {
        if (processed[i] * processed[i-1] < 0) {
            zero_crossings++;
        }
    }
    features[NUM_MFCC_COEFFS + 1] = (float)zero_crossings / FFT_SIZE;
    
    // 9. Pad remaining features
    for (int i = NUM_MFCC_COEFFS + 2; i < TOTAL_FEATURES; i++) {
        features[i] = 0.0f;
    }
}

// Process multi-channel audio
void process_multi_channel(const int16_t audio[NUM_CHANNELS][AUDIO_BUFFER_SIZE],
                          float* features) {
    // Combine channels (simple averaging)
    float combined[FFT_SIZE];
    
    for (int i = 0; i < FFT_SIZE; i++) {
        float sum = 0.0f;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            sum += audio[ch][i] / 32768.0f;
        }
        combined[i] = sum / NUM_CHANNELS;
    }
    
    // Extract features
    extract_features((int16_t*)combined, FFT_SIZE, features);
}

#include "music_localization.h"
#include <math.h>
#include <string.h>
#include "xil_printf.h"

// Microphone positions (hexagonal array, 10cm radius)
static const MicPosition mic_positions[NUM_MICROPHONES] = {
    {0.0f, 0.10f},    // Mic 1: 0 degrees
    {0.0866f, 0.05f}, // Mic 2: 60 degrees
    {0.0866f, -0.05f},// Mic 3: 120 degrees
    {0.0f, -0.10f},   // Mic 4: 180 degrees
    {-0.0866f, -0.05f},// Mic 5: 240 degrees
    {-0.0866f, 0.05f} // Mic 6: 300 degrees
};

// Initialize MUSIC localization
void music_init(void) {
    xil_printf("MUSIC Localization Initialized\r\n");
    xil_printf("Array: %d microphones, %.2f m radius\r\n", 
               NUM_MICROPHONES, 0.10f);
}

// Compute spatial covariance matrix
void music_compute_covariance(const int16_t audio[NUM_MICROPHONES][1024],
                             float covariance[NUM_MICROPHONES][NUM_MICROPHONES]) {
    // Reset covariance matrix
    for (int i = 0; i < NUM_MICROPHONES; i++) {
        for (int j = 0; j < NUM_MICROPHONES; j++) {
            covariance[i][j] = 0.0f;
        }
    }
    
    // Compute covariance
    for (int s = 0; s < 1024; s++) {
        for (int i = 0; i < NUM_MICROPHONES; i++) {
            float sample_i = audio[i][s] / 32768.0f; // Normalize to [-1, 1]
            for (int j = 0; j < NUM_MICROPHONES; j++) {
                float sample_j = audio[j][s] / 32768.0f;
                covariance[i][j] += sample_i * sample_j;
            }
        }
    }
    
    // Normalize
    for (int i = 0; i < NUM_MICROPHONES; i++) {
        for (int j = 0; j < NUM_MICROPHONES; j++) {
            covariance[i][j] /= 1024.0f;
        }
    }
}

// Compute steering vector for a given position
void music_compute_steering_vector(float x, float y, float steering[NUM_MICROPHONES]) {
    for (int m = 0; m < NUM_MICROPHONES; m++) {
        // Calculate distance difference
        float dx = x - mic_positions[m].x;
        float dy = y - mic_positions[m].y;
        float distance = sqrtf(dx*dx + dy*dy);
        
        // Calculate time delay in samples
        float time_delay = distance / SPEED_OF_SOUND;
        float sample_delay = time_delay * SAMPLE_RATE;
        
        // Steering vector element (simplified - using cosine)
        float phase = 2.0f * M_PI * sample_delay / 1024.0f;
        steering[m] = cosf(phase);
    }
}

// Compute MUSIC pseudospectrum for a single position
float music_compute_pseudospectrum(const float covariance[NUM_MICROPHONES][NUM_MICROPHONES],
                                  const float steering[NUM_MICROPHONES]) {
    // Simple implementation for single source
    // Full MUSIC would require eigenvalue decomposition
    
    float numerator = 0.0f;
    float denominator = 0.0f;
    
    // Simplified: a^H * R * a (steering vector Hermitian * covariance * steering vector)
    for (int i = 0; i < NUM_MICROPHONES; i++) {
        for (int j = 0; j < NUM_MICROPHONES; j++) {
            numerator += steering[i] * covariance[i][j] * steering[j];
        }
        denominator += steering[i] * steering[i];
    }
    
    // Avoid division by zero
    if (denominator < 1e-6f) return 0.0f;
    
    return numerator / denominator;
}

// Find peak in pseudospectrum
void music_find_peak(const float spectrum[41][41], float* x, float* y, float* confidence) {
    float max_val = -INFINITY;
    int max_i = 0, max_j = 0;
    
    // Search grid
    for (int i = 0; i < 41; i++) {
        for (int j = 0; j < 41; j++) {
            if (spectrum[i][j] > max_val) {
                max_val = spectrum[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }
    
    // Convert grid indices to coordinates
    *x = (max_i - 20) * GRID_RESOLUTION;
    *y = (max_j - 20) * GRID_RESOLUTION;
    *confidence = fminf(max_val / 100.0f, 1.0f);
}

// Main localization function
void music_localize(const int16_t audio[NUM_MICROPHONES][32000],
                   LocalizationResult* result) {
    float covariance[NUM_MICROPHONES][NUM_MICROPHONES];
    float spectrum[41][41];
    
    // 1. Compute covariance matrix (use first 1024 samples)
    music_compute_covariance(audio, covariance);
    
    // 2. Compute pseudospectrum over search grid
    float max_val = -INFINITY;
    float best_x = 0.0f, best_y = 0.0f;
    
    for (int xi = 0; xi < 41; xi++) {
        float x = (xi - 20) * GRID_RESOLUTION;
        for (int yj = 0; yj < 41; yj++) {
            float y = (yj - 20) * GRID_RESOLUTION;
            
            // Compute steering vector
            float steering[NUM_MICROPHONES];
            music_compute_steering_vector(x, y, steering);
            
            // Compute pseudospectrum
            spectrum[xi][yj] = music_compute_pseudospectrum(covariance, steering);
            
            // Track maximum
            if (spectrum[xi][yj] > max_val) {
                max_val = spectrum[xi][yj];
                best_x = x;
                best_y = y;
            }
        }
    }
    
    // 3. Set result
    result->x = best_x;
    result->y = best_y;
    result->confidence = fminf(max_val / 100.0f, 1.0f);
    result->azimuth = atan2f(best_y, best_x) * 180.0f / M_PI;
    result->elevation = 0.0f; // 2D only
}

// Simplified localization for SPI ADC with 3 microphones
void music_localize_spi(const int16_t* audio_data, uint32_t num_samples,
                       int16_t* x_pos, int16_t* y_pos, uint8_t* confidence) {
    // Simplified TDOA (Time Difference of Arrival) for 3 microphones
    
    // Find peak in each channel
    int peak_indices[3] = {0};
    for (int ch = 0; ch < 3; ch++) {
        int max_val = 0;
        for (int i = 0; i < fmin(num_samples, 1024); i++) {
            int16_t val = abs(audio_data[ch * num_samples + i]);
            if (val > max_val) {
                max_val = val;
                peak_indices[ch] = i;
            }
        }
    }
    
    // Calculate TDOA between channels
    int diff12 = peak_indices[1] - peak_indices[0];
    int diff13 = peak_indices[2] - peak_indices[0];
    
    // Convert to distance (samples to meters)
    float dist12 = (diff12 * SPEED_OF_SOUND) / SAMPLE_RATE;
    float dist13 = (diff13 * SPEED_OF_SOUND) / SAMPLE_RATE;
    
    // Simplified triangulation (assume linear array along x-axis)
    // Mic positions: (-0.1, 0), (0, 0), (0.1, 0)
    float x = dist12 * 5.0f;  // Scale factor
    float y = dist13 * 5.0f;
    
    // Constrain to search radius
    if (x > 10.0f) x = 10.0f;
    if (x < -10.0f) x = -10.0f;
    if (y > 10.0f) y = 10.0f;
    if (y < -10.0f) y = -10.0f;
    
    *x_pos = (int16_t)(x * 100);  // Convert to cm
    *y_pos = (int16_t)(y * 100);
    
    // Simple confidence based on peak clarity
    float conf = 0.5f + 0.5f * fminf(fabs(diff12) / 100.0f, 1.0f);
    *confidence = (uint8_t)(conf * 255);
}

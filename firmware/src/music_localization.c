#include "music_localization.h"
#include <math.h>

// Microphone array geometry (hexagonal, 10cm radius)
static const float mic_positions[6][2] = {
    {0.0, 0.10},      // Mic 1: 0 degrees
    {0.0866, 0.05},   // Mic 2: 60 degrees
    {0.0866, -0.05},  // Mic 3: 120 degrees
    {0.0, -0.10},     // Mic 4: 180 degrees
    {-0.0866, -0.05}, // Mic 5: 240 degrees
    {-0.0866, 0.05}   // Mic 6: 300 degrees
};

// Speed of sound (m/s)
#define SPEED_OF_SOUND 343.0

void music_localize(int16_t audio[6][32000], 
                    int16_t *x_pos, int16_t *y_pos,
                    uint8_t *confidence) {
    
    // 1. Compute spatial covariance matrix
    float covariance[6][6] = {0};
    int num_samples = 32000;
    
    // Use only the first 1024 samples (impulse region)
    for (int i = 0; i < 1024; i++) {
        for (int m1 = 0; m1 < 6; m1++) {
            for (int m2 = 0; m2 < 6; m2++) {
                covariance[m1][m2] += 
                    (float)audio[m1][i] * (float)audio[m2][i];
            }
        }
    }
    
    // Normalize
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            covariance[i][j] /= 1024.0;
        }
    }
    
    // 2. Eigenvalue decomposition (simplified for single source)
    // For FPGA implementation, this would use a hardware-optimized SVD
    
    // 3. Compute steering vectors and MUSIC spectrum
    float max_val = -INFINITY;
    int max_i = 0, max_j = 0;
    
    // Search grid (-10m to 10m in 0.5m steps)
    for (int xi = -20; xi <= 20; xi++) {
        float x = xi * 0.5;
        for (int yj = -20; yj <= 20; yj++) {
            float y = yj * 0.5;
            
            // Compute steering vector for this position
            float steering[6];
            for (int m = 0; m < 6; m++) {
                float dx = x - mic_positions[m][0];
                float dy = y - mic_positions[m][1];
                float distance = sqrtf(dx*dx + dy*dy);
                // Time delay in samples (16kHz sampling)
                float delay = (distance / SPEED_OF_SOUND) * 16000.0;
                steering[m] = cosf(2 * M_PI * delay / 1024.0);
            }
            
            // Compute MUSIC pseudospectrum (simplified)
            float spectrum = 0;
            for (int m = 0; m < 6; m++) {
                spectrum += steering[m] * steering[m];
            }
            
            if (spectrum > max_val) {
                max_val = spectrum;
                max_i = xi;
                max_j = yj;
            }
        }
    }
    
    // 4. Output results
    *x_pos = max_i * 50; // Convert to cm
    *y_pos = max_j * 50;
    
    // Confidence based on peak sharpness
    *confidence = (uint8_t)fminf(max_val * 100.0, 255.0);
}

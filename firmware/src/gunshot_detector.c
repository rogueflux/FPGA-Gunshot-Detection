#include "gunshot_detector.h"
#include <math.h>
#include <string.h>
#include "xil_io.h"
#include "xil_cache.h"

// Feature extraction buffers
static float feature_buffer[MAX_FEATURE_SIZE];
static uint32_t detector_base_addr = DETECTOR_BASE_ADDR;

// Initialize the gunshot detector
void gunshot_detector_init(uint32_t base_addr) {
    detector_base_addr = base_addr;
    
    // Reset the hardware accelerator
    Xil_Out32(detector_base_addr + 0x00, 0x01); // Control register
    Xil_Out32(detector_base_addr + 0x00, 0x00); // Clear reset
    
    // Set detection threshold (scaled to fixed-point)
    uint32_t threshold_fixed = (uint32_t)(DETECTION_THRESHOLD * 256);
    Xil_Out32(detector_base_addr + 0x04, threshold_fixed);
    
    // Flush cache
    Xil_DCacheFlush();
}

// Extract audio features (simplified version)
void gunshot_detector_extract_features(const int16_t* audio_data, uint32_t num_samples, float* features) {
    // Bandpass filter: 50Hz - 8kHz (for 16kHz sampling)
    float filtered[1024];
    
    // Simple DC removal and normalization
    float sum = 0.0f;
    for (int i = 0; i < num_samples && i < 1024; i++) {
        sum += audio_data[i];
    }
    float mean = sum / fmin(num_samples, 1024);
    
    // Extract features (simplified for FPGA)
    // 1. Energy
    float energy = 0.0f;
    for (int i = 0; i < 1024; i++) {
        float sample = (i < num_samples) ? (audio_data[i] - mean) / 32768.0f : 0.0f;
        filtered[i] = sample;
        energy += sample * sample;
    }
    features[0] = energy;
    
    // 2. Zero-crossing rate
    int zero_crossings = 0;
    for (int i = 1; i < 1024; i++) {
        if (filtered[i] * filtered[i-1] < 0) {
            zero_crossings++;
        }
    }
    features[1] = zero_crossings / 1024.0f;
    
    // 3. Spectral features (simplified FFT)
    // In real implementation, use hardware FFT IP
    for (int i = 2; i < 50; i++) {
        features[i] = fabs(filtered[(i-2)*20]) * 0.1f; // Dummy spectral features
    }
    
    // 4. Temporal features
    float max_val = 0.0f;
    float min_val = 0.0f;
    for (int i = 0; i < 1024; i++) {
        if (filtered[i] > max_val) max_val = filtered[i];
        if (filtered[i] < min_val) min_val = filtered[i];
    }
    features[50] = max_val;
    features[51] = min_val;
    features[52] = max_val - min_val; // Peak-to-peak
    
    // Pad remaining features
    for (int i = 53; i < MAX_FEATURE_SIZE; i++) {
        features[i] = 0.0f;
    }
}

// Predict using hardware accelerator
void gunshot_detector_predict(const float* features, uint8_t* confidence, uint8_t* is_gunshot) {
    // Write features to hardware accelerator
    hw_accel_write_features(features);
    
    // Start inference
    hw_accel_start_inference();
    
    // Wait for completion (polling)
    while ((Xil_In32(detector_base_addr + 0x08) & 0x01) == 0) {
        // Busy wait - in real implementation use interrupts
    }
    
    // Read results
    uint32_t result = Xil_In32(detector_base_addr + 0x0C);
    *confidence = (result >> 8) & 0xFF;
    *is_gunshot = result & 0x01;
}

// Write features to hardware accelerator
void hw_accel_write_features(const float* features) {
    // Convert float to fixed-point (Q1.15 format)
    for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
        int16_t fixed_val = (int16_t)(features[i] * 32768.0f);
        Xil_Out16(detector_base_addr + 0x1000 + i*2, fixed_val);
    }
    
    // Signal that features are ready
    Xil_Out32(detector_base_addr + 0x00, 0x02);
}

// Start inference
void hw_accel_start_inference(void) {
    Xil_Out32(detector_base_addr + 0x00, 0x04);
}

// Get result from hardware accelerator
uint8_t hw_accel_get_result(void) {
    return Xil_In32(detector_base_addr + 0x0C) & 0xFF;
}

// Reset the detector
void gunshot_detector_reset(void) {
    Xil_Out32(detector_base_addr + 0x00, 0x01);
    Xil_Out32(detector_base_addr + 0x00, 0x00);
}

// Get version information
uint32_t gunshot_detector_get_version(void) {
    return Xil_In32(detector_base_addr + 0xFC);
}

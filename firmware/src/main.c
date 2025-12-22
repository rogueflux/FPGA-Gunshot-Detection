#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include "xil_printf.h"
#include "spi_adc_driver.h"
#include "gunshot_detector.h"
#include "music_localization.h"

#define SAMPLE_RATE 8000
#define FRAME_SIZE 2048  // 256ms at 8kHz
#define NUM_CHANNELS 3

// Main processing function for SPI ADC
void process_spi_audio_frame(void) {
    static int16_t audio_buffer[FRAME_SIZE * NUM_CHANNELS];
    static uint32_t frame_count = 0;
    
    // 1. Capture audio from 3 microphones via SPI
    capture_audio_samples(audio_buffer, FRAME_SIZE);
    
    // 2. Extract features
    float features[2052];
    extract_features_from_spi(audio_buffer, FRAME_SIZE, NUM_CHANNELS, features);
    
    // 3. Run neural network inference
    uint8_t confidence;
    uint8_t is_gunshot;
    gunshot_detector_predict(features, &confidence, &is_gunshot);
    
    if (is_gunshot) {
        // 4. Localize using MUSIC algorithm
        int16_t x_pos, y_pos;
        uint8_t loc_confidence;
        music_localize_spi(audio_buffer, &x_pos, &y_pos, &loc_confidence);
        
        // 5. Alert and log
        send_gunshot_alert(confidence, x_pos, y_pos);
        log_detection_event(frame_count, confidence, x_pos, y_pos);
        
        xil_printf("GUNSHOT! Conf: %d%%, Pos: (%d, %d)\r\n", 
                   confidence, x_pos, y_pos);
    }
    
    frame_count++;
}

int main() {
    xil_printf("Gunshot Detection System Starting...\r\n");
    xil_printf("Using SPI ADC with 3 microphones\r\n");
    
    // Initialize subsystems
    spi_adc_init();
    gunshot_detector_init();
    music_localization_init();
    
    // Main loop
    while (1) {
        process_spi_audio_frame();
        
        // Status indicator
        toggle_status_led();
        
        // Check for UART commands
        check_uart_commands();
    }
    
    return 0;
}

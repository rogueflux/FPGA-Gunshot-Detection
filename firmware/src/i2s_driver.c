#include "i2s_driver.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "xparameters.h"

// I2S buffer
static int16_t i2s_buffer[I2S_CHANNELS][I2S_BUFFER_SIZE];
static volatile uint32_t buffer_index = 0;
static volatile uint8_t capture_active = 0;

// Initialize I2S interface
void i2s_init(void) {
    // Reset I2S controller
    Xil_Out32(I2S_CR_REG, I2S_CR_RESET);
    
    // Configure as master, RX enabled
    uint32_t control = I2S_CR_ENABLE | I2S_CR_MASTER | I2S_CR_RX_EN;
    Xil_Out32(I2S_CR_REG, control);
    
    // Clear status register
    Xil_Out32(I2S_SR_REG, 0x00);
    
    buffer_index = 0;
    capture_active = 0;
    
    // Flush cache
    Xil_DCacheFlush();
}

// Start audio capture
void i2s_start_capture(void) {
    capture_active = 1;
    buffer_index = 0;
    
    // Enable RX
    uint32_t control = Xil_In32(I2S_CR_REG);
    control |= I2S_CR_RX_EN;
    Xil_Out32(I2S_CR_REG, control);
}

// Stop audio capture
void i2s_stop_capture(void) {
    capture_active = 0;
    
    // Disable RX
    uint32_t control = Xil_In32(I2S_CR_REG);
    control &= ~I2S_CR_RX_EN;
    Xil_Out32(I2S_CR_REG, control);
}

// Check if data is ready
uint8_t i2s_is_data_ready(void) {
    return (Xil_In32(I2S_SR_REG) & 0x01) ? 1 : 0;
}

// Read data from a specific channel
int16_t i2s_read_channel(uint8_t channel) {
    if (channel >= I2S_CHANNELS) return 0;
    
    // Wait for data ready
    while (!i2s_is_data_ready());
    
    // Read data (simplified - real implementation would handle all channels)
    int16_t data = (int16_t)(Xil_In32(I2S_RX_REG) & 0xFFFF);
    
    return data;
}

// Capture audio to buffer
void i2s_capture_audio(int16_t audio[I2S_CHANNELS][I2S_BUFFER_SIZE]) {
    capture_active = 1;
    buffer_index = 0;
    
    while (buffer_index < I2S_BUFFER_SIZE && capture_active) {
        // Read all 6 channels for this sample
        for (int ch = 0; ch < I2S_CHANNELS; ch++) {
            audio[ch][buffer_index] = i2s_read_channel(ch);
        }
        buffer_index++;
        
        // Add small delay to prevent overrun
        for (volatile int i = 0; i < 10; i++);
    }
    
    capture_active = 0;
}

// Set sample rate
void i2s_set_sample_rate(uint32_t sample_rate) {
    // Calculate clock divider for MCLK = 12.288MHz
    uint32_t divider = 12288000 / (sample_rate * 256); // 256 = MCLK/BCLK ratio
    Xil_Out32(I2S_BASE_ADDR + 0x10, divider);
}

// Set number of channels
void i2s_set_channels(uint8_t num_channels) {
    // Configure channel count (up to 8 channels)
    uint32_t config = Xil_In32(I2S_BASE_ADDR + 0x14);
    config = (config & 0xFFFFFFF8) | (num_channels & 0x07);
    Xil_Out32(I2S_BASE_ADDR + 0x14, config);
}

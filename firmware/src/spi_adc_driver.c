#include "spi_adc_driver.h"
#include "xil_io.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xspips.h"
#include "xil_printf.h"
#include <math.h>
#include <string.h>

// Global driver instance
static SPI_ADC_Driver adc_driver;

// SPI Instance
static XSpi SpiInstance;

// Calibration data
static int8_t calibration_offsets[ADC_NUM_CHANNELS] = {0, 0, 0};
static float calibration_scales[ADC_NUM_CHANNELS] = {1.0f, 1.0f, 1.0f};

// Initialize SPI ADC driver
void spi_adc_init(uint32_t spi_base_addr) {
    XSpi_Config *SpiConfig;
    
    // Initialize driver structure
    memset(&adc_driver, 0, sizeof(SPI_ADC_Driver));
    adc_driver.spi_base_addr = spi_base_addr;
    
    // Initialize SPI
    SpiConfig = XSpi_LookupConfig(XPAR_XSPIPS_0_DEVICE_ID);
    if (SpiConfig == NULL) {
        xil_printf("ERROR: SPI configuration not found\r\n");
        return;
    }
    
    // Initialize SPI driver
    if (XSpi_CfgInitialize(&SpiInstance, SpiConfig, SpiConfig->BaseAddress) != XST_SUCCESS) {
        xil_printf("ERROR: SPI initialization failed\r\n");
        return;
    }
    
    // Set SPI options
    XSpi_SetOptions(&SpiInstance, XSP_MASTER_OPTION | XSP_MANUAL_SSELECT_OPTION);
    
    // Set slave select register (use slave 0)
    XSpi_SetSlaveSelect(&SpiInstance, 0x01);
    
    // Start the SPI driver
    XSpi_Start(&SpiInstance);
    
    // Disable interrupts (polling mode for simplicity)
    XSpi_IntrGlobalDisable(&SpiInstance);
    
    // Initialize ADC channels
    for (int i = 0; i < ADC_NUM_CHANNELS; i++) {
        adc_driver.channels[i].channel = i + 1;
        adc_driver.channels[i].gain = ADC_GAIN_4X;  // Default 4x gain
        adc_driver.channels[i].mode = ADC_MODE_SINGLE;
        adc_driver.channels[i].data_ready = 0;
        adc_driver.channels[i].last_value = 0;
        adc_driver.channels[i].voltage = 0.0f;
        
        // Configure each channel
        spi_adc_configure_channel(i, ADC_GAIN_4X, ADC_MODE_SINGLE);
    }
    
    // Run self-test
    spi_adc_run_self_test();
    
    adc_driver.initialized = 1;
    xil_printf("SPI ADC Driver Initialized\r\n");
    xil_printf("  Channels: %d, Sample Rate: %d Hz\r\n", 
               ADC_NUM_CHANNELS, ADC_SAMPLE_RATE);
    xil_printf("  Resolution: %d-bit, Reference: %.2fV\r\n",
               ADC_BIT_DEPTH, ADC_REF_VOLTAGE);
}

// Configure ADC channel
void spi_adc_configure_channel(uint8_t channel, uint8_t gain, uint8_t mode) {
    if (channel >= ADC_NUM_CHANNELS) return;
    
    uint8_t command = 0;
    
    // Build command byte
    switch (channel) {
        case 0: command = ADC_CMD_SELECT_CH1; break;
        case 1: command = ADC_CMD_SELECT_CH2; break;
        case 2: command = ADC_CMD_SELECT_CH3; break;
        default: return;
    }
    
    // Add gain bits (bits 2-3)
    command |= ((gain & 0x03) << 2);
    
    // Add mode bits (bits 0-1)
    command |= (mode & 0x03);
    
    // Send configuration
    spi_adc_send_command(command);
    
    // Update driver structure
    adc_driver.channels[channel].gain = gain;
    adc_driver.channels[channel].mode = mode;
    
    // Small delay for configuration to take effect
    for (volatile int i = 0; i < 100; i++);
}

// Set sample rate (affects all channels)
void spi_adc_set_sample_rate(uint32_t sample_rate) {
    // For MAX11060, sample rate is controlled by conversion time
    // This is a simplified implementation
    
    uint8_t rate_reg = 0;
    
    if (sample_rate >= 8000) rate_reg = 0x00;      // 8 kHz
    else if (sample_rate >= 4000) rate_reg = 0x01; // 4 kHz
    else if (sample_rate >= 2000) rate_reg = 0x02; // 2 kHz
    else rate_reg = 0x03;                          // 1 kHz
    
    // Write to sample rate register (register 0x02)
    spi_adc_write_register(0x02, rate_reg);
}

// Read single channel
uint8_t spi_adc_read_channel(uint8_t channel) {
    uint8_t tx_buffer[2], rx_buffer[2];
    uint8_t command_byte;
    
    if (channel >= ADC_NUM_CHANNELS) return 0;
    
    // Build command byte
    ADC_Channel *config = &adc_driver.channels[channel];
    command_byte = (config->channel << 4) | (config->gain << 2) | config->mode;
    
    // Prepare transmission buffers
    tx_buffer[0] = command_byte;
    tx_buffer[1] = 0x00;  // Dummy byte for reading
    
    // Assert chip select
    XSpi_SetSlaveSelect(&SpiInstance, 0x01);
    
    // Transfer data
    XSpi_Transfer(&SpiInstance, tx_buffer, rx_buffer, 2);
    
    // Deassert chip select
    XSpi_SetSlaveSelect(&SpiInstance, 0x00);
    
    // Store the result
    uint8_t adc_value = rx_buffer[1];
    config->last_value = adc_value;
    config->voltage = spi_adc_convert_to_voltage(adc_value);
    config->data_ready = 1;
    
    adc_driver.sample_count++;
    
    return adc_value;
}

// Read all channels sequentially
void spi_adc_read_all_channels(uint8_t* channel_data) {
    for (int ch = 0; ch < ADC_NUM_CHANNELS; ch++) {
        channel_data[ch] = spi_adc_read_channel(ch);
        
        // Small delay between channels
        for (volatile int i = 0; i < 10; i++);
    }
}

// Capture multiple samples to buffer
void spi_adc_capture_samples(int16_t* buffer, uint32_t num_samples) {
    uint32_t sample_count = 0;
    uint8_t channel_data[ADC_NUM_CHANNELS];
    
    while (sample_count < num_samples) {
        // Read all channels
        spi_adc_read_all_channels(channel_data);
        
        // Convert and store in buffer
        for (int ch = 0; ch < ADC_NUM_CHANNELS; ch++) {
            // Apply calibration offset
            int8_t calibrated_value = channel_data[ch] + calibration_offsets[ch];
            
            // Convert to 16-bit audio sample
            int16_t sample = spi_adc_convert_to_audio_sample(calibrated_value);
            
            // Store in interleaved format
            buffer[sample_count * ADC_NUM_CHANNELS + ch] = sample;
        }
        
        sample_count++;
        
        // Wait for next sample period (125µs for 8kHz)
        // Using busy wait for simplicity - in real implementation use timer
        for (volatile int i = 0; i < 1250; i++); // Approximate delay
    }
}

// Convert ADC value to voltage
float spi_adc_convert_to_voltage(uint8_t adc_value) {
    // 8-bit ADC with 5V reference
    float lsb_voltage = ADC_REF_VOLTAGE / 256.0f;
    return adc_value * lsb_voltage;
}

// Convert ADC value to 16-bit audio sample
int16_t spi_adc_convert_to_audio_sample(uint8_t adc_value) {
    // Convert 8-bit unsigned to 16-bit signed
    // 0 -> -32768, 128 -> 0, 255 -> 32767
    int16_t sample = ((int16_t)adc_value - 128) * 256;
    return sample;
}

// Get voltage for a specific channel
float spi_adc_get_channel_voltage(uint8_t channel) {
    if (channel >= ADC_NUM_CHANNELS) return 0.0f;
    return adc_driver.channels[channel].voltage;
}

// Check if data is ready
uint8_t spi_adc_is_data_ready(void) {
    // Poll status register (simplified)
    uint8_t status = spi_adc_read_register(0x00);
    return (status & 0x80) ? 1 : 0;
}

// Reset ADC
void spi_adc_reset(void) {
    // Send reset command
    spi_adc_send_command(0xFF);
    
    // Reconfigure all channels
    for (int i = 0; i < ADC_NUM_CHANNELS; i++) {
        spi_adc_configure_channel(i, adc_driver.channels[i].gain, 
                                  adc_driver.channels[i].mode);
    }
    
    adc_driver.sample_count = 0;
    adc_driver.error_count = 0;
}

// Get sample count
uint32_t spi_adc_get_sample_count(void) {
    return adc_driver.sample_count;
}

// Get error count
uint32_t spi_adc_get_error_count(void) {
    return adc_driver.error_count;
}

// Calculate RMS voltage for a channel
float spi_adc_get_channel_rms(uint8_t channel, uint32_t num_samples) {
    if (channel >= ADC_NUM_CHANNELS || num_samples == 0) return 0.0f;
    
    float sum_squares = 0.0f;
    
    // Capture samples
    for (uint32_t i = 0; i < num_samples; i++) {
        uint8_t adc_value = spi_adc_read_channel(channel);
        float voltage = spi_adc_convert_to_voltage(adc_value);
        
        // Remove DC offset (assume 2.5V for biasing)
        float ac_voltage = voltage - 2.5f;
        sum_squares += ac_voltage * ac_voltage;
        
        // Delay between samples
        for (volatile int j = 0; j < 1250; j++);
    }
    
    return sqrtf(sum_squares / num_samples);
}

// Run self-test
void spi_adc_run_self_test(void) {
    xil_printf("Running ADC Self-Test...\r\n");
    
    uint8_t test_values[3] = {0x80, 0x40, 0xC0}; // Test patterns
    
    for (int ch = 0; ch < ADC_NUM_CHANNELS; ch++) {
        xil_printf("  Testing Channel %d: ", ch + 1);
        
        // Read multiple samples
        float sum = 0.0f;
        int num_test_samples = 10;
        
        for (int i = 0; i < num_test_samples; i++) {
            uint8_t value = spi_adc_read_channel(ch);
            sum += value;
            
            // Delay between reads
            for (volatile int j = 0; j < 1000; j++);
        }
        
        float avg = sum / num_test_samples;
        
        if (avg > 10 && avg < 245) { // Reasonable range
            xil_printf("PASS (avg=%.1f)\r\n", avg);
        } else {
            xil_printf("FAIL (avg=%.1f)\r\n", avg);
            adc_driver.error_count++;
        }
    }
    
    xil_printf("Self-test complete. Errors: %d\r\n", adc_driver.error_count);
}

// Print driver status
void spi_adc_print_status(void) {
    xil_printf("=== SPI ADC Status ===\r\n");
    xil_printf("Initialized: %s\r\n", adc_driver.initialized ? "Yes" : "No");
    xil_printf("Sample Count: %lu\r\n", adc_driver.sample_count);
    xil_printf("Error Count: %lu\r\n", adc_driver.error_count);
    
    for (int i = 0; i < ADC_NUM_CHANNELS; i++) {
        ADC_Channel *ch = &adc_driver.channels[i];
        xil_printf("Channel %d: Gain=%dx, Mode=%s, Last Value=%d (%.3fV)\r\n",
                   ch->channel,
                   1 << ch->gain,
                   ch->mode == ADC_MODE_SINGLE ? "Single" : "Continuous",
                   ch->last_value,
                   ch->voltage);
    }
    xil_printf("=====================\r\n");
}

// Low-level SPI functions

// Send command to ADC
void spi_adc_send_command(uint8_t command) {
    uint8_t tx_buffer[1] = {command};
    uint8_t rx_buffer[1];
    
    XSpi_SetSlaveSelect(&SpiInstance, 0x01);
    XSpi_Transfer(&SpiInstance, tx_buffer, rx_buffer, 1);
    XSpi_SetSlaveSelect(&SpiInstance, 0x00);
}

// Read ADC register
uint8_t spi_adc_read_register(uint8_t reg_addr) {
    uint8_t tx_buffer[2] = {0x80 | reg_addr, 0x00}; // Read command
    uint8_t rx_buffer[2];
    
    XSpi_SetSlaveSelect(&SpiInstance, 0x01);
    XSpi_Transfer(&SpiInstance, tx_buffer, rx_buffer, 2);
    XSpi_SetSlaveSelect(&SpiInstance, 0x00);
    
    return rx_buffer[1];
}

// Write ADC register
void spi_adc_write_register(uint8_t reg_addr, uint8_t value) {
    uint8_t tx_buffer[2] = {reg_addr, value}; // Write command
    uint8_t rx_buffer[2];
    
    XSpi_SetSlaveSelect(&SpiInstance, 0x01);
    XSpi_Transfer(&SpiInstance, tx_buffer, rx_buffer, 2);
    XSpi_SetSlaveSelect(&SpiInstance, 0x00);
}

// Calculate noise floor
float spi_adc_calculate_noise_floor(uint8_t channel, uint32_t num_samples) {
    if (channel >= ADC_NUM_CHANNELS || num_samples < 10) return 0.0f;
    
    float sum = 0.0f;
    float sum_squares = 0.0f;
    
    // Capture samples with input shorted (or silent)
    for (uint32_t i = 0; i < num_samples; i++) {
        uint8_t value = spi_adc_read_channel(channel);
        sum += value;
        sum_squares += value * value;
        
        for (volatile int j = 0; j < 1250; j++);
    }
    
    float mean = sum / num_samples;
    float variance = (sum_squares / num_samples) - (mean * mean);
    
    return sqrtf(variance) * (ADC_REF_VOLTAGE / 256.0f);
}

// Calculate SNR
float spi_adc_calculate_snr(uint8_t channel, uint32_t num_samples) {
    float noise_floor = spi_adc_calculate_noise_floor(channel, num_samples);
    
    if (noise_floor < 0.001f) return 0.0f; // Avoid division by zero
    
    // Assume full-scale signal is reference voltage
    float full_scale = ADC_REF_VOLTAGE / 2.0f; // Peak AC voltage
    
    return 20.0f * log10f(full_scale / noise_floor);
}

// Apply calibration offset
void spi_adc_apply_calibration_offset(uint8_t channel, int8_t offset) {
    if (channel < ADC_NUM_CHANNELS) {
        calibration_offsets[channel] = offset;
    }
}

// Start continuous sampling mode
void spi_adc_start_continuous_mode(void) {
    for (int i = 0; i < ADC_NUM_CHANNELS; i++) {
        spi_adc_configure_channel(i, adc_driver.channels[i].gain, ADC_MODE_CONTINUOUS);
    }
}

// Stop continuous sampling mode
void spi_adc_stop_continuous_mode(void) {
    for (int i = 0; i < ADC_NUM_CHANNELS; i++) {
        spi_adc_configure_channel(i, adc_driver.channels[i].gain, ADC_MODE_SINGLE);
    }
}

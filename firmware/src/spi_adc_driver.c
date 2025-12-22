#include "spi_adc_driver.h"
#include "platform_config.h"

// MAX11060 ADC Configuration
#define ADC_CHANNELS 3
#define ADC_RESOLUTION 8
#define ADC_REF_VOLTAGE 5.0f
#define ADC_SAMPLE_RATE 8000  // 8 kHz

// SPI Configuration
static XIicPs IicInstance;
static XSpi SpiInstance;

// ADC Channel Configuration
typedef struct {
    uint8_t channel;      // AIN1, AIN2, AIN3
    uint8_t gain;         // 1x, 2x, 4x, 8x
    uint8_t conversion_mode; // Single-shot or continuous
} ADC_Config;

static ADC_Config adc_config[ADC_CHANNELS] = {
    {0x01, 0x02, 0x01},  // Channel 1: AIN1, 4x gain, single-shot
    {0x02, 0x02, 0x01},  // Channel 2: AIN2, 4x gain, single-shot
    {0x03, 0x02, 0x01}   // Channel 3: AIN3, 4x gain, single-shot
};

void spi_adc_init(void) {
    XSpi_Config *SpiConfig;
    
    // Initialize SPI
    SpiConfig = XSpi_LookupConfig(XPAR_SPI_0_DEVICE_ID);
    XSpi_CfgInitialize(&SpiInstance, SpiConfig, SpiConfig->BaseAddress);
    
    // Set SPI options
    XSpi_SetOptions(&SpiInstance, XSP_MASTER_OPTION | XSP_MANUAL_SSELECT_OPTION);
    XSpi_Start(&SpiInstance);
    XSpi_IntrGlobalDisable(&SpiInstance);
    
    xil_printf("SPI ADC Interface Initialized\r\n");
}

uint8_t read_adc_channel(uint8_t channel) {
    uint8_t tx_buffer[2], rx_buffer[2];
    uint8_t command_byte;
    
    if (channel >= ADC_CHANNELS) return 0;
    
    // Construct command byte
    ADC_Config *config = &adc_config[channel];
    command_byte = (config->channel << 4) | (config->gain << 2) | config->conversion_mode;
    
    // Prepare transmission
    tx_buffer[0] = command_byte;
    tx_buffer[1] = 0x00;  // Dummy byte for reading
    
    // Assert chip select
    XSpi_SetSlaveSelect(&SpiInstance, 0x01);
    
    // Transfer data
    XSpi_Transfer(&SpiInstance, tx_buffer, rx_buffer, 2);
    
    // Deassert chip select
    XSpi_SetSlaveSelect(&SpiInstance, 0x00);
    
    return rx_buffer[1];  // ADC conversion result
}

void capture_audio_samples(int16_t *buffer, uint32_t num_samples) {
    uint32_t sample_count = 0;
    
    while (sample_count < num_samples) {
        for (int ch = 0; ch < ADC_CHANNELS; ch++) {
            uint8_t adc_value = read_adc_channel(ch);
            
            // Convert 8-bit to 16-bit signed
            int16_t sample = ((int16_t)adc_value - 128) * 256;
            buffer[sample_count * ADC_CHANNELS + ch] = sample;
        }
        
        sample_count++;
        
        // Wait for next sample period (125µs for 8kHz)
        usleep(125);
    }
}

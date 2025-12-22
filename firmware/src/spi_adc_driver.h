#ifndef SPI_ADC_DRIVER_H
#define SPI_ADC_DRIVER_H

#include <stdint.h>
#include "platform_config.h"

// MAX11060 ADC Configuration
#define ADC_NUM_CHANNELS 3
#define ADC_BIT_DEPTH 8
#define ADC_REF_VOLTAGE 5.0f
#define ADC_SAMPLE_RATE 8000  // 8 kHz
#define ADC_SPI_CLOCK_RATE 1000000  // 1 MHz SPI clock

// MAX11060 Register Addresses
#define ADC_CMD_SELECT_CH1  0x10  // Select channel 1
#define ADC_CMD_SELECT_CH2  0x20  // Select channel 2
#define ADC_CMD_SELECT_CH3  0x30  // Select channel 3

// ADC Gain Settings
#define ADC_GAIN_1X  0x00
#define ADC_GAIN_2X  0x01
#define ADC_GAIN_4X  0x02
#define ADC_GAIN_8X  0x03

// Conversion Modes
#define ADC_MODE_SINGLE     0x00
#define ADC_MODE_CONTINUOUS 0x01
#define ADC_MODE_POWER_DOWN 0x02

// ADC Channel Configuration Structure
typedef struct {
    uint8_t channel;        // ADC channel (1-3)
    uint8_t gain;           // Gain setting (0-3)
    uint8_t mode;           // Conversion mode
    uint8_t data_ready;     // Data ready flag
    uint16_t last_value;    // Last converted value
    float voltage;          // Converted voltage
} ADC_Channel;

// SPI ADC Driver Structure
typedef struct {
    ADC_Channel channels[ADC_NUM_CHANNELS];
    uint32_t spi_base_addr;
    uint32_t sample_count;
    uint32_t error_count;
    uint8_t initialized;
} SPI_ADC_Driver;

// Function Prototypes

// Initialization
void spi_adc_init(uint32_t spi_base_addr);
void spi_adc_configure_channel(uint8_t channel, uint8_t gain, uint8_t mode);
void spi_adc_set_sample_rate(uint32_t sample_rate);
void spi_adc_calibrate(void);

// Data Acquisition
uint8_t spi_adc_read_channel(uint8_t channel);
void spi_adc_read_all_channels(uint8_t* channel_data);
void spi_adc_capture_samples(int16_t* buffer, uint32_t num_samples);
void spi_adc_start_continuous_mode(void);
void spi_adc_stop_continuous_mode(void);

// Data Conversion
float spi_adc_convert_to_voltage(uint8_t adc_value);
int16_t spi_adc_convert_to_audio_sample(uint8_t adc_value);
float spi_adc_get_channel_voltage(uint8_t channel);

// Status and Control
uint8_t spi_adc_is_data_ready(void);
void spi_adc_reset(void);
uint32_t spi_adc_get_sample_count(void);
uint32_t spi_adc_get_error_count(void);
float spi_adc_get_channel_rms(uint8_t channel, uint32_t num_samples);

// Diagnostic Functions
void spi_adc_run_self_test(void);
void spi_adc_print_status(void);
void spi_adc_dump_registers(void);

// SPI Low-Level Functions
void spi_adc_send_command(uint8_t command);
uint8_t spi_adc_read_register(uint8_t reg_addr);
void spi_adc_write_register(uint8_t reg_addr, uint8_t value);

// Interrupt Handlers (if using interrupts)
void spi_adc_interrupt_handler(void);
void spi_adc_enable_interrupts(void);
void spi_adc_disable_interrupts(void);

// Utility Functions
float spi_adc_calculate_noise_floor(uint8_t channel, uint32_t num_samples);
float spi_adc_calculate_snr(uint8_t channel, uint32_t num_samples);
void spi_adc_apply_calibration_offset(uint8_t channel, int8_t offset);

#endif // SPI_ADC_DRIVER_H

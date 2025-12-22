#ifndef I2S_DRIVER_H
#define I2S_DRIVER_H

#include <stdint.h>
#include "platform_config.h"

// I2S Configuration
#define I2S_SAMPLE_RATE 16000
#define I2S_BIT_DEPTH 16
#define I2S_CHANNELS 6
#define I2S_BUFFER_SIZE 32000  // 2 seconds at 16kHz

// I2S Register Addresses
#define I2S_BASE_ADDR 0xE0004000
#define I2S_CR_REG (I2S_BASE_ADDR + 0x00)
#define I2S_SR_REG (I2S_BASE_ADDR + 0x04)
#define I2S_TX_REG (I2S_BASE_ADDR + 0x08)
#define I2S_RX_REG (I2S_BASE_ADDR + 0x0C)

// I2S Control Register Bits
#define I2S_CR_ENABLE   (1 << 0)
#define I2S_CR_TX_EN    (1 << 1)
#define I2S_CR_RX_EN    (1 << 2)
#define I2S_CR_RESET    (1 << 3)
#define I2S_CR_MASTER   (1 << 4)

// Function Prototypes
void i2s_init(void);
void i2s_start_capture(void);
void i2s_stop_capture(void);
uint8_t i2s_is_data_ready(void);
int16_t i2s_read_channel(uint8_t channel);
void i2s_capture_audio(int16_t audio[I2S_CHANNELS][I2S_BUFFER_SIZE]);
void i2s_set_sample_rate(uint32_t sample_rate);
void i2s_set_channels(uint8_t num_channels);

#endif // I2S_DRIVER_H

#ifndef PLATFORM_CONFIG_H
#define PLATFORM_CONFIG_H

#include "xparameters.h"

// Platform-specific configuration for Spartan Edge-7 Accelerator Board

// Clock frequencies
#define CPU_CLOCK_FREQ_HZ 666666667
#define FPGA_CLOCK_FREQ_HZ 100000000
#define I2S_CLOCK_FREQ_HZ 12288000

// Memory addresses
#define DDR_BASEADDR XPAR_DDR_MEM_BASEADDR
#define DDR_HIGHADDR XPAR_DDR_MEM_HIGHADDR

// Peripheral addresses
#define UART_BASEADDR XPAR_XUARTPS_0_BASEADDR
#define SPI_BASEADDR XPAR_XSPIPS_0_BASEADDR
#define GPIO_BASEADDR XPAR_GPIO_0_BASEADDR
#define TIMER_BASEADDR XPAR_XSCUTIMER_0_BASEADDR

// GPIO LED pins
#define LED_ALERT_PIN 0
#define LED_STATUS_PIN 1

// SPI configuration
#define SPI_DEVICE_ID XPAR_XSPIPS_0_DEVICE_ID
#define SPI_CLOCK_FREQ 10000000  // 10 MHz

// ADC configuration (MAX11060)
#define ADC_NUM_CHANNELS 3
#define ADC_BIT_DEPTH 8
#define ADC_REF_VOLTAGE 5.0f
#define ADC_SAMPLE_RATE 8000
#define ADC_SPI_CLOCK_RATE 1000000  // 1 MHz for ADC

// Audio processing parameters
#define AUDIO_SAMPLE_RATE 16000
#define AUDIO_FRAME_SIZE 2048  // 128ms at 16kHz
#define AUDIO_BUFFER_SIZE 32000  // 2 seconds

// Detection parameters
#define GUNSHOT_DETECTION_THRESHOLD 0.65f
#define MIN_DETECTION_CONFIDENCE 0.5f
#define MAX_EVENTS_STORED 512

// System parameters
#define SYSTEM_UART_BAUDRATE 115200
#define SYSTEM_TICK_RATE_HZ 1000
#define SYSTEM_WATCHDOG_TIMEOUT_MS 5000

// Feature extraction
#define NUM_MFCC_COEFFS 13
#define NUM_SPECTRAL_FEATURES 40
#define TOTAL_FEATURES 2052

// Error codes
#define ERROR_NONE 0
#define ERROR_HARDWARE_FAILURE -1
#define ERROR_MEMORY_ALLOCATION -2
#define ERROR_PERIPHERAL_INIT -3
#define ERROR_COMMUNICATION -4
#define ERROR_INVALID_PARAMETER -5

// SPI specific error codes
#define ERROR_SPI_INIT_FAILED -10
#define ERROR_SPI_TRANSFER_FAILED -11
#define ERROR_SPI_TIMEOUT -12
#define ERROR_ADC_NOT_RESPONDING -13

// Type definitions for fixed-point arithmetic
typedef int16_t audio_sample_t;
typedef int32_t audio_accum_t;
typedef float audio_float_t;

// System status structure
typedef struct {
    uint32_t uptime_seconds;
    uint32_t total_frames_processed;
    uint32_t detections_count;
    uint32_t false_positives;
    uint8_t system_status;
    int8_t temperature;
    uint16_t free_memory_kb;
    uint32_t spi_sample_count;
    uint32_t spi_error_count;
} SystemStatus;

// Function prototypes
void platform_init(void);
uint32_t get_system_tick(void);
void delay_ms(uint32_t ms);
void system_reset(void);
uint32_t get_free_memory(void);

// Debug utilities
#ifdef DEBUG
    #define DEBUG_PRINT(fmt, ...) xil_printf("[DEBUG] " fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

#define ERROR_PRINT(fmt, ...) xil_printf("[ERROR] " fmt, ##__VA_ARGS__)
#define INFO_PRINT(fmt, ...) xil_printf("[INFO] " fmt, ##__VA_ARGS__)

// SPI-specific debug
#define SPI_DEBUG_PRINT(fmt, ...) xil_printf("[SPI] " fmt, ##__VA_ARGS__)

#endif // PLATFORM_CONFIG_H

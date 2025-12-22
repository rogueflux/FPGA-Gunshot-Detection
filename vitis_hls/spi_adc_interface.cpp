#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

// SPI ADC Interface for MAX11060
// 3-channel, 8-bit ADC with SPI interface

// SPI Configuration
#define SPI_CLOCK_DIVIDER 10  // For 100MHz -> 10MHz SPI clock
#define ADC_NUM_CHANNELS 3
#define ADC_BIT_DEPTH 8
#define SAMPLE_RATE 8000
#define SAMPLES_PER_FRAME 1024

// Fixed-point types
typedef ap_fixed<16, 1> adc_sample_t;  // ADC sample (8-bit extended to 16-bit)
typedef ap_fixed<16, 6> audio_sample_t; // Processed audio sample

// SPI Command Definitions (MAX11060)
#define ADC_CMD_CH1 0x10  // Select channel 1
#define ADC_CMD_CH2 0x20  // Select channel 2
#define ADC_CMD_CH3 0x30  // Select channel 3
#define ADC_GAIN_4X 0x02  // 4x gain setting
#define ADC_MODE_SINGLE 0x01

// Structure for ADC configuration
struct ADC_Config {
    ap_uint<8> channel;
    ap_uint<8> gain;
    ap_uint<8> mode;
};

// Structure for processed audio frame
struct AudioFrame {
    audio_sample_t samples[SAMPLES_PER_FRAME][ADC_NUM_CHANNELS];
    ap_uint<32> timestamp;
    ap_uint<8> channel_status;
};

// SPI Master Interface
void spi_master(
    // Control signals
    ap_uint<1> start,
    ap_uint<1> reset,
    
    // SPI physical interface
    ap_uint<1>& spi_clk,
    ap_uint<1>& spi_mosi,
    ap_uint<1>& spi_cs,
    ap_uint<1> spi_miso,
    
    // Data interface
    ap_uint<8> tx_data,
    ap_uint<8>& rx_data,
    
    // Status
    ap_uint<1>& busy,
    ap_uint<1>& data_ready
) {
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE ap_none port=start
    #pragma HLS INTERFACE ap_none port=reset
    #pragma HLS INTERFACE ap_none port=spi_clk
    #pragma HLS INTERFACE ap_none port=spi_mosi
    #pragma HLS INTERFACE ap_none port=spi_cs
    #pragma HLS INTERFACE ap_none port=spi_miso
    #pragma HLS INTERFACE ap_none port=tx_data
    #pragma HLS INTERFACE ap_none port=rx_data
    #pragma HLS INTERFACE ap_none port=busy
    #pragma HLS INTERFACE ap_none port=data_ready
    
    static ap_uint<4> bit_counter = 0;
    static ap_uint<8> shift_tx = 0;
    static ap_uint<8> shift_rx = 0;
    static ap_uint<1> internal_busy = 0;
    static ap_uint<16> clock_counter = 0;
    
    // Reset logic
    if (reset) {
        bit_counter = 0;
        shift_tx = 0;
        shift_rx = 0;
        internal_busy = 0;
        clock_counter = 0;
        spi_cs = 1;
        spi_clk = 0;
        spi_mosi = 0;
        data_ready = 0;
        busy = 0;
        return;
    }
    
    busy = internal_busy;
    
    // Idle state
    if (!start && !internal_busy) {
        spi_cs = 1;
        spi_clk = 0;
        spi_mosi = 0;
        data_ready = 0;
        return;
    }
    
    // Start new transaction
    if (start && !internal_busy) {
        internal_busy = 1;
        bit_counter = 0;
        shift_tx = tx_data;
        shift_rx = 0;
        spi_cs = 0;
        spi_clk = 0;
        data_ready = 0;
        clock_counter = 0;
        return;
    }
    
    // SPI clock generation (divide by SPI_CLOCK_DIVIDER)
    if (internal_busy) {
        clock_counter++;
        
        if (clock_counter == SPI_CLOCK_DIVIDER / 2) {
            // Rising edge - sample MISO
            spi_clk = 1;
            shift_rx = (shift_rx << 1) | spi_miso;
        }
        else if (clock_counter >= SPI_CLOCK_DIVIDER) {
            // Falling edge - setup MOSI
            spi_clk = 0;
            spi_mosi = shift_tx[7];  // MSB first
            shift_tx = shift_tx << 1;
            
            clock_counter = 0;
            bit_counter++;
            
            // Check if all bits transmitted
            if (bit_counter >= 8) {
                // End of transaction
                spi_cs = 1;
                spi_mosi = 0;
                internal_busy = 0;
                rx_data = shift_rx;
                data_ready = 1;
            }
        }
    }
}

// ADC Controller
void adc_controller(
    // Control signals
    ap_uint<1> start_capture,
    ap_uint<1> reset,
    
    // SPI interface
    ap_uint<1>& spi_start,
    ap_uint<8>& spi_tx_data,
    ap_uint<8> spi_rx_data,
    ap_uint<1> spi_busy,
    ap_uint<1> spi_data_ready,
    
    // ADC configuration
    ADC_Config config[ADC_NUM_CHANNELS],
    
    // Data output
    hls::stream<adc_sample_t>& adc_data_out,
    
    // Status
    ap_uint<1>& capture_active,
    ap_uint<32>& sample_count,
    ap_uint<8>& error_count
) {
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE ap_none port=start_capture
    #pragma HLS INTERFACE ap_none port=reset
    #pragma HLS INTERFACE ap_none port=spi_start
    #pragma HLS INTERFACE ap_none port=spi_tx_data
    #pragma HLS INTERFACE ap_none port=spi_rx_data
    #pragma HLS INTERFACE ap_none port=spi_busy
    #pragma HLS INTERFACE ap_none port=spi_data_ready
    #pragma HLS INTERFACE ap_none port=capture_active
    #pragma HLS INTERFACE ap_none port=sample_count
    #pragma HLS INTERFACE ap_none port=error_count
    #pragma HLS INTERFACE axis port=adc_data_out
    
    static enum {
        IDLE,
        INIT_ADC,
        SELECT_CHANNEL,
        START_CONVERSION,
        READ_DATA,
        PROCESS_DATA,
        WAIT_NEXT_SAMPLE
    } state = IDLE;
    
    static ap_uint<8> current_channel = 0;
    static ap_uint<32> internal_sample_count = 0;
    static ap_uint<8> internal_error_count = 0;
    static ap_uint<1> internal_capture_active = 0;
    static ap_uint<16> sample_timer = 0;
    static ap_uint<8> command_byte = 0;
    
    // Update outputs
    capture_active = internal_capture_active;
    sample_count = internal_sample_count;
    error_count = internal_error_count;
    
    // Reset logic
    if (reset) {
        state = IDLE;
        current_channel = 0;
        internal_sample_count = 0;
        internal_error_count = 0;
        internal_capture_active = 0;
        sample_timer = 0;
        spi_start = 0;
        return;
    }
    
    // State machine
    switch (state) {
        case IDLE:
            spi_start = 0;
            if (start_capture) {
                internal_capture_active = 1;
                state = INIT_ADC;
                current_channel = 0;
                internal_sample_count = 0;
            }
            break;
            
        case INIT_ADC:
            // Initialize ADC (send reset command)
            command_byte = 0xFF;  // Reset command
            spi_tx_data = command_byte;
            spi_start = 1;
            state = SELECT_CHANNEL;
            break;
            
        case SELECT_CHANNEL:
            if (!spi_busy) {
                spi_start = 0;
                
                // Build command byte for current channel
                ADC_Config cfg = config[current_channel];
                command_byte = (cfg.channel << 4) | (cfg.gain << 2) | cfg.mode;
                
                spi_tx_data = command_byte;
                spi_start = 1;
                state = START_CONVERSION;
            }
            break;
            
        case START_CONVERSION:
            if (!spi_busy) {
                spi_start = 0;
                // Wait for conversion (simplified)
                sample_timer = 125;  // 125 clocks for 8kHz @ 1MHz SPI
                state = READ_DATA;
            }
            break;
            
        case READ_DATA:
            if (sample_timer > 0) {
                sample_timer--;
            } else {
                // Send dummy byte to read result
                spi_tx_data = 0x00;
                spi_start = 1;
                state = PROCESS_DATA;
            }
            break;
            
        case PROCESS_DATA:
            if (spi_data_ready) {
                spi_start = 0;
                
                // Read ADC value
                ap_uint<8> adc_value = spi_rx_data;
                
                // Convert to signed fixed-point
                adc_sample_t sample;
                sample = (adc_sample_t)adc_value - 128.0;  // Convert to signed
                sample = sample / 128.0;  // Normalize to [-1, 1]
                
                // Write to output stream
                adc_data_out.write(sample);
                
                // Move to next channel
                current_channel++;
                if (current_channel >= ADC_NUM_CHANNELS) {
                    current_channel = 0;
                    internal_sample_count++;
                    
                    // Check if we have enough samples
                    if (internal_sample_count >= SAMPLES_PER_FRAME) {
                        internal_capture_active = 0;
                        state = IDLE;
                    } else {
                        state = WAIT_NEXT_SAMPLE;
                    }
                } else {
                    state = SELECT_CHANNEL;
                }
            }
            break;
            
        case WAIT_NEXT_SAMPLE:
            // Wait for next sample period (125µs for 8kHz)
            sample_timer = 1250;  // For 100MHz clock
            if (sample_timer == 0) {
                state = SELECT_CHANNEL;
            }
            break;
    }
}

// Audio Preprocessor
void audio_preprocessor(
    // Input from ADC
    hls::stream<adc_sample_t>& adc_data_in,
    
    // Output to detector
    hls::stream<audio_sample_t>& audio_data_out,
    
    // Control
    ap_uint<1> enable,
    ap_uint<1> reset,
    
    // Configuration
    ap_uint<8> gain_factor,
    
    // Status
    ap_uint<1>& processing_active,
    ap_uint<32>& processed_count
) {
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE axis port=adc_data_in
    #pragma HLS INTERFACE axis port=audio_data_out
    #pragma HLS INTERFACE ap_none port=enable
    #pragma HLS INTERFACE ap_none port=reset
    #pragma HLS INTERFACE ap_none port=gain_factor
    #pragma HLS INTERFACE ap_none port=processing_active
    #pragma HLS INTERFACE ap_none port=processed_count
    
    static audio_sample_t dc_offset[ADC_NUM_CHANNELS] = {0};
    static audio_sample_t channel_gain[ADC_NUM_CHANNELS] = {1.0, 1.0, 1.0};
    static ap_uint<32> internal_processed_count = 0;
    static ap_uint<1> internal_processing_active = 0;
    
    // Update outputs
    processing_active = internal_processing_active;
    processed_count = internal_processed_count;
    
    // Reset logic
    if (reset) {
        for (int i = 0; i < ADC_NUM_CHANNELS; i++) {
            dc_offset[i] = 0;
            channel_gain[i] = 1.0;
        }
        internal_processed_count = 0;
        internal_processing_active = 0;
        return;
    }
    
    if (!enable) {
        internal_processing_active = 0;
        return;
    }
    
    internal_processing_active = 1;
    
    // Process incoming samples
    if (!adc_data_in.empty()) {
        adc_sample_t raw_sample = adc_data_in.read();
        
        // Apply DC offset removal (simplified high-pass filter)
        static audio_sample_t prev_sample[ADC_NUM_CHANNELS] = {0};
        static ap_uint<8> channel_idx = 0;
        
        audio_sample_t processed;
        
        // Simple high-pass: y[n] = x[n] - x[n-1]
        processed = raw_sample - prev_sample[channel_idx];
        prev_sample[channel_idx] = raw_sample;
        
        // Apply gain
        processed = processed * channel_gain[channel_idx] * gain_factor;
        
        // Clamp to [-1, 1]
        if (processed > 1.0) processed = 1.0;
        if (processed < -1.0) processed = -1.0;
        
        // Write to output
        audio_data_out.write(processed);
        
        // Update channel index
        channel_idx++;
        if (channel_idx >= ADC_NUM_CHANNELS) {
            channel_idx = 0;
            internal_processed_count++;
        }
    }
}

// Complete SPI ADC Interface
void spi_adc_interface_complete(
    // Control signals
    ap_uint<1> start,
    ap_uint<1> reset,
    
    // SPI physical interface
    ap_uint<1>& spi_clk,
    ap_uint<1>& spi_mosi,
    ap_uint<1>& spi_cs,
    ap_uint<1> spi_miso,
    
    // Audio output
    hls::stream<audio_sample_t>& audio_out,
    
    // Status
    ap_uint<1>& ready,
    ap_uint<1>& data_valid,
    ap_uint<32>& sample_counter,
    ap_uint<8>& error_counter
) {
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE ap_none port=start
    #pragma HLS INTERFACE ap_none port=reset
    #pragma HLS INTERFACE ap_none port=spi_clk
    #pragma HLS INTERFACE ap_none port=spi_mosi
    #pragma HLS INTERFACE ap_none port=spi_cs
    #pragma HLS INTERFACE ap_none port=spi_miso
    #pragma HLS INTERFACE axis port=audio_out
    #pragma HLS INTERFACE ap_none port=ready
    #pragma HLS INTERFACE ap_none port=data_valid
    #pragma HLS INTERFACE ap_none port=sample_counter
    #pragma HLS INTERFACE ap_none port=error_counter
    
    #pragma HLS DATAFLOW
    
    // Internal signals
    static ap_uint<1> spi_start_signal = 0;
    static ap_uint<8> spi_tx_data_signal = 0;
    static ap_uint<8> spi_rx_data_signal = 0;
    static ap_uint<1> spi_busy_signal = 0;
    static ap_uint<1> spi_data_ready_signal = 0;
    
    static hls::stream<adc_sample_t> adc_raw_data;
    #pragma HLS STREAM variable=adc_raw_data depth=32
    
    static ap_uint<1> capture_active_signal = 0;
    static ap_uint<32> adc_sample_count_signal = 0;
    static ap_uint<8> adc_error_count_signal = 0;
    
    static ap_uint<1> processing_active_signal = 0;
    static ap_uint<32> processed_count_signal = 0;
    
    // ADC configuration
    static ADC_Config adc_config[ADC_NUM_CHANNELS] = {
        {0x01, ADC_GAIN_4X, ADC_MODE_SINGLE},  // Channel 1
        {0x02, ADC_GAIN_4X, ADC_MODE_SINGLE},  // Channel 2
        {0x03, ADC_GAIN_4X, ADC_MODE_SINGLE}   // Channel 3
    };
    
    // SPI Master
    spi_master(
        spi_start_signal,
        reset,
        spi_clk,
        spi_mosi,
        spi_cs,
        spi_miso,
        spi_tx_data_signal,
        spi_rx_data_signal,
        spi_busy_signal,
        spi_data_ready_signal
    );
    
    // ADC Controller
    adc_controller(
        start,
        reset,
        spi_start_signal,
        spi_tx_data_signal,
        spi_rx_data_signal,
        spi_busy_signal,
        spi_data_ready_signal,
        adc_config,
        adc_raw_data,
        capture_active_signal,
        adc_sample_count_signal,
        adc_error_count_signal
    );
    
    // Audio Preprocessor
    audio_preprocessor(
        adc_raw_data,
        audio_out,
        capture_active_signal,
        reset,
        2,  // gain_factor = 2
        processing_active_signal,
        processed_count_signal
    );
    
    // Status outputs
    ready = !capture_active_signal && !processing_active_signal;
    data_valid = !audio_out.empty();
    sample_counter = processed_count_signal;
    error_counter = adc_error_count_signal;
}

// Testbench for SPI ADC Interface
#ifdef __SYNTHESIS__
#else
#include <iostream>
#include <fstream>

void test_spi_adc_interface() {
    std::cout << "Testing SPI ADC Interface\n";
    std::cout << "=========================\n";
    
    // Test signals
    ap_uint<1> start = 0;
    ap_uint<1> reset = 1;
    ap_uint<1> spi_clk;
    ap_uint<1> spi_mosi;
    ap_uint<1> spi_cs;
    ap_uint<1> spi_miso = 0;
    
    hls::stream<audio_sample_t> audio_out;
    ap_uint<1> ready;
    ap_uint<1> data_valid;
    ap_uint<32> sample_counter;
    ap_uint<8> error_counter;
    
    // Reset
    spi_adc_interface_complete(
        start, reset,
        spi_clk, spi_mosi, spi_cs, spi_miso,
        audio_out,
        ready, data_valid, sample_counter, error_counter
    );
    
    std::cout << "After reset:\n";
    std::cout << "  Ready: " << (int)ready << "\n";
    std::cout << "  Sample counter: " << sample_counter << "\n";
    
    // Start capture
    reset = 0;
    start = 1;
    
    // Run for a few cycles
    for (int i = 0; i < 1000; i++) {
        // Simulate MISO data (random ADC values)
        spi_miso = rand() & 0x01;
        
        spi_adc_interface_complete(
            start, reset,
            spi_clk, spi_mosi, spi_cs, spi_miso,
            audio_out,
            ready, data_valid, sample_counter, error_counter
        );
        
        // Read audio data if available
        if (data_valid && !audio_out.empty()) {
            audio_sample_t sample = audio_out.read();
            std::cout << "Sample " << i << ": " << sample << "\n";
        }
        
        // Stop after first capture
        if (i > 100) {
            start = 0;
        }
    }
    
    std::cout << "\nTest complete:\n";
    std::cout << "  Samples captured: " << sample_counter << "\n";
    std::cout << "  Errors: " << (int)error_counter << "\n";
    std::cout << "  Ready: " << (int)ready << "\n";
}

int main() {
    test_spi_adc_interface();
    return 0;
}
#endif

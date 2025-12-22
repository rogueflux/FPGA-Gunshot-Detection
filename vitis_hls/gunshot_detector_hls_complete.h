#ifndef GUNSHOT_DETECTOR_HLS_COMPLETE_H
#define GUNSHOT_DETECTOR_HLS_COMPLETE_H

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

// Fixed-point precision configuration
#define AP_INT_WIDTH 8
#define AP_FIXED_WIDTH 16
#define AP_FIXED_INT_WIDTH 6

// Define fixed-point types for FPGA efficiency
typedef ap_int<AP_INT_WIDTH> weight_t;           // 8-bit weights
typedef ap_int<AP_INT_WIDTH * 2> accum_t;        // 16-bit accumulators
typedef ap_fixed<AP_FIXED_WIDTH, AP_FIXED_INT_WIDTH> fixed_t; // 16-bit fixed-point (6 integer, 10 fractional)

// Model configuration
#define INPUT_SIZE 1024
#define NUM_CHANNELS 1
#define CONV1_OUT_CHANNELS 32
#define CONV1_KERNEL_SIZE 3

#define DEPTHWISE1_OUT_CHANNELS 32
#define DEPTHWISE1_KERNEL_SIZE 3

#define POINTWISE1_OUT_CHANNELS 64
#define POINTWISE1_KERNEL_SIZE 1

#define DEPTHWISE2_OUT_CHANNELS 64
#define DEPTHWISE2_KERNEL_SIZE 3

#define POINTWISE2_OUT_CHANNELS 128
#define POINTWISE2_KERNEL_SIZE 1

#define GLOBAL_POOL_SIZE 128
#define FC1_SIZE 64
#define FC2_SIZE 32
#define OUTPUT_SIZE 2

// Buffer sizes
#define BUFFER_SIZE_1024 1024
#define BUFFER_SIZE_512 512
#define BUFFER_SIZE_256 256
#define BUFFER_SIZE_128 128
#define BUFFER_SIZE_64 64
#define BUFFER_SIZE_32 32

// Structure for detection results
struct DetectionResult {
    ap_uint<1> is_gunshot;
    ap_uint<8> confidence;
    ap_uint<16> timestamp;
};

// Main function prototype
void gunshot_detector_hls_complete(
    // Input interface
    hls::stream<fixed_t>& audio_in,
    
    // Output interface
    hls::stream<DetectionResult>& detection_out,
    
    // Configuration interface
    ap_uint<1> start,
    ap_uint<1> reset,
    
    // Status
    ap_uint<1>& ready,
    ap_uint<32>& sample_count
);

// Layer function prototypes
template<int IN_SIZE, int OUT_SIZE, int KERNEL_SIZE>
void conv1d_layer(
    fixed_t input[IN_SIZE],
    fixed_t output[OUT_SIZE],
    weight_t weights[OUT_SIZE][KERNEL_SIZE],
    weight_t bias[OUT_SIZE],
    bool use_relu = true
);

void depthwise_conv1d(
    fixed_t input[BUFFER_SIZE_1024][CONV1_OUT_CHANNELS],
    fixed_t output[BUFFER_SIZE_512][DEPTHWISE1_OUT_CHANNELS],
    weight_t weights[DEPTHWISE1_OUT_CHANNELS][DEPTHWISE1_KERNEL_SIZE],
    weight_t bias[DEPTHWISE1_OUT_CHANNELS]
);

void pointwise_conv1d(
    fixed_t input[BUFFER_SIZE_512][DEPTHWISE1_OUT_CHANNELS],
    fixed_t output[BUFFER_SIZE_512][POINTWISE1_OUT_CHANNELS],
    weight_t weights[POINTWISE1_OUT_CHANNELS][DEPTHWISE1_OUT_CHANNELS],
    weight_t bias[POINTWISE1_OUT_CHANNELS]
);

void se_attention_block(
    fixed_t input[BUFFER_SIZE_128],
    fixed_t output[BUFFER_SIZE_128],
    weight_t fc1_weights[FC1_SIZE][BUFFER_SIZE_128],
    weight_t fc1_bias[FC1_SIZE],
    weight_t fc2_weights[BUFFER_SIZE_128][FC1_SIZE],
    weight_t fc2_bias[BUFFER_SIZE_128]
);

void fully_connected_layer(
    fixed_t input[BUFFER_SIZE_128],
    fixed_t output[FC1_SIZE],
    weight_t weights[FC1_SIZE][BUFFER_SIZE_128],
    weight_t bias[FC1_SIZE],
    bool use_relu = true
);

void output_layer(
    fixed_t input[FC2_SIZE],
    fixed_t output[OUTPUT_SIZE],
    weight_t weights[OUTPUT_SIZE][FC2_SIZE],
    weight_t bias[OUTPUT_SIZE]
);

// Activation functions
fixed_t relu(fixed_t x);
fixed_t fast_sigmoid(fixed_t x);
fixed_t fast_tanh(fixed_t x);

// Utility functions
void max_pool_1d(
    fixed_t input[BUFFER_SIZE_1024][CONV1_OUT_CHANNELS],
    fixed_t output[BUFFER_SIZE_512][CONV1_OUT_CHANNELS],
    int pool_size = 2
);

void global_avg_pool(
    fixed_t input[BUFFER_SIZE_256][POINTWISE2_OUT_CHANNELS],
    fixed_t output[POINTWISE2_OUT_CHANNELS]
);

void normalize_input(
    fixed_t input[INPUT_SIZE],
    fixed_t output[INPUT_SIZE]
);

// Testbench helper functions (for simulation only)
#ifdef __SYNTHESIS__
#else
void generate_test_input(fixed_t input[INPUT_SIZE]);
void verify_results(fixed_t output[OUTPUT_SIZE]);
#endif

#endif // GUNSHOT_DETECTOR_HLS_COMPLETE_H

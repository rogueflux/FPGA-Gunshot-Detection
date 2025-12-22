#ifndef GUNSHOT_WEIGHTS_H
#define GUNSHOT_WEIGHTS_H

#include "ap_int.h"

// Model weights for FPGA implementation
// Quantized to 8-bit integers for efficient FPGA usage

// Layer 1: Initial Convolution (1->32 channels, kernel=3)
const ap_int<8> conv1_weights[32][3] = {
    #include "weights/conv1_weights.dat"
};

const ap_int<8> conv1_bias[32] = {
    #include "weights/conv1_bias.dat"
};

// Layer 2: Depthwise Conv 1 (32 channels, kernel=3)
const ap_int<8> dwconv1_weights[32][3] = {
    #include "weights/dwconv1_weights.dat"
};

const ap_int<8> dwconv1_bias[32] = {
    #include "weights/dwconv1_bias.dat"
};

// Layer 3: Pointwise Conv 1 (32->64 channels, kernel=1)
const ap_int<8> pwconv1_weights[64][32] = {
    #include "weights/pwconv1_weights.dat"
};

const ap_int<8> pwconv1_bias[64] = {
    #include "weights/pwconv1_bias.dat"
};

// Layer 4: Depthwise Conv 2 (64 channels, kernel=3)
const ap_int<8> dwconv2_weights[64][3] = {
    #include "weights/dwconv2_weights.dat"
};

const ap_int<8> dwconv2_bias[64] = {
    #include "weights/dwconv2_bias.dat"
};

// Layer 5: Pointwise Conv 2 (64->128 channels, kernel=1)
const ap_int<8> pwconv2_weights[128][64] = {
    #include "weights/pwconv2_weights.dat"
};

const ap_int<8> pwconv2_bias[128] = {
    #include "weights/pwconv2_bias.dat"
};

// SE Attention Block weights
// FC1: 128 -> 8 (reduced for efficiency)
const ap_int<8> se_fc1_weights[8][128] = {
    #include "weights/se_fc1_weights.dat"
};

const ap_int<8> se_fc1_bias[8] = {
    #include "weights/se_fc1_bias.dat"
};

// FC2: 8 -> 128
const ap_int<8> se_fc2_weights[128][8] = {
    #include "weights/se_fc2_weights.dat"
};

const ap_int<8> se_fc2_bias[128] = {
    #include "weights/se_fc2_bias.dat"
};

// Fully Connected Layer 1: 128 -> 64
const ap_int<8> fc1_weights[64][128] = {
    #include "weights/fc1_weights.dat"
};

const ap_int<8> fc1_bias[64] = {
    #include "weights/fc1_bias.dat"
};

// Fully Connected Layer 2: 64 -> 32
const ap_int<8> fc2_weights[32][64] = {
    #include "weights/fc2_weights.dat"
};

const ap_int<8> fc2_bias[32] = {
    #include "weights/fc2_bias.dat"
};

// Output Layer: 32 -> 2
const ap_int<8> output_weights[2][32] = {
    #include "weights/output_weights.dat"
};

const ap_int<8> output_bias[2] = {
    #include "weights/output_bias.dat"
};

// Precomputed test weights for simulation
// These are random weights for testing - real weights should be trained
const ap_int<8> test_conv1_weights[32][3] = {
    {12, -5, 8}, {-3, 15, -7}, {9, -2, 11}, {-6, 14, -4},
    {13, -8, 6}, {-1, 10, -9}, {7, -3, 12}, {-5, 11, -6},
    {10, -4, 9}, {-2, 13, -8}, {8, -1, 14}, {-7, 12, -5},
    {11, -6, 7}, {-4, 9, -10}, {6, -2, 13}, {-8, 15, -3},
    {14, -7, 5}, {-9, 8, -11}, {5, -1, 15}, {-10, 13, -2},
    {15, -9, 4}, {-11, 7, -12}, {4, -3, 10}, {-12, 14, -1},
    {9, -10, 3}, {-13, 6, -14}, {3, -4, 8}, {-14, 11, -13},
    {8, -11, 2}, {-15, 5, -15}, {2, -5, 7}, {-16, 10, -16}
};

const ap_int<8> test_conv1_bias[32] = {
    1, -2, 3, -1, 2, -3, 1, -2, 
    3, -1, 2, -3, 1, -2, 3, -1,
    2, -3, 1, -2, 3, -1, 2, -3,
    1, -2, 3, -1, 2, -3, 1, -2
};

#endif // GUNSHOT_WEIGHTS_H

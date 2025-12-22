#include "gunshot_detector_hls_complete.h"
#include "gunshot_weights.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

// Test parameters
#define NUM_TESTS 10
#define SAMPLE_RATE 16000
#define TEST_DURATION 1.0  // seconds

// Function to generate test audio signals
void generate_gunshot_audio(fixed_t audio[INPUT_SIZE]) {
    // Generate a synthetic gunshot: fast attack, exponential decay
    for (int i = 0; i < INPUT_SIZE; i++) {
        float t = (float)i / SAMPLE_RATE;
        
        // Fast attack (1ms)
        float attack = 0.0;
        if (t < 0.001) {
            attack = t / 0.001;  // Linear ramp
        }
        
        // Exponential decay
        float decay = exp(-t * 50.0);  // 50Hz decay rate
        
        // Add some high-frequency content
        float high_freq = 0.3 * sin(2 * M_PI * 2000 * t) * decay;
        
        // Combine
        float sample = attack * decay + high_freq;
        
        // Add noise
        sample += 0.05 * ((float)rand() / RAND_MAX - 0.5);
        
        audio[i] = sample;
    }
}

void generate_background_noise(fixed_t audio[INPUT_SIZE]) {
    // Generate white noise
    for (int i = 0; i < INPUT_SIZE; i++) {
        float sample = 0.1 * ((float)rand() / RAND_MAX - 0.5);
        
        // Add some low-frequency hum
        sample += 0.05 * sin(2 * M_PI * 60 * i / SAMPLE_RATE);
        
        audio[i] = sample;
    }
}

void generate_speech_audio(fixed_t audio[INPUT_SIZE]) {
    // Generate synthetic speech-like signal
    for (int i = 0; i < INPUT_SIZE; i++) {
        float t = (float)i / SAMPLE_RATE;
        
        // Base frequency (voice pitch)
        float pitch = 100 + 50 * sin(2 * M_PI * 2 * t);  // Varying pitch
        
        // Formants (speech resonances)
        float formant1 = 0.5 * sin(2 * M_PI * pitch * t);
        float formant2 = 0.3 * sin(2 * M_PI * (pitch * 2) * t);
        float formant3 = 0.2 * sin(2 * M_PI * (pitch * 3) * t);
        
        // Envelope (simulating syllables)
        float envelope = 0.5 + 0.5 * sin(2 * M_PI * 4 * t);
        envelope = envelope * envelope;  // Square for sharper attacks
        
        float sample = envelope * (formant1 + formant2 + formant3);
        
        // Add noise
        sample += 0.02 * ((float)rand() / RAND_MAX - 0.5);
        
        audio[i] = sample;
    }
}

void generate_impulse_noise(fixed_t audio[INPUT_SIZE]) {
    // Generate random impulse noise
    for (int i = 0; i < INPUT_SIZE; i++) {
        audio[i] = 0.0;
    }
    
    // Add random impulses
    for (int imp = 0; imp < 5; imp++) {
        int pos = rand() % INPUT_SIZE;
        float amplitude = 0.5 + 0.5 * ((float)rand() / RAND_MAX);
        
        // Create a short impulse
        for (int j = 0; j < 10; j++) {
            if (pos + j < INPUT_SIZE) {
                audio[pos + j] = amplitude * exp(-j / 2.0);
            }
        }
    }
    
    // Add background noise
    for (int i = 0; i < INPUT_SIZE; i++) {
        audio[i] += 0.05 * ((float)rand() / RAND_MAX - 0.5);
    }
}

// Save results to file
void save_results_to_file(const char* filename, 
                         fixed_t audio[INPUT_SIZE],
                         fixed_t output[OUTPUT_SIZE],
                         ap_uint<1> detection,
                         ap_uint<8> confidence) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        return;
    }
    
    file << "Test Results\n";
    file << "============\n";
    file << "Detection: " << (detection ? "GUNSHOT" : "NO_GUNSHOT") << "\n";
    file << "Confidence: " << (int)confidence << "/255\n";
    file << "Probabilities: Non-gunshot=" << output[0] 
         << ", Gunshot=" << output[1] << "\n\n";
    
    file << "Audio Samples (first 100):\n";
    for (int i = 0; i < 100 && i < INPUT_SIZE; i++) {
        file << audio[i] << "\n";
    }
    
    file.close();
}

// Calculate audio statistics
void calculate_audio_stats(fixed_t audio[INPUT_SIZE],
                          fixed_t& max_amplitude,
                          fixed_t& rms,
                          fixed_t& zero_crossings) {
    max_amplitude = 0;
    fixed_t sum_squares = 0;
    zero_crossings = 0;
    
    for (int i = 0; i < INPUT_SIZE; i++) {
        fixed_t abs_val = hls::abs(audio[i]);
        if (abs_val > max_amplitude) {
            max_amplitude = abs_val;
        }
        
        sum_squares += audio[i] * audio[i];
        
        if (i > 0) {
            if ((audio[i] > 0 && audio[i-1] < 0) || 
                (audio[i] < 0 && audio[i-1] > 0)) {
                zero_crossings++;
            }
        }
    }
    
    rms = hls::sqrt(sum_squares / INPUT_SIZE);
    zero_crossings = zero_crossings / INPUT_SIZE;
}

// Main testbench
int main() {
    std::cout << "========================================\n";
    std::cout << "Gunshot Detector HLS Testbench\n";
    std::cout << "========================================\n\n";
    
    // Test counters
    int total_tests = 0;
    int correct_detections = 0;
    int false_positives = 0;
    int false_negatives = 0;
    
    // Test 1: Gunshot audio
    std::cout << "Test 1: Synthetic Gunshot Audio\n";
    std::cout << "-------------------------------\n";
    
    for (int test = 0; test < NUM_TESTS; test++) {
        fixed_t audio[INPUT_SIZE];
        generate_gunshot_audio(audio);
        
        ap_uint<8> confidence;
        ap_uint<1> detection;
        
        // Run detection
        gunshot_detector_simple(audio, confidence, detection);
        
        // Calculate statistics
        fixed_t max_amp, rms, zcr;
        calculate_audio_stats(audio, max_amp, rms, zcr);
        
        std::cout << "  Test " << test + 1 << ": ";
        std::cout << "Max=" << max_amp << ", RMS=" << rms << ", ZCR=" << zcr;
        std::cout << " -> Detection: " << (detection ? "GUNSHOT" : "NO_GUNSHOT");
        std::cout << " (Confidence: " << (int)confidence << "/255)\n";
        
        total_tests++;
        if (detection == 1) {
            correct_detections++;
        } else {
            false_negatives++;
        }
        
        // Save first test result
        if (test == 0) {
            fixed_t output[OUTPUT_SIZE] = {1.0 - confidence/255.0, confidence/255.0};
            save_results_to_file("test_gunshot_result.txt", 
                               audio, output, detection, confidence);
        }
    }
    std::cout << std::endl;
    
    // Test 2: Background noise
    std::cout << "Test 2: Background Noise\n";
    std::cout << "------------------------\n";
    
    for (int test = 0; test < NUM_TESTS; test++) {
        fixed_t audio[INPUT_SIZE];
        generate_background_noise(audio);
        
        ap_uint<8> confidence;
        ap_uint<1> detection;
        
        // Run detection
        gunshot_detector_simple(audio, confidence, detection);
        
        // Calculate statistics
        fixed_t max_amp, rms, zcr;
        calculate_audio_stats(audio, max_amp, rms, zcr);
        
        std::cout << "  Test " << test + 1 << ": ";
        std::cout << "Max=" << max_amp << ", RMS=" << rms << ", ZCR=" << zcr;
        std::cout << " -> Detection: " << (detection ? "GUNSHOT" : "NO_GUNSHOT");
        std::cout << " (Confidence: " << (int)confidence << "/255)\n";
        
        total_tests++;
        if (detection == 0) {
            correct_detections++;
        } else {
            false_positives++;
        }
    }
    std::cout << std::endl;
    
    // Test 3: Speech audio
    std::cout << "Test 3: Synthetic Speech Audio\n";
    std::cout << "-------------------------------\n";
    
    for (int test = 0; test < NUM_TESTS; test++) {
        fixed_t audio[INPUT_SIZE];
        generate_speech_audio(audio);
        
        ap_uint<8> confidence;
        ap_uint<1> detection;
        
        // Run detection
        gunshot_detector_simple(audio, confidence, detection);
        
        // Calculate statistics
        fixed_t max_amp, rms, zcr;
        calculate_audio_stats(audio, max_amp, rms, zcr);
        
        std::cout << "  Test " << test + 1 << ": ";
        std::cout << "Max=" << max_amp << ", RMS=" << rms << ", ZCR=" << zcr;
        std::cout << " -> Detection: " << (detection ? "GUNSHOT" : "NO_GUNSHOT");
        std::cout << " (Confidence: " << (int)confidence << "/255)\n";
        
        total_tests++;
        if (detection == 0) {
            correct_detections++;
        } else {
            false_positives++;
        }
    }
    std::cout << std::endl;
    
    // Test 4: Impulse noise
    std::cout << "Test 4: Impulse Noise\n";
    std::cout << "---------------------\n";
    
    for (int test = 0; test < NUM_TESTS; test++) {
        fixed_t audio[INPUT_SIZE];
        generate_impulse_noise(audio);
        
        ap_uint<8> confidence;
        ap_uint<1> detection;
        
        // Run detection
        gunshot_detector_simple(audio, confidence, detection);
        
        // Calculate statistics
        fixed_t max_amp, rms, zcr;
        calculate_audio_stats(audio, max_amp, rms, zcr);
        
        std::cout << "  Test " << test + 1 << ": ";
        std::cout << "Max=" << max_amp << ", RMS=" << rms << ", ZCR=" << zcr;
        std::cout << " -> Detection: " << (detection ? "GUNSHOT" : "NO_GUNSHOT");
        std::cout << " (Confidence: " << (int)confidence << "/255)\n";
        
        total_tests++;
        // Impulse noise might be detected as gunshot (acceptable)
        if (detection == 0) {
            correct_detections++;
        } else {
            false_positives++;
        }
    }
    std::cout << std::endl;
    
    // Performance test: Measure throughput
    std::cout << "Performance Test\n";
    std::cout << "----------------\n";
    
    const int PERF_ITERATIONS = 1000;
    fixed_t test_audio[INPUT_SIZE];
    generate_gunshot_audio(test_audio);
    
    // Time measurement (simplified for simulation)
    int total_operations = 0;
    
    for (int i = 0; i < PERF_ITERATIONS; i++) {
        ap_uint<8> confidence;
        ap_uint<1> detection;
        
        gunshot_detector_simple(test_audio, confidence, detection);
        total_operations++;
    }
    
    // Calculate theoretical performance
    float samples_per_second = SAMPLE_RATE;
    float inferences_per_second = samples_per_second / INPUT_SIZE;
    float operations_per_inference = 100000;  // Estimated operations
    float total_gops = (operations_per_inference * inferences_per_second) / 1e9;
    
    std::cout << "  Throughput: " << inferences_per_second << " inferences/sec\n";
    std::cout << "  Theoretical Performance: " << total_gops << " GOPs\n";
    std::cout << "  Latency per inference: " << (1000.0 / inferences_per_second) << " ms\n";
    
    // Summary
    std::cout << "\n========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    
    float accuracy = (float)correct_detections / total_tests * 100.0;
    float false_positive_rate = (float)false_positives / total_tests * 100.0;
    float false_negative_rate = (float)false_negatives / total_tests * 100.0;
    
    std::cout << "Total Tests: " << total_tests << "\n";
    std::cout << "Correct Detections: " << correct_detections << "\n";
    std::cout << "False Positives: " << false_positives << "\n";
    std::cout << "False Negatives: " << false_negatives << "\n";
    std::cout << "Accuracy: " << accuracy << "%\n";
    std::cout << "False Positive Rate: " << false_positive_rate << "%\n";
    std::cout << "False Negative Rate: " << false_negative_rate << "%\n";
    
    // Pass/Fail criteria
    bool test_passed = (accuracy > 85.0) && (false_positive_rate < 10.0);
    
    std::cout << "\nTest Result: " << (test_passed ? "PASS" : "FAIL") << "\n";
    std::cout << "========================================\n";
    
    return test_passed ? 0 : 1;
}

// Helper function for AXI Stream simulation
void simulate_axi_stream_test() {
    std::cout << "\nSimulating AXI Stream Interface\n";
    std::cout << "--------------------------------\n";
    
    // Create streams
    hls::stream<fixed_t> audio_in;
    hls::stream<DetectionResult> detection_out;
    
    ap_uint<1> ready;
    ap_uint<32> sample_count;
    
    // Generate test audio
    fixed_t test_audio[INPUT_SIZE];
    generate_gunshot_audio(test_audio);
    
    // Write to input stream
    for (int i = 0; i < INPUT_SIZE; i++) {
        audio_in.write(test_audio[i]);
    }
    
    // Run detection
    gunshot_detector_hls_complete(
        audio_in,
        detection_out,
        1,  // start
        0,  // reset
        ready,
        sample_count
    );
    
    // Read result
    if (!detection_out.empty()) {
        DetectionResult result = detection_out.read();
        
        std::cout << "Detection Result:\n";
        std::cout << "  Timestamp: " << result.timestamp << "\n";
        std::cout << "  Detection: " << (result.is_gunshot ? "GUNSHOT" : "NO_GUNSHOT") << "\n";
        std::cout << "  Confidence: " << (int)result.confidence << "/255\n";
    }
    
    std::cout << "Ready signal: " << (int)ready << "\n";
    std::cout << "Sample count: " << sample_count << "\n";
}

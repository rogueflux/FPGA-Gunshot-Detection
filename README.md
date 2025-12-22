# Gunshot Detection and Localization System

## 📋 Overview
A lightweight, real-time gunshot detection and localization system designed for edge deployment on FPGA platforms. The system uses a six-microphone hexagonal array and an attention-augmented MobileNet1D neural network to detect gunshots with high recall, followed by MUSIC-based localization for directional estimation.

**Key Capabilities:**
- Real-time gunshot detection with 98.22% recall
- 2D source localization using MUSIC algorithm
- Ultra-low latency (6.03 ms end-to-end)
- Edge-optimized for FPGA deployment

## Project Structure

```
gunshot_detection_fpga/
├── notebooks/                    # Jupyter notebooks for R&D
│   ├── Bi_Audioprocessor_and_extractor.ipynb
│   ├── Gunshot_NewAttentionModel.ipynb
│   └── Source_Separation_MUSIC_algorithm.ipynb
├── python/                       # Production Python code
├── hardware/                     # FPGA implementation
├── firmware/                     # Embedded software
├── models/                       # Trained models and conversion
├── scripts/                      # Build and deployment scripts
├── docs/                         # Documentation
└── tests/                        # Test suites
```

## Quick Start

### Prerequisites
- Python 3.8+
- Xilinx Vivado 2022.1
- Xilinx Vitis HLS 2022.1
- PyTorch 1.12+

### Installation
```bash
# Clone repository
git clone <repository-url>
cd gunshot_detection_fpga

# Install Python dependencies
pip install -r requirements.txt

# Setup environment variables
source scripts/setup_environment.sh
```

### Training the Model
```bash
# Train the attention-augmented MobileNet1D
python python/gunshot_model/train.py --config configs/train_config.yaml

# Convert to HLS-compatible format
python models/convert_pytorch_to_hls.py --model models/gunshot_model_weights.pth
```

### FPGA Deployment
```bash
# Complete build (FPGA bitstream + firmware)
./scripts/complete_build.sh

# Program FPGA
./scripts/program_fpga.sh --device /dev/ttyUSB0

# Run system test
./scripts/test_system.py
```

## System Performance

### Detection Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 89.89% |
| **Recall (Gunshot)** | **98.22%** |
| Precision | 72.22% |
| F1-Score | 0.8324 |
| ROC AUC | 0.9826 |

### Hardware Performance
| Platform | Latency | Power | Throughput |
|----------|---------|-------|------------|
| **FPGA (Spartan-7)** | **6.03 ms** | **305 mW** | **165 fps** |
| Raspberry Pi 4 | 45 ms | 4.5 W | 22 fps |
| Cloud (AWS) | 120 ms | N/A | 80 fps |

## Hardware Implementation

### Target Platform
- **FPGA**: Xilinx Spartan-7 XC7S50T
- **Microphones**: 6× INMP441 MEMS array (hexagonal)
- **Interfaces**: I2S audio, UART, GPIO alerts

### Resource Utilization
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs | 28,450 | 52,160 | 54.5% |
| BRAM | 65 | 145 | 44.8% |
| DSP Slices | 120 | 200 | 60.0% |

## Model Architecture

### Attention-Augmented MobileNet1D
- **Parameters**: 18,000 (~80 KB)
- **Input**: 2052-dimensional spectral features
- **Architecture**:
  1. Initial 1D convolution (1→48 channels)
  2. 3× depthwise separable blocks
  3. Squeeze-and-Excitation attention
  4. 3-layer classifier (128→64→32→2)

### MUSIC Localization
- **Algorithm**: Multiple Signal Classification
- **Output**: 2D source position (x, y)
- **Confidence**: Peak sharpness-based certainty

## Development Workflow

### 1. Research & Experimentation
```python
# Explore in Jupyter notebooks
jupyter notebook notebooks/Gunshot_NewAttentionModel.ipynb
```

### 2. Model Development
```bash
# Train and validate model
python python/gunshot_model/train.py

# Test preprocessing pipeline
python python/preprocessing/audio_processor.py --audio sample.wav
```

### 3. FPGA Implementation
```bash
# Convert model to HLS
python models/convert_pytorch_to_hls.py

# Build FPGA project
cd hardware/vivado
vivado -source vivado_project.tcl

# Synthesize HLS IP
cd ../vitis_hls
vitis_hls -f hls_script.tcl
```

### 4. System Integration
```bash
# Build complete system
./scripts/complete_build.sh

# Deploy and test
./scripts/test_system.py --mode hardware
```

## Key Files

### Notebooks
- `Bi_Audioprocessor_and_extractor.ipynb`: Audio preprocessing and feature extraction
- `Gunshot_NewAttentionModel.ipynb`: Attention model development and training
- `Source_Separation_MUSIC_algorithm.ipynb`: Localization algorithm implementation

### Python Modules
- `python/gunshot_model/attention_model.py`: Neural network architecture
- `python/preprocessing/feature_extractor.py`: STFT-based feature extraction
- `python/preprocessing/music_algorithm.py`: MUSIC localization implementation

### FPGA Sources
- `hardware/vitis_hls/gunshot_detector_hls_complete.cpp`: HLS-optimized model
- `hardware/vivado/spartan_edge_accel_gunshot.xdc`: Timing constraints
- `firmware/src/music_localization.c`: Embedded MUSIC implementation

## Testing

### Unit Tests
```bash
# Run Python unit tests
python -m pytest tests/unit_tests/

# Run hardware simulation tests
cd tests/hardware_tests/
./run_simulation.sh
```

### System Validation
```bash
# Full system test with recorded audio
./scripts/test_system.py --audio data/audio_samples/test_gunshot.wav

# Hardware-in-the-loop test
./scripts/test_system.py --mode hardware --duration 60
```

## Applications

### Public Safety
- Real-time urban gunshot surveillance
- Rapid police response coordination
- Forensic evidence collection

### Smart Cities
- Distributed acoustic sensor networks
- Edge-deployed IoT monitoring
- Critical infrastructure protection

### Conservation
- Anti-poaching in protected areas
- Remote wildlife monitoring
- Low-power field deployment

## Performance Optimization

### Model Compression
```bash
# 8-bit quantization (4× memory reduction)
python models/quantize_model.py --model gunshot_model_weights.pth --bits 8

# Pruning (30% sparsity)
python models/prune_model.py --model gunshot_model_weights.pth --sparsity 0.3
```

### FPGA Optimization
```cpp
// HLS pragmas for optimization
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=input cyclic factor=16
#pragma HLS UNROLL factor=8
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Submit a pull request**

### Development Guidelines
- Maintain 98%+ recall for gunshot class
- Keep model under 25,000 parameters
- Ensure <10 ms latency on target hardware
- Document all public APIs

## Documentation

- [System Architecture](docs/architecture_diagrams/)
- [API Reference](docs/api_documentation.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

Proprietary - Contact authors for licensing information.

## Contact

**Research Team**  
Email: research@example.com  
Website: https://example.com/research

**Technical Support**  
Email: support@example.com  
Issue Tracker: GitHub Issues

## Acknowledgments

- Based on research from Kabir et al. (2022) and Morehead et al. (2019)
- Implemented on Digilent Spartan Edge Accelerator platform
- Tested with custom-collected urban audio dataset

---

**Safety Note:** This system is designed for public safety applications. Always verify detections through secondary means before initiating emergency responses. False positive rate: 12.98% - implement appropriate alert verification protocols.

**Last Updated:** December 2025  
**Version:** 1.0.0  
**Compatibility:** Python 3.8+, Vivado 2022.1+, Spartan-7 FPGA

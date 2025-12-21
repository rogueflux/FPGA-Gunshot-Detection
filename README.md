# Gunshot Detection and Localization System

## Overview
A lightweight, real-time gunshot detection and localization system designed for edge deployment. The system uses a six-microphone hexagonal array and an attention-augmented MobileNet1D neural network to detect gunshots with high recall, followed by MUSIC-based localization for directional estimation.

## System Architecture

### Processing Pipeline
1. **Audio Acquisition** – Six-microphone array, 16 kHz, 2-second windows
2. **Preprocessing** – Bandpass filter (500–7000 Hz), impulse detection, spectral validation
3. **Feature Extraction** – STFT-based spectral features (2052-dimensional vector)
4. **Neural Network Classification** – Attention-enhanced MobileNet1D (18,000 parameters)
5. **Localization** – MUSIC algorithm for 2D direction-of-arrival estimation

## Key Features
- **High Recall**: 98.22% gunshot detection rate
- **Low Latency**: 6.03 ms end-to-end on FPGA
- **Lightweight Model**: 18,000 parameters (~80 KB)
- **Integrated Localization**: MUSIC-based 2D source estimation
- **Edge-Optimized**: Deployable on FPGA, Raspberry Pi, or ARM processors
- **Low Power**: 305 mW average power consumption

## Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 89.89% |
| Recall (Gunshot) | 98.22% |
| Precision | 72.22% |
| F1-Score | 0.8324 |
| ROC AUC | 0.9826 |
| PR AUC | 0.9468 |

## Comparison with Baselines
| Model | Params | Accuracy | Recall | F1-Score |
|-------|--------|----------|--------|----------|
| **Ours (Attention-MobileNet1D)** | **18,000** | 89.89% | **98.22%** | 0.8324 |
| Morehead et al. [2] | 43,600 | **99.4%** | 96.6% | **0.973** |
| Kabir et al. [6] | – | 97.3% | 97.8% | – |

## Hardware Implementation
### Target Platform
- **FPGA**: Xilinx Spartan-7 XC7S50T (Digilent Spartan Edge Accelerator)
- **Microphones**: 6× INMP441 MEMS microphones in hexagonal array
- **Interfaces**: I2S audio, UART, LED/buzzer alerts

### Resource Utilization (Spartan-7)
| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs | 28,450 | 52,160 | 54.5% |
| FFs | 15,230 | 104,320 | 14.6% |
| BRAM | 65 | 145 | 44.8% |
| DSP Slices | 120 | 200 | 60.0% |

### Real-Time Performance
- **Total Latency**: 6.03 ms
- **Throughput**: 165 frames/second
- **Power**: 305 mW average

## Deployment
### Supported Platforms
1. **FPGA**: Xilinx Zynq, Spartan-7, Intel Cyclone V
2. **Embedded**: Raspberry Pi 4, ARM Cortex-A series
3. **Cloud**: AWS, Azure (with higher latency)

### Deployment Steps
1. Train model using PyTorch
2. Export to ONNX format
3. Convert to FPGA-optimized format (Vitis AI)
4. Synthesize hardware accelerator
5. Integrate microphone array and MUSIC module
6. Deploy and calibrate detection threshold

## Applications
- Public safety and law enforcement
- Smart city IoT edge networks
- Wildlife conservation and anti-poaching
- Critical infrastructure protection
- Forensic audio analysis

## Repository Structure
```
├── models/           # Pretrained models (PyTorch, ONNX)
├── hardware/         # FPGA sources (Vivado, Vitis HLS)
├── firmware/         # Embedded software (C/C++)
├── datasets/         # Sample audio data
├── scripts/          # Training and conversion scripts
└── docs/             # Documentation and datasheets
```

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run inference: `python detect.py --audio sample.wav`
4. For FPGA deployment: follow `hardware/README.md`

## License
Proprietary – Contact authors for licensing information.

## Citation
If using this work, please reference the associated patent and technical report.

## Contact
For inquiries, deployment support, or collaboration, please contact the research team.

---
*System designed for low-power, high-recall gunshot detection in edge environments.*

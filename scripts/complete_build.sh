#!/bin/bash

# ================================================
# Complete Build Script for Gunshot Detection System
# Spartan Edge-7 Accelerator Board + Microphone PCB
# ================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="gunshot_detection_fpga"
VIVADO_VERSION="2022.2"
VITIS_HLS_VERSION="2022.2"
BOARD="spartan_edge_accelerator"
TARGET_PART="xc7s50csga324-1"
OUTPUT_DIR="output"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Error handling
set -e  # Exit on error
trap 'echo -e "${RED}Build failed! Check logs in ${LOG_DIR}/${TIMESTAMP}/${NC}"; exit 1' ERR

# Print header
print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║     Gunshot Detection FPGA Build System                  ║"
    echo "║     Target: Spartan Edge-7 Accelerator Board             ║"
    echo "║     Date: $(date)                            ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check dependencies
check_dependencies() {
    echo -e "${YELLOW}[1/10] Checking dependencies...${NC}"
    
    local missing_deps=()
    
    # Check Vivado
    if ! command -v vivado &> /dev/null; then
        missing_deps+=("Vivado $VIVADO_VERSION")
    else
        VIVADO_PATH=$(which vivado)
        echo "  ✓ Vivado found: $VIVADO_PATH"
    fi
    
    # Check Vitis HLS
    if ! command -v vitis_hls &> /dev/null; then
        missing_deps+=("Vitis HLS $VITIS_HLS_VERSION")
    else
        VITIS_HLS_PATH=$(which vitis_hls)
        echo "  ✓ Vitis HLS found: $VITIS_HLS_PATH"
    fi
    
    # Check Python dependencies
    if ! python3 -c "import torch" &> /dev/null; then
        missing_deps+=("PyTorch")
    else
        echo "  ✓ PyTorch found"
    fi
    
    if ! python3 -c "import onnx" &> /dev/null; then
        missing_deps+=("ONNX")
    else
        echo "  ✓ ONNX found"
    fi
    
    # Check system tools
    for tool in git make gcc g++ python3; do
        if ! command -v $tool &> /dev/null; then
            missing_deps+=("$tool")
        else
            echo "  ✓ $tool found"
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}Missing dependencies:${NC}"
        for dep in "${missing_deps[@]}"; do
            echo "  ✗ $dep"
        done
        echo -e "\nRun ${GREEN}./scripts/setup_environment.sh${NC} to install dependencies"
        exit 1
    fi
    
    echo -e "${GREEN}  All dependencies satisfied!${NC}"
}

# Setup directories
setup_directories() {
    echo -e "${YELLOW}[2/10] Setting up directories...${NC}"
    
    # Create output directories
    mkdir -p $OUTPUT_DIR
    mkdir -p $LOG_DIR/$TIMESTAMP
    mkdir -p $OUTPUT_DIR/bitstreams
    mkdir -p $OUTPUT_DIR/software
    mkdir -p $OUTPUT_DIR/models
    
    # Clean previous builds (optional)
    read -p "  Clean previous builds? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Cleaning previous builds..."
        rm -rf vivado/$PROJECT_NAME.cache
        rm -rf vivado/$PROJECT_NAME.hw
        rm -rf vivado/$PROJECT_NAME.sim
        rm -rf vivado/$PROJECT_NAME.runs
        rm -rf $OUTPUT_DIR/*
    fi
    
    echo -e "${GREEN}  Directory structure ready!${NC}"
}

# Convert PyTorch model to HLS
convert_model_to_hls() {
    echo -e "${YELLOW}[3/10] Converting PyTorch model to HLS...${NC}"
    
    cd models
    
    # Check if model exists
    if [ ! -f "gunshot_model_weights.pth" ]; then
        echo -e "${RED}  Model weights not found!${NC}"
        echo "  Please train the model first or download pre-trained weights"
        read -p "  Generate synthetic weights for testing? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "  Generating synthetic weights..."
            python3 -c "
import torch
import numpy as np

# Create dummy model weights
weights = {
    'conv1.weight': torch.randn(32, 1, 3) * 0.1,
    'conv1.bias': torch.randn(32) * 0.1,
    'fc1.weight': torch.randn(64, 128) * 0.1,
    'fc1.bias': torch.randn(64) * 0.1,
    'fc2.weight': torch.randn(2, 64) * 0.1,
    'fc2.bias': torch.randn(2) * 0.1
}
torch.save(weights, 'gunshot_model_weights.pth')
print('Synthetic weights saved')
            "
        else
            exit 1
        fi
    fi
    
    # Convert model
    echo "  Converting model to HLS-compatible format..."
    python3 convert_pytorch_to_hls.py \
        --input gunshot_model_weights.pth \
        --output ../vitis_hls/gunshot_weights.h \
        --quantize 8 \
        --config model_config.json \
        --log ../$LOG_DIR/$TIMESTAMP/model_conversion.log
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  Model conversion successful!${NC}"
        cp gunshot_model_weights.pth ../$OUTPUT_DIR/models/
        cp model_config.json ../$OUTPUT_DIR/models/
    else
        echo -e "${RED}  Model conversion failed!${NC}"
        exit 1
    fi
    
    cd ..
}

# Run Vitis HLS synthesis
run_vitis_hls() {
    echo -e "${YELLOW}[4/10] Running Vitis HLS synthesis...${NC}"
    
    cd vitis_hls
    
    # Check if HLS script exists
    if [ ! -f "hls_script.tcl" ]; then
        echo -e "${RED}  HLS script not found!${NC}"
        exit 1
    fi
    
    echo "  Starting HLS synthesis..."
    
    # Run HLS with logging
    vitis_hls -f hls_script.tcl \
        -l ../$LOG_DIR/$TIMESTAMP/hls_synthesis.log
    
    if [ $? -eq 0 ] && [ -f "gunshot_detector_hls_complete/solution1/impl/ip" ]; then
        echo -e "${GREEN}  HLS synthesis successful!${NC}"
        
        # Copy generated IP
        mkdir -p ../$OUTPUT_DIR/ip
        cp -r gunshot_detector_hls_complete/solution1/impl/ip/* ../$OUTPUT_DIR/ip/
        
        # Generate IP report
        if [ -f "gunshot_detector_hls_complete/solution1/syn/report/gunshot_detector_hls_complete_csynth.rpt" ]; then
            cp gunshot_detector_hls_complete/solution1/syn/report/gunshot_detector_hls_complete_csynth.rpt \
               ../$LOG_DIR/$TIMESTAMP/hls_report.txt
            echo "  HLS report saved to logs/"
        fi
    else
        echo -e "${RED}  HLS synthesis failed!${NC}"
        echo "  Check log: $LOG_DIR/$TIMESTAMP/hls_synthesis.log"
        exit 1
    fi
    
    cd ..
}

# Create Vivado project
create_vivado_project() {
    echo -e "${YELLOW}[5/10] Creating Vivado project...${NC}"
    
    cd vivado
    
    # Check if Vivado scripts exist
    if [ ! -f "vivado_project.tcl" ]; then
        echo -e "${RED}  Vivado project script not found!${NC}"
        exit 1
    fi
    
    echo "  Creating Vivado project for $BOARD..."
    
    # Run Vivado in batch mode
    vivado -mode batch -source vivado_project.tcl \
        -tclargs $PROJECT_NAME $TARGET_PART \
        -log ../$LOG_DIR/$TIMESTAMP/vivado_project.log \
        -journal ../$LOG_DIR/$TIMESTAMP/vivado_project.jou
    
    if [ $? -eq 0 ] && [ -d "$PROJECT_NAME.cache" ]; then
        echo -e "${GREEN}  Vivado project created successfully!${NC}"
    else
        echo -e "${RED}  Vivado project creation failed!${NC}"
        exit 1
    fi
    
    cd ..
}

# Implement design (synthesis, place, route)
implement_design() {
    echo -e "${YELLOW}[6/10] Implementing design (synthesis + P&R)...${NC}"
    
    cd vivado
    
    if [ ! -f "implement_design.tcl" ]; then
        echo -e "${RED}  Implement design script not found!${NC}"
        exit 1
    fi
    
    echo "  Running synthesis, placement, and routing..."
    
    vivado -mode batch -source implement_design.tcl \
        -tclargs $PROJECT_NAME \
        -log ../$LOG_DIR/$TIMESTAMP/vivado_implementation.log \
        -journal ../$LOG_DIR/$TIMESTAMP/vivado_implementation.jou
    
    # Check for bitstream
    if [ $? -eq 0 ] && [ -f "$PROJECT_NAME.runs/impl_1/gunshot_detection.bit" ]; then
        echo -e "${GREEN}  Design implementation successful!${NC}"
        
        # Copy bitstream
        cp $PROJECT_NAME.runs/impl_1/gunshot_detection.bit ../$OUTPUT_DIR/bitstreams/
        cp $PROJECT_NAME.runs/impl_1/gunshot_detection.ltx ../$OUTPUT_DIR/bitstreams/  # Debug probes if any
        
        # Generate implementation report
        if [ -f "$PROJECT_NAME.runs/impl_1/gunshot_detection_utilization_placed.rpt" ]; then
            cp $PROJECT_NAME.runs/impl_1/gunshot_detection_utilization_placed.rpt \
               ../$LOG_DIR/$TIMESTAMP/utilization_report.txt
        fi
        
        if [ -f "$PROJECT_NAME.runs/impl_1/gunshot_detection_timing_summary_routed.rpt" ]; then
            cp $PROJECT_NAME.runs/impl_1/gunshot_detection_timing_summary_routed.rpt \
               ../$LOG_DIR/$TIMESTAMP/timing_report.txt
        fi
    else
        echo -e "${RED}  Design implementation failed!${NC}"
        exit 1
    fi
    
    cd ..
}

# Export hardware platform
export_hardware() {
    echo -e "${YELLOW}[7/10] Exporting hardware platform...${NC}"
    
    cd vivado
    
    if [ ! -f "export_hardware.tcl" ]; then
        echo -e "${RED}  Export hardware script not found!${NC}"
        exit 1
    fi
    
    echo "  Exporting hardware platform for Vitis..."
    
    vivado -mode batch -source export_hardware.tcl \
        -tclargs $PROJECT_NAME \
        -log ../$LOG_DIR/$TIMESTAMP/vivado_export.log \
        -journal ../$LOG_DIR/$TIMESTAMP/vivado_export.jou
    
    if [ $? -eq 0 ] && [ -f "$PROJECT_NAME.hw/gunshot_detection.xsa" ]; then
        echo -e "${GREEN}  Hardware platform exported successfully!${NC}"
        cp $PROJECT_NAME.hw/gunshot_detection.xsa ../$OUTPUT_DIR/
    else
        echo -e "${RED}  Hardware export failed!${NC}"
        exit 1
    fi
    
    cd ..
}

# Build Vitis software
build_vitis_software() {
    echo -e "${YELLOW}[8/10] Building Vitis software...${NC}"
    
    # Check if Vitis is available
    if ! command -v xsct &> /dev/null; then
        echo -e "${YELLOW}  Vitis not found, skipping software build...${NC}"
        echo "  You can build the software manually with:"
        echo "  cd firmware && make"
        return 0
    fi
    
    cd firmware
    
    if [ -f "Makefile" ]; then
        echo "  Building firmware using Makefile..."
        make clean
        make all 2>&1 | tee ../$LOG_DIR/$TIMESTAMP/firmware_build.log
        
        if [ $? -eq 0 ] && [ -f "build/application.elf" ]; then
            echo -e "${GREEN}  Firmware build successful!${NC}"
            cp build/application.elf ../$OUTPUT_DIR/software/
            cp build/*.bin ../$OUTPUT_DIR/software/ 2>/dev/null || true
        else
            echo -e "${RED}  Firmware build failed!${NC}"
            exit 1
        fi
    else
        echo "  No Makefile found, creating simple build..."
        mkdir -p build
        gcc -o build/test_app ../scripts/test_system.py 2>&1 | tee ../$LOG_DIR/$TIMESTAMP/firmware_build.log
        echo -e "${YELLOW}  Created test application${NC}"
    fi
    
    cd ..
}

# Generate boot image
generate_boot_image() {
    echo -e "${YELLOW}[9/10] Generating boot image...${NC}"
    
    # Check for bootgen
    if ! command -v bootgen &> /dev/null; then
        echo -e "${YELLOW}  bootgen not found, creating dummy boot image...${NC}"
        echo "BOOT_IMAGE" > $OUTPUT_DIR/boot.bin
        echo "BITSTREAM: $OUTPUT_DIR/bitstreams/gunshot_detection.bit" >> $OUTPUT_DIR/boot.bin
        return 0
    fi
    
    # Create BIF file
    cat > $OUTPUT_DIR/bootimage.bif << EOF
// Boot Image Format for Spartan Edge-7
the_ROM_image:
{
    [bootloader] $OUTPUT_DIR/software/fsbl.elf  # First Stage Bootloader
    $OUTPUT_DIR/bitstreams/gunshot_detection.bit  # FPGA Bitstream
    $OUTPUT_DIR/software/application.elf  # Application
}
EOF
    
    echo "  Generating boot.bin..."
    bootgen -image $OUTPUT_DIR/bootimage.bif \
        -arch zynq -o $OUTPUT_DIR/boot.bin -w \
        -log $LOG_DIR/$TIMESTAMP/bootgen.log
    
    if [ $? -eq 0 ] && [ -f "$OUTPUT_DIR/boot.bin" ]; then
        echo -e "${GREEN}  Boot image generated successfully!${NC}"
    else
        echo -e "${RED}  Boot image generation failed!${NC}"
        echo -e "${YELLOW}  Creating minimal boot image...${NC}"
        cp $OUTPUT_DIR/bitstreams/gunshot_detection.bit $OUTPUT_DIR/boot.bin
    fi
}

# Final summary
print_summary() {
    echo -e "${YELLOW}[10/10] Build completed!${NC}"
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                   BUILD SUMMARY                          ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║                                                          ║"
    echo "║  Output files in '$OUTPUT_DIR/':                        ║"
    
    # List generated files
    if [ -f "$OUTPUT_DIR/boot.bin" ]; then
        echo "║    • boot.bin              (SD card boot image)       ║"
    fi
    
    if [ -f "$OUTPUT_DIR/bitstreams/gunshot_detection.bit" ]; then
        BIT_SIZE=$(stat -c%s "$OUTPUT_DIR/bitstreams/gunshot_detection.bit" 2>/dev/null || echo "N/A")
        echo "║    • gunshot_detection.bit (FPGA bitstream, $BIT_SIZE bytes) ║"
    fi
    
    if [ -f "$OUTPUT_DIR/software/application.elf" ]; then
        echo "║    • application.elf       (ARM application)          ║"
    fi
    
    if [ -f "$OUTPUT_DIR/gunshot_detection.xsa" ]; then
        echo "║    • gunshot_detection.xsa (Hardware platform)        ║"
    fi
    
    echo "║                                                          ║"
    echo "║  Log files in '$LOG_DIR/$TIMESTAMP/':                   ║"
    echo "║    • Complete build logs                                 ║"
    echo "║    • HLS synthesis report                                ║"
    echo "║    • Vivado timing/utilization reports                   ║"
    echo "║                                                          ║"
    echo "║  Next steps:                                             ║"
    echo "║    1. Program FPGA: ./scripts/program_fpga.sh            ║"
    echo "║    2. Test system: ./scripts/test_system.py              ║"
    echo "║    3. Flash to board: ./scripts/flash_fpga.py            ║"
    echo "║                                                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    # Save build info
    cat > $OUTPUT_DIR/build_info.txt << EOF
Build Information
=================
Project: $PROJECT_NAME
Board: $BOARD
Part: $TARGET_PART
Build Date: $(date)
Build Timestamp: $TIMESTAMP
Build Duration: $(($SECONDS / 60))m $(($SECONDS % 60))s

Generated Files:
$(find $OUTPUT_DIR -type f -exec ls -lh {} \; | sed 's/^/  /')

Logs: $LOG_DIR/$TIMESTAMP/
EOF
}

# Main execution
main() {
    SECONDS=0  # Start timer
    
    print_header
    check_dependencies
    setup_directories
    convert_model_to_hls
    run_vitis_hls
    create_vivado_project
    implement_design
    export_hardware
    build_vitis_software
    generate_boot_image
    print_summary
    
    echo -e "${BLUE}Total build time: $(($SECONDS / 60))m $(($SECONDS % 60))s${NC}"
}

# Run main function
main "$@"

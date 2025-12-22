#!/bin/bash

# ================================================
# FPGA Programming Script
# Spartan Edge-7 Accelerator Board
# ================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_NAME="gunshot_detection_fpga"
OUTPUT_DIR="output"
SD_CARD_MOUNT="/media/$USER/BOOT"  # Adjust for your system
BOARD_SERIAL=""  # Leave empty to auto-detect
PROGRAM_METHOD="sd_card"  # Options: sd_card, jtag, xsct

# Print header
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          FPGA Programming Utility                        ║"
echo "║          Spartan Edge-7 Accelerator Board                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if output files exist
check_files() {
    echo -e "${YELLOW}[1/5] Checking required files...${NC}"
    
    local missing_files=()
    
    # Check for bitstream
    if [ ! -f "$OUTPUT_DIR/bitstreams/gunshot_detection.bit" ]; then
        missing_files+=("gunshot_detection.bit")
    else
        BIT_FILE="$OUTPUT_DIR/bitstreams/gunshot_detection.bit"
        echo "  ✓ Bitstream found: $(basename $BIT_FILE)"
    fi
    
    # Check for boot image
    if [ ! -f "$OUTPUT_DIR/boot.bin" ]; then
        missing_files+=("boot.bin")
    else
        BOOT_FILE="$OUTPUT_DIR/boot.bin"
        echo "  ✓ Boot image found: $(basename $BOOT_FILE)"
    fi
    
    # Check for application
    if [ ! -f "$OUTPUT_DIR/software/application.elf" ]; then
        echo "  ⚠ Application not found, will program bitstream only"
    else
        APP_FILE="$OUTPUT_DIR/software/application.elf"
        echo "  ✓ Application found: $(basename $APP_FILE)"
    fi
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        echo -e "${RED}Missing required files:${NC}"
        for file in "${missing_files[@]}"; do
            echo "  ✗ $file"
        done
        echo -e "\nRun ${GREEN}./scripts/complete_build.sh${NC} first"
        exit 1
    fi
    
    echo -e "${GREEN}  All required files found!${NC}"
}

# Detect programming method
detect_programming_method() {
    echo -e "${YELLOW}[2/5] Detecting programming method...${NC}"
    
    # Check for SD card
    if [ -d "$SD_CARD_MOUNT" ]; then
        echo "  ✓ SD card detected at: $SD_CARD_MOUNT"
        PROGRAM_METHOD="sd_card"
        return 0
    fi
    
    # Check for JTAG via xsct
    if command -v xsct &> /dev/null; then
        echo "  ✓ XSCT (JTAG) available"
        PROGRAM_METHOD="xsct"
        return 0
    fi
    
    # Check for vivado_lab
    if command -v vivado_lab &> /dev/null; then
        echo "  ✓ Vivado Lab available"
        PROGRAM_METHOD="vivado_lab"
        return 0
    fi
    
    echo -e "${RED}No programming method detected!${NC}"
    echo "Please connect:"
    echo "  1. SD card to computer (mounted at $SD_CARD_MOUNT)"
    echo "  2. JTAG cable and install Vivado Lab Edition"
    echo "  3. Or use direct programming with xsct"
    
    read -p "Enter programming method (sd_card/jtag/xsct): " PROGRAM_METHOD
}

# Prepare SD card
prepare_sd_card() {
    echo -e "${YELLOW}[3/5] Preparing SD card...${NC}"
    
    # Check if SD card is mounted
    if [ ! -d "$SD_CARD_MOUNT" ]; then
        echo -e "${RED}SD card not mounted at $SD_CARD_MOUNT${NC}"
        echo "Please insert SD card and mount it, or update SD_CARD_MOUNT in script"
        
        # Try to find SD card
        echo "Searching for SD card..."
        possible_mounts=$(ls -d /media/$USER/* 2>/dev/null || ls -d /Volumes/* 2>/dev/null)
        
        if [ -n "$possible_mounts" ]; then
            echo "Found possible mount points:"
            select mount in $possible_mounts; do
                SD_CARD_MOUNT=$mount
                break
            done
        else
            read -p "Enter SD card mount path: " SD_CARD_MOUNT
        fi
    fi
    
    # Confirm SD card
    echo "SD card path: $SD_CARD_MOUNT"
    ls -la "$SD_CARD_MOUNT/"
    
    read -p "Is this the correct SD card? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting..."
        exit 1
    fi
    
    # Backup existing files
    echo "Backing up existing files..."
    BACKUP_DIR="$HOME/sd_card_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    cp -r "$SD_CARD_MOUNT/"* "$BACKUP_DIR/" 2>/dev/null || true
    echo "Backup saved to: $BACKUP_DIR"
    
    # Clear SD card
    echo "Clearing SD card..."
    rm -rf "$SD_CARD_MOUNT"/*
    
    # Copy files
    echo "Copying files to SD card..."
    
    # Boot files
    cp "$BOOT_FILE" "$SD_CARD_MOUNT/"
    
    # Optional: Device tree and u-boot
    if [ -f "$OUTPUT_DIR/software/devicetree.dtb" ]; then
        cp "$OUTPUT_DIR/software/devicetree.dtb" "$SD_CARD_MOUNT/"
    fi
    
    if [ -f "$OUTPUT_DIR/software/u-boot.elf" ]; then
        cp "$OUTPUT_DIR/software/u-boot.elf" "$SD_CARD_MOUNT/"
    fi
    
    # Create README
    cat > "$SD_CARD_MOUNT/README.txt" << EOF
Gunshot Detection System - Spartan Edge-7
=========================================

Files on this SD card:
- boot.bin: Boot image with FPGA bitstream and application
- README.txt: This file

Boot sequence:
1. Insert SD card into Spartan Edge-7
2. Set boot mode to SD card (check board jumpers)
3. Power on the board
4. FPGA will configure automatically
5. Application will start running

For debugging:
- Connect UART to PC (115200 baud, 8N1)
- Use 'screen /dev/ttyUSB0 115200' to view output

Generated on: $(date)
Project: $PROJECT_NAME
EOF
    
    # Verify copy
    echo "Verifying copy..."
    ls -la "$SD_CARD_MOUNT/"
    
    # Sync to ensure writes complete
    sync
    
    echo -e "${GREEN}  SD card prepared successfully!${NC}"
}

# Program via JTAG/XSCT
program_via_jtag() {
    echo -e "${YELLOW}[3/5] Programming via JTAG...${NC}"
    
    # Create XSCT script
    cat > program_fpga.tcl << EOF
# XSCT script for programming Spartan Edge-7

# Connect to hardware server
connect -url tcp:localhost:3121

# Select target
targets -set -filter {name =~ "*microblaze*"}

# Load bitstream
fpga -file "$BIT_FILE"

# Load application if available
if {[file exists "$APP_FILE"]} {
    # For MicroBlaze systems
    dow "$APP_FILE"
    con
} else {
    puts "No application found, only bitstream loaded"
}

puts "Programming complete!"
EOF
    
    # Check if hardware server is running
    if ! ps aux | grep -q "[h]w_server"; then
        echo "Starting hardware server..."
        hw_server &
        sleep 2
    fi
    
    # Run programming
    echo "Programming FPGA..."
    xsct program_fpga.tcl
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  JTAG programming successful!${NC}"
    else
        echo -e "${RED}  JTAG programming failed!${NC}"
        echo "Make sure:"
        echo "  1. JTAG cable is connected"
        echo "  2. Board is powered on"
        echo "  3. Vivado Hardware Server is running"
    fi
}

# Program via Vivado Lab
program_via_vivado_lab() {
    echo -e "${YELLOW}[3/5] Programming via Vivado Lab...${NC}"
    
    # Create TCL script for Vivado Lab
    cat > program_vivado.tcl << EOF
# Vivado TCL script for programming

open_hw
connect_hw_server
open_hw_target

# Set programming file
set_property PROGRAM.FILE {$BIT_FILE} [get_hw_devices xc7s50_0]

# Program device
program_hw_devices [get_hw_devices xc7s50_0]
refresh_hw_device [get_hw_devices xc7s50_0]

puts "Programming complete!"
EOF
    
    echo "Programming FPGA using Vivado Lab..."
    vivado_lab -mode batch -source program_vivado.tcl
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  Vivado Lab programming successful!${NC}"
    else
        echo -e "${RED}  Vivado Lab programming failed!${NC}"
    fi
}

# Test connection
test_connection() {
    echo -e "${YELLOW}[4/5] Testing connection...${NC}"
    
    # Try to detect UART
    echo "Looking for UART devices..."
    UART_DEVICES=$(ls /dev/ttyUSB* 2>/dev/null || ls /dev/ttyACM* 2>/dev/null || echo "")
    
    if [ -n "$UART_DEVICES" ]; then
        echo "Found UART devices:"
        for dev in $UART_DEVICES; do
            echo "  $dev"
        done
        
        read -p "Test UART connection? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Try to read from UART
            echo "Reading from UART (press Ctrl+C to stop)..."
            timeout 5 cat /dev/ttyUSB0 2>/dev/null || echo "No data received"
        fi
    else
        echo "No UART devices found"
    fi
}

# Final instructions
print_instructions() {
    echo -e "${YELLOW}[5/5] Programming complete!${NC}"
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              NEXT STEPS                                  ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    
    case $PROGRAM_METHOD in
        "sd_card")
            echo "║                                                          ║"
            echo "║  1. Safely eject SD card:                               ║"
            echo "║     sudo umount $SD_CARD_MOUNT                   ║"
            echo "║                                                          ║"
            echo "║  2. Insert SD card into Spartan Edge-7                  ║"
            echo "║                                                          ║"
            echo "║  3. Set boot mode jumpers to SD card:                   ║"
            echo "║     • J11: Closed (SD card boot)                        ║"
            echo "║     • Check board manual for details                    ║"
            echo "║                                                          ║"
            echo "║  4. Connect:                                            ║"
            echo "║     • USB-C for power                                   ║"
            echo "║     • UART for debugging (115200 baud)                  ║"
            echo "║     • 3x microphones to SPI ADC board                   ║"
            echo "║                                                          ║"
            echo "║  5. Power on the board                                  ║"
            echo "║                                                          ║"
            echo "║  6. Monitor output:                                     ║"
            echo "║     screen /dev/ttyUSB0 115200                          ║"
            ;;
        "jtag"|"xsct")
            echo "║                                                          ║"
            echo "║  1. FPGA is now programmed via JTAG                     ║"
            echo "║                                                          ║"
            echo "║  2. Connect:                                            ║"
            echo "║     • USB-C for power                                   ║"
            echo "║     • UART for debugging (115200 baud)                  ║"
            echo "║     • 3x microphones to SPI ADC board                   ║"
            echo "║                                                          ║"
            echo "║  3. Application should be running                       ║"
            echo "║                                                          ║"
            echo "║  4. Monitor output:                                     ║"
            echo "║     screen /dev/ttyUSB0 115200                          ║"
            ;;
    esac
    
    echo "║                                                          ║"
    echo "║  Test the system:                                        ║"
    echo "║     ./scripts/test_system.py                             ║"
    echo "║                                                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Main execution
main() {
    check_files
    detect_programming_method
    
    case $PROGRAM_METHOD in
        "sd_card")
            prepare_sd_card
            ;;
        "xsct")
            program_via_jtag
            ;;
        "vivado_lab")
            program_via_vivado_lab
            ;;
        *)
            echo -e "${RED}Unknown programming method: $PROGRAM_METHOD${NC}"
            exit 1
            ;;
    esac
    
    test_connection
    print_instructions
}

# Run main
main "$@"

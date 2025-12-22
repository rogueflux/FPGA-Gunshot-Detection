#!/bin/bash

# ================================================
# Environment Setup Script
# Gunshot Detection System
# ================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REQUIRED_PYTHON_VERSION="3.8"
VIVADO_VERSION="2022.2"
VITIS_HLS_VERSION="2022.2"
PROJECT_DIR=$(pwd)
VENV_DIR="$PROJECT_DIR/venv"
LOG_FILE="$PROJECT_DIR/setup.log"

# Print header
print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║     Environment Setup for Gunshot Detection System       ║"
    echo "║                                                          ║"
    echo "║     This script will install all required dependencies   ║"
    echo "║     for development, simulation, and FPGA deployment     ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    echo -e "${YELLOW}Project Directory: $PROJECT_DIR${NC}"
    echo -e "${YELLOW}Python Virtual Environment: $VENV_DIR${NC}"
    echo -e "${YELLOW}Log File: $LOG_FILE${NC}"
    echo ""
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        echo -e "${RED}Error: Do not run this script as root${NC}"
        echo "Please run as a regular user"
        exit 1
    fi
}

# Check Python version
check_python() {
    echo -e "${YELLOW}[1/10] Checking Python version...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python3 not found!${NC}"
        echo "Installing Python3..."
        
        # Detect OS
        if [[ -f /etc/debian_version ]]; then
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
        elif [[ -f /etc/redhat-release ]]; then
            sudo yum install -y python3 python3-pip
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            brew install python3
        else
            echo -e "${RED}Unsupported OS. Please install Python3 manually${NC}"
            exit 1
        fi
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    echo "  Python version: $PYTHON_VERSION"
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MINOR -lt 8 ]]; then
        echo -e "${RED}Python 3.8 or higher required${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}  Python version OK!${NC}"
}

# Create virtual environment
create_venv() {
    echo -e "${YELLOW}[2/10] Creating Python virtual environment...${NC}"
    
    if [ -d "$VENV_DIR" ]; then
        echo "  Virtual environment already exists"
        read -p "  Recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "  Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            echo "  Using existing virtual environment"
            return 0
        fi
    fi
    
    echo "  Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  Virtual environment created!${NC}"
    else
        echo -e "${RED}  Failed to create virtual environment${NC}"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    echo "  Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    # Update pip
    echo "  Upgrading pip..."
    pip install --upgrade pip
    
    echo -e "${GREEN}  Virtual environment activated!${NC}"
}

# Install Python packages
install_python_packages() {
    echo -e "${YELLOW}[3/10] Installing Python packages...${NC}"
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "requirements.txt" ]; then
        echo "  Creating requirements.txt..."
        cat > requirements.txt << EOF
# Core dependencies
torch>=1.13.0
torchvision>=0.14.0
numpy>=1.21.0
scipy>=1.7.0

# Audio processing
librosa>=0.9.0
soundfile>=0.10.0
pyaudio>=0.2.11
pydub>=0.25.1

# Machine learning
scikit-learn>=1.0.0
onnx>=1.12.0
onnxruntime>=1.13.0
tensorboard>=2.11.0

# Data processing
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
notebook>=6.4.0
ipython>=8.5.0

# Development
black>=22.0.0
flake8>=5.0.0
pytest>=7.0.0
tqdm>=4.64.0

# Serial communication
pyserial>=3.5

# FPGA tools (Python interface)
xrfclk>=0.1.0
pynq>=2.7.0

# Utilities
pyyaml>=6.0
json5>=0.9.0
psutil>=5.9.0
EOF
    fi
    
    echo "  Installing packages from requirements.txt..."
    pip install -r requirements.txt
    
    # Install additional development packages
    echo "  Installing development packages..."
    pip install jupyterlab ipywidgets
    
    echo -e "${GREEN}  Python packages installed!${NC}"
}

# Install system dependencies
install_system_deps() {
    echo -e "${YELLOW}[4/10] Installing system dependencies...${NC}"
    
    # Detect OS
    if [[ -f /etc/debian_version ]]; then
        echo "  Detected Debian/Ubuntu system"
        
        echo "  Updating package list..."
        sudo apt-get update
        
        echo "  Installing development tools..."
        sudo apt-get install -y \
            build-essential \
            cmake \
            git \
            wget \
            curl \
            software-properties-common \
            libssl-dev \
            libffi-dev \
            python3-dev \
            libportaudio2 \
            portaudio19-dev \
            libsndfile1 \
            ffmpeg \
            udev \
            usbutils
        
        # For Vivado/Vitis
        echo "  Installing Vivado dependencies..."
        sudo apt-get install -y \
            libtinfo5 \
            libncurses5 \
            libtinfo-dev \
            ncurses-dev \
            libncursesw5-dev \
            libncursesw5 \
            libusb-1.0-0-dev \
            libc6-dev-i386 \
            gcc-multilib \
            lib32z1 \
            lib32stdc++6
        
    elif [[ -f /etc/redhat-release ]]; then
        echo "  Detected RedHat/CentOS system"
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            cmake \
            git \
            wget \
            curl \
            openssl-devel \
            libffi-devel \
            python3-devel \
            portaudio-devel \
            libsndfile-devel \
            ffmpeg
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Detected macOS system"
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "  Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        echo "  Installing dependencies via Homebrew..."
        brew install \
            cmake \
            git \
            wget \
            curl \
            portaudio \
            libsndfile \
            ffmpeg \
            python-tk
        
    else
        echo -e "${YELLOW}  Unsupported OS. Please install dependencies manually${NC}"
    fi
    
    echo -e "${GREEN}  System dependencies installed!${NC}"
}

# Setup udev rules for FPGA board
setup_udev_rules() {
    echo -e "${YELLOW}[5/10] Setting up udev rules for FPGA board...${NC}"
    
    UDEV_RULES_FILE="/etc/udev/rules.d/99-spartan-edge.rules"
    
    if [ ! -f "$UDEV_RULES_FILE" ]; then
        echo "  Creating udev rules for Spartan Edge-7..."
        
        cat > /tmp/99-spartan-edge.rules << EOF
# Spartan Edge-7 Accelerator Board
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6010", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6014", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="10c4", ATTR{idProduct}=="ea60", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666"

# Digilent JTAG cables
SUBSYSTEM=="usb", ATTR{idVendor}=="1443", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6010", MODE="0666"

# Xilinx JTAG cables
SUBSYSTEM=="usb", ATTR{idVendor}=="03fd", MODE="0666"
EOF
        
        echo "  Installing udev rules..."
        sudo cp /tmp/99-spartan-edge.rules "$UDEV_RULES_FILE"
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        
        echo "  Adding user to dialout group for serial access..."
        sudo usermod -a -G dialout $USER
        
        echo -e "${GREEN}  Udev rules configured!${NC}"
        echo "  Note: You may need to logout and login again for group changes to take effect"
    else
        echo "  Udev rules already exist"
    fi
}

# Check for Vivado/Vitis
check_vivado_vitis() {
    echo -e "${YELLOW}[6/10] Checking for Xilinx tools...${NC}"
    
    local missing_tools=()
    
    # Check Vivado
    if ! command -v vivado &> /dev/null; then
        missing_tools+=("Vivado $VIVADO_VERSION")
        echo "  ✗ Vivado not found"
    else
        VIVADO_PATH=$(which vivado)
        echo "  ✓ Vivado found: $VIVADO_PATH"
    fi
    
    # Check Vitis HLS
    if ! command -v vitis_hls &> /dev/null; then
        missing_tools+=("Vitis HLS $VITIS_HLS_VERSION")
        echo "  ✗ Vitis HLS not found"
    else
        VITIS_HLS_PATH=$(which vitis_hls)
        echo "  ✓ Vitis HLS found: $VITIS_HLS_PATH"
    fi
    
    # Check xsct
    if ! command -v xsct &> /dev/null; then
        missing_tools+=("XSCT")
        echo "  ✗ XSCT not found"
    else
        echo "  ✓ XSCT found"
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo -e "${YELLOW}  Missing Xilinx tools:${NC}"
        for tool in "${missing_tools[@]}"; do
            echo "    • $tool"
        done
        
        echo -e "\n${YELLOW}Installation options:${NC}"
        echo "  1. Download from Xilinx website: https://www.xilinx.com/support/download.html"
        echo "  2. Use package manager (if available)"
        echo "  3. Continue without FPGA tools (simulation only)"
        
        read -p "Continue without FPGA tools? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Please install missing tools and run setup again"
            exit 1
        fi
    else
        echo -e "${GREEN}  All Xilinx tools found!${NC}"
    fi
}

# Setup project structure
setup_project_structure() {
    echo -e "${YELLOW}[7/10] Setting up project structure...${NC}"
    
    # Create directory structure
    DIRECTORIES=(
        "data/audio_samples/gunshots"
        "data/audio_samples/background"
        "data/audio_samples/validation"
        "data/test_vectors"
        "output/bitstreams"
        "output/software"
        "output/models"
        "logs"
        "notebooks"
        "models/onnx_models"
        "models/hls_models"
        "scripts"
        "docs"
        "tests/unit_tests"
        "tests/integration_tests"
        "tests/hardware_tests"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "  Creating $dir"
            mkdir -p "$dir"
        fi
    done
    
    # Create placeholder files
    PLACEHOLDER_FILES=(
        "data/audio_samples/README.md"
        "data/test_vectors/README.md"
        "models/gunshot_model_weights.pth"
        "models/model_config.json"
        "docs/architecture.md"
        "docs/deployment_guide.md"
        ".gitignore"
    )
    
    for file in "${PLACEHOLDER_FILES[@]}"; do
        if [ ! -f "$file" ]; then
            echo "  Creating $file"
            touch "$file"
        fi
    done
    
    # Create .gitignore if not exists
    if [ ! -s ".gitignore" ]; then
        echo "  Creating .gitignore..."
        cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# Virtual Environment
venv/
env/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data files
data/audio_samples/*.wav
data/test_vectors/*.dat
*.pth
*.h5
*.pkl

# Output files
output/*
!output/.gitkeep
logs/*

# Vivado/Vitis
*.jou
*.log
*.str
*.xpr
*.bit
*.bin
*.elf
*.mcs
*.prm
.xilinx/
.cache/
.hw/
.sim/
.ip_user_files/
*.xsa

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary files
*~
*.tmp
*.temp
EOF
    fi
    
    echo -e "${GREEN}  Project structure created!${NC}"
}

# Setup development tools
setup_dev_tools() {
    echo -e "${YELLOW}[8/10] Setting up development tools...${NC}"
    
    # Git configuration
    echo "  Configuring Git..."
    if [ ! -f ".git/config" ]; then
        git init
    fi
    
    # Pre-commit hook
    echo "  Setting up pre-commit hook..."
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for gunshot detection project

echo "Running pre-commit checks..."

# Check Python syntax
echo "  Checking Python syntax..."
python3 -m py_compile $(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ $? -ne 0 ]; then
    echo "Python syntax check failed!"
    exit 1
fi

# Run tests
echo "  Running unit tests..."
python3 -m pytest tests/unit_tests/ -v

if [ $? -ne 0 ]; then
    echo "Unit tests failed!"
    exit 1
fi

echo "Pre-commit checks passed!"
EOF
    
    chmod +x .git/hooks/pre-commit
    
    # Create Makefile for common tasks
    if [ ! -f "Makefile" ]; then
        echo "  Creating Makefile..."
        cat > Makefile << 'EOF'
# Makefile for Gunshot Detection System

.PHONY: help setup test build program clean

help:
	@echo

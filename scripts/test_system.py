#!/usr/bin/env python3
"""
System Test Script for Gunshot Detection
Tests hardware, software, and neural network components
"""

import os
import sys
import time
import serial
import numpy as np
import subprocess
import threading
import json
from datetime import datetime
from pathlib import Path
import soundfile as sf
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'uart_port': '/dev/ttyUSB0',
    'uart_baud': 115200,
    'timeout': 2,
    'test_duration': 10,  # seconds
    'sample_rate': 16000,
    'test_audio_dir': 'data/test_vectors',
    'output_dir': 'test_results',
    'fpga_bitstream': 'output/bitstreams/gunshot_detection.bit',
    'boot_image': 'output/boot.b

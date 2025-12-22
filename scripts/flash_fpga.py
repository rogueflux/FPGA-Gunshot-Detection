#!/usr/bin/env python3
"""
FPGA Flashing Utility
Advanced programming and monitoring for Spartan Edge-7
"""

import os
import sys
import time
import serial
import subprocess
import threading
import hashlib
import json
from pathlib import Path
from datetime import datetime
import argparse
import tempfile

class FPGAFlasher:
    """Advanced FPGA flashing and monitoring utility"""
    
    def __init__(self, config_file='config/flash_config.json'):
        self.config = self.load_config(config_file)
        self.serial_port = None
        self.flash_log = []
        self.programming_methods = ['jtag', 'sd_card', 'qspi', 'jtag_proxy']
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            'board': {
                'name': 'Spartan Edge-7 Accelerator',
                'part': 'xc7s50csga324-1',
                'jtag_id': '0403:6010',
                'uart_baud': 115200
            },
            'paths': {
                'bitstream': 'output/bitstreams/gunshot_detection.bit',
                'boot_image': 'output/boot.bin',
                'application': 'output/software/application.elf',
                'output_dir': 'flash_output'
            },
            'programming': {
                'default_method': 'sd_card',
                'verify_write': True,
                'max_retries': 3,
                'timeout': 30
            },
            'monitoring': {
                'enable_uart': True,
                'enable_jtag': False,
                'log_level': 'INFO'
            }
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                self.merge_dicts(default_config, user_config)
        
        # Create output directory
        Path(default_config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)
        
        return default_config
    
    def merge_dicts(self, base, update):
        """Recursively merge dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self.merge_dicts(base[key], value)
            else:
                base[key] = value
    
    def log(self, message, level='INFO'):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.flash_log.append(log_entry)
        
        # Also write to log file
        log_file = Path(self.config['paths']['output_dir']) / 'flash.log'
        with open(log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def detect_hardware(self):
        """Detect connected FPGA hardware"""
        self.log("Detecting hardware...")
        
        devices = {
            'jtag': False,
            'uart': False,
            'sd_card': False
        }
        
        # Check for JTAG devices
        try:
            if sys.platform == 'linux':
                # Check for FTDI devices (common for Spartan boards)
                result = subprocess.run(['lsusb'], capture_output=True, text=True)
                if '0403:6010' in result.stdout or 'Future Technology' in result.stdout:
                    devices['jtag'] = True
                    self.log("JTAG device detected")
            elif sys.platform == 'darwin':
                result = subprocess.run(['system_profiler', 'SPUSBDataType'], 
                                      capture_output=True, text=True)
                if 'FTDI' in result.stdout or 'Future Technology' in result.stdout:
                    devices['jtag'] = True
                    self.log("JTAG device detected")
        except Exception as e:
            self.log(f"JTAG detection error: {e}", 'WARNING')
        
        # Check for UART devices
        try:
            if sys.platform == 'linux':
                ports = list(Path('/dev').glob('ttyUSB*'))
                if ports:
                    devices['uart'] = True
                    self.log(f"UART devices found: {ports}")
            elif sys.platform == 'darwin':
                ports = list(Path('/dev').glob('tty.usbserial*'))
                if ports:
                    devices['uart'] = True
                    self.log(f"UART devices found: {ports}")
        except Exception as e:
            self.log(f"UART detection error: {e}", 'WARNING')
        
        # Check for SD card
        if sys.platform == 'linux':
            if Path('/media').exists():
                devices['sd_card'] = True
                self.log("SD card mount point available")
        
        return devices
    
    def verify_files(self):
        """Verify programming files exist and are valid"""
        self.log("Verifying programming files...")
        
        files_to_check = [
            ('bitstream', self.config['paths']['bitstream']),
            ('boot_image', self.config['paths']['boot_image'])
        ]
        
        for file_type, file_path in files_to_check:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"{file_type} not found: {file_path}")
            
            # Check file size
            size = path.stat().st_size
            self.log(f"{file_type}: {path.name} ({size:,} bytes)")
            
            # Calculate checksum
            with open(path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            
            self.log(f"{file_type} checksum: {file_hash}")
        
        return True
    
    def program_via_jtag(self, bitstream_path):
        """Program FPGA via JTAG using Vivado Hardware Server"""
        self.log("Programming via JTAG...")
        
        # Create TCL script for programming
        tcl_script = f"""
# JTAG Programming Script
open_hw
connect_hw_server
open_hw_target

# Set programming file
set_property PROGRAM.FILE {{{bitstream_path}}} [get_hw_devices xc7s50_0]

# Program device
program_hw_devices [get_hw_devices xc7s50_0]
refresh_hw_device [get_hw_devices xc7s50_0]

puts "Programming complete!"
"""
        
        # Write TCL script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tcl', delete=False) as f:
            f.write(tcl_script)
            tcl_file = f.name
        
        try:
            # Run Vivado in batch mode
            self.log("Starting Vivado Hardware Server...")
            
            # Start hw_server if not running
            if not self.is_hw_server_running():
                subprocess.Popen(['hw_server'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(2)
            
            # Program using vivado_lab or vivado
            program_cmd = 'vivado_lab' if self.command_exists('vivado_lab') else 'vivado'
            
            self.log(f"Running {program_cmd}...")
            result = subprocess.run(
                [program_cmd, '-mode', 'batch', '-source', tcl_file],
                capture_output=True,
                text=True,
                timeout=self.config['programming']['timeout']
            )
            
            if result.returncode == 0:
                self.log("JTAG programming successful!")
                return True
            else:
                self.log(f"JTAG programming failed: {result.stderr}", 'ERROR')
                return False
                
        except subprocess.TimeoutExpired:
            self.log("JTAG programming timeout", 'ERROR')
            return False
        finally:
            # Clean up temp file
            os.unlink(tcl_file)
    
    def program_via_sd_card(self, boot_image_path):
        """Program by writing to SD card"""
        self.log("Programming via SD card...")
        
        # Find SD card
        sd_card_path = self.find_sd_card()
        
        if not sd_card_path:
            self.log("No SD card found", 'ERROR')
            return False
        
        self.log(f"SD card found at: {sd_card_path}")
        
        # Backup existing files
        backup_dir = self.backup_sd_card(sd_card_path)
        self.log(f"Backup created at: {backup_dir}")
        
        # Copy files
        try:
            # Clear SD card
            for item in Path(sd_card_path).iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)
            
            # Copy boot image
            boot_image = Path(boot_image_path)
            target_path = Path(sd_card_path) / boot_image.name
            import shutil
            shutil.copy2(boot_image, target_path)
            
            # Create boot files
            self.create_boot_files(sd_card_path)
            
            # Sync to ensure writes
            subprocess.run(['sync'], check=True)
            
            self.log("SD card programming successful!")
            return True
            
        except Exception as e:
            self.log(f"SD card programming failed: {e}", 'ERROR')
            
            # Restore backup
            self.restore_sd_card_backup(sd_card_path, backup_dir)
            return False
    
    def find_sd_card(self):
        """Find mounted SD card"""
        if sys.platform == 'linux':
            # Common mount points
            possible_paths = [
                f'/media/{os.getenv("USER")}/BOOT',
                '/media/BOOT',
                '/mnt/sd',
                '/Volumes/BOOT'  # macOS
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    return path
            
            # Try to find by label
            try:
                result = subprocess.run(['lsblk', '-o', 'MOUNTPOINT,LABEL'], 
                                      capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if 'BOOT' in line:
                        mount_point = line.split()[0]
                        if mount_point and mount_point != 'MOUNTPOINT':
                            return mount_point
            except:
                pass
        
        return None
    
    def backup_sd_card(self, sd_card_path):
        """Backup SD card contents"""
        backup_dir = Path(self.config['paths']['output_dir']) / 'backups' / f'backup_{int(time.time())}'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        for item in Path(sd_card_path).iterdir():
            dest = backup_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                shutil.copytree(item, dest)
        
        return backup_dir
    
    def restore_sd_card_backup(self, sd_card_path, backup_dir):
        """Restore SD card from backup"""
        import shutil
        for item in Path(backup_dir).iterdir():
            dest = Path(sd_card_path) / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            elif item.is_dir():
                shutil.copytree(item, dest)
    
    def create_boot_files(self, sd_card_path):
        """Create additional boot files on SD card"""
        sd_path = Path(sd_card_path)
        
        # Create boot.bif
        boot_bif = sd_path / 'boot.bif'
        with open(boot_bif, 'w') as f:
            f.write("""// Boot Image Format
the_ROM_image:
{
    [bootloader] fsbl.elf
    gunshot_detection.bit
    application.elf
}
""")
        
        # Create uEnv.txt for U-Boot
        uenv_txt = sd_path / 'uEnv.txt'
        with open(uenv_txt, 'w') as f:
            f.write("""bootcmd=load mmc 0 0x1000000 gunshot_detection.bit && fpga loadb 0 0x1000000 ${filesize} && bootm
bootargs=console=ttyPS0,115200 root=/dev/mmcblk0p2 rw earlyprintk rootfstype=ext4 rootwait
""")
        
        # Create README
        readme = sd_path / 'README.txt'
        with open(readme, 'w') as f:
            f.write(f"""Gunshot Detection System - Spartan Edge-7
=========================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Boot Instructions:
1. Insert SD card into Spartan Edge-7
2. Set boot mode jumpers to SD card
3. Connect USB-C for power
4. Connect UART for debugging (115200 baud)
5. Power on the board

Debugging:
- UART: screen /dev/ttyUSB0 115200
- Press 's' for status, 'h' for history

Project: https://github.com/your-repo/gunshot-detection
""")
    
    def monitor_uart(self, port, baud_rate=115200, duration=30):
        """Monitor UART output during and after programming"""
        self.log(f"Monitoring UART {port} at {baud_rate} baud...")
        
        try:
            ser = serial.Serial(port, baud_rate, timeout=1)
            
            # Start monitoring thread
            stop_monitoring = threading.Event()
            monitor_thread = threading.Thread(
                target=self._uart_monitor_thread,
                args=(ser, stop_monitoring)
            )
            monitor_thread.start()
            
            # Wait for specified duration
            time.sleep(duration)
            
            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()
            
            ser.close()
            
        except serial.SerialException as e:
            self.log(f"UART monitoring failed: {e}", 'WARNING')
    
    def _uart_monitor_thread(self, ser, stop_event):
        """Thread for monitoring UART"""
        log_file = Path(self.config['paths']['output_dir']) / 'uart_log.txt'
        
        with open(log_file, 'w') as f:
            f.write(f"UART Log - Started at {datetime.now()}\n")
            f.write("="*50 + "\n")
            
            while not stop_event.is_set():
                try:
                    if ser.in_waiting > 0:
                        data = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                        print(data, end='')
                        f.write(data)
                        
                        # Check for boot messages
                        if 'FPGA configured successfully' in data:
                            self.log("FPGA configuration detected in UART output")
                        if 'Gunshot Detection System' in data:
                            self.log("Application started successfully")
                        
                        # Send test command
                        if 'Ready' in data:
                            ser.write(b'test\n')
                    
                    time.sleep(0.1)
                except Exception as e:
                    self.log(f"UART read error: {e}", 'WARNING')
                    break
    
    def verify_programming(self):
        """Verify successful programming"""
        self.log("Verifying programming...")
        
        verification_passed = True
        
        # Check if we can communicate with the board
        try:
            # Try to find UART port
            uart_port = self.find_uart_port()
            if uart_port:
                self.log(f"Testing communication on {uart_port}...")
                
                ser = serial.Serial(uart_port, self.config['board']['uart_baud'], timeout=2)
                
                # Send status command
                ser.write(b's\n')
                time.sleep(0.1)
                
                # Read response
                response = ser.read(100).decode('utf-8', errors='ignore')
                ser.close()
                
                if response and ('status' in response.lower() or 'uptime' in response.lower()):
                    self.log("Board communication verified")
                else:
                    self.log("No valid response from board", 'WARNING')
                    verification_passed = False
            else:
                self.log("No UART port found for verification", 'WARNING')
        
        except Exception as e:
            self.log(f"Verification failed: {e}", 'WARNING')
            verification_passed = False
        
        return verification_passed
    
    def find_uart_port(self):
        """Find UART port for the board"""
        if sys.platform == 'linux':
            ports = list(Path('/dev').glob('ttyUSB*'))
            if ports:
                return str(ports[0])
        elif sys.platform == 'darwin':
            ports = list(Path('/dev').glob('tty.usbserial*'))
            if ports:
                return str(ports[0])
        return None
    
    def is_hw_server_running(self):
        """Check if Vivado Hardware Server is running"""
        try:
            result = subprocess.run(['pgrep', 'hw_server'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def command_exists(self, cmd):
        """Check if command exists in PATH"""
        return subprocess.run(['which', cmd], 
                            capture_output=True, text=True).returncode == 0
    
    def generate_report(self):
        """Generate flashing report"""
        report_path = Path(self.config['paths']['output_dir']) / f'flash_report_{int(time.time())}.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'log': self.flash_log,
            'summary': {
                'files_verified': self.verify_files(),
                'hardware_detected': self.detect_hardware(),
                'programming_method': self.config['programming']['default_method'],
                'success': True  # Will be updated
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Report saved to: {report_path}")
        return report_path
    
    def run(self, method=None, verify=True):
        """Main flashing routine"""
        self.log("Starting FPGA flashing process")
        
        try:
            # Step 1: Verify files
            self.verify_files()
            
            # Step 2: Detect hardware
            devices = self.detect_hardware()
            
            # Step 3: Determine programming method
            if not method:
                method = self.config['programming']['default_method']
            
            if method not in self.programming_methods:
                raise ValueError(f"Unknown programming method: {method}")
            
            self.log(f"Using programming method: {method}")
            
            # Step 4: Program
            success = False
            if method == 'jtag':
                success = self.program_via_jtag(self.config['paths']['bitstream'])
            elif method == 'sd_card':
                success = self.program_via_sd_card(self.config['paths']['boot_image'])
            else:
                self.log(f"Method {method} not yet implemented", 'ERROR')
                success = False
            
            # Step 5: Verify
            if success and verify:
                self.log("Verifying programming...")
                verification = self.verify_programming()
                if not verification:
                    self.log("Verification failed", 'WARNING')
                    success = False
            
            # Step 6: Monitor
            if success:
                self.log("Starting post-programming monitoring...")
                uart_port = self.find_uart_port()
                if uart_port:
                    # Monitor in background
                    monitor_thread = threading.Thread(
                        target=self.monitor_uart,
                        args=(uart_port, self.config['board']['uart_baud'], 10)
                    )
                    monitor_thread.daemon = True
                    monitor_thread.start()
            
            # Step 7: Generate report
            report_path = self.generate_report()
            
            if success:
                self.log(f"Flashing completed successfully!")
                self.log(f"Report: {report_path}")
                return True
            else:
                self.log("Flashing failed", 'ERROR')
                return False
                
        except Exception as e:
            self.log(f"Flashing process failed: {e}", 'ERROR')
            import traceback
            self.log(traceback.format_exc(), 'ERROR')
            return False

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='FPGA Flashing Utility for Gunshot Detection System')
    parser.add_argument('--method', choices=['jtag', 'sd_card', 'auto'],
                       default='auto', help='Programming method')
    parser.add_argument('--config', default='config/flash_config.json',
                       help='Configuration file')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip verification after programming')
    parser.add_argument('--monitor', type=int, default=30,
                       help='Monitor duration in seconds')
    parser.add_argument('--list-devices', action='store_true',
                       help='List detected devices and exit')
    
    args = parser.parse_args()
    
    # Create flasher
    flasher = FPGAFlasher(args.config)
    
    if args.list_devices:
        devices = flasher.detect_hardware()
        print("\nDetected Devices:")
        print(json.dumps(devices, indent=2))
        return
    
    # Determine method
    method = args.method
    if method == 'auto':
        devices = flasher.detect_hardware()
        if devices['jtag']:
            method = 'jtag'
        elif devices['sd_card']:
            method = 'sd_card'
        else:
            print("Error: No programming method available")
            sys.exit(1)
    
    # Run flashing
    success = flasher.run(method=method, verify=not args.no_verify)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

# Hardware Platform Export Script
# For Vitis Software Development

# Open the project
open_project gunshot_detection_fpga/gunshot_detection_fpga.xpr

puts "========================================"
puts "Exporting Hardware Platform"
puts "========================================"
puts "Project: gunshot_detection_fpga"
puts "Platform: Spartan Edge-7 Accelerator"
puts "========================================"

# Make sure implementation is complete
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation not complete. Run implementation first."
    return 1
}

# Open implemented design
open_run impl_1

# Create hardware platform
set platform_name "gunshot_detection_platform"
set xsa_file "${platform_name}.xsa"

# Write hardware platform file
write_hw_platform \
    -fixed \
    -include_bit \
    -force \
    -file $xsa_file

puts "Generated hardware platform: $xsa_file"

# Generate hardware definition files
write_hwdef -force -file ${platform_name}.hwdef
puts "Generated hardware definition: ${platform_name}.hwdef"

# Generate block design TCL for recreation
write_bd_tcl -force ${platform_name}_bd.tcl
puts "Generated block design TCL: ${platform_name}_bd.tcl"

# Generate system diagram
write_sysdef \
    -hwdef ${platform_name}.hwdef \
    -bitfile gunshot_detection_system.bit \
    -file ${platform_name}.sysdef

puts "Generated system definition: ${platform_name}.sysdef"

# Export PS7 configuration
set ps7_cell [get_bd_cells -hierarchical -filter {VLNV =~ "*processing_system7*"}]
if {[llength $ps7_cell] > 0} {
    set ps7_cell [lindex $ps7_cell 0]
    write_bd_cell_ps7_tcl -cell $ps7_cell -file ps7_configuration.tcl
    puts "Generated PS7 configuration: ps7_configuration.tcl"
}

# Generate address map
set addr_map_file "address_map.csv"
set addr_file [open $addr_map_file w]
puts $addr_file "Instance,Base Address,Range"

foreach addr_seg [get_bd_addr_segs] {
    set instance [get_property NAME $addr_seg]
    set base_addr [get_property OFFSET $addr_seg]
    set range [get_property RANGE $addr_seg]
    
    # Skip invalid entries
    if {$base_addr != "" && $range != ""} {
        puts $addr_file "$instance,$base_addr,$range"
    }
}
close $addr_file
puts "Generated address map: $addr_map_file"

# Generate interrupt map
set int_map_file "interrupt_map.csv"
set int_file [open $int_map_file w]
puts $int_file "Interrupt Source,IRQ Number"

# Get interrupt connections
set intc_cell [get_bd_cells -hierarchical -filter {VLNV =~ "*axi_intc*"}]
if {[llength $intc_cell] > 0} {
    set intc_cell [lindex $intc_cell 0]
    set intr_pins [get_bd_pins -of_objects $intc_cell -filter {TYPE == intr}]
    
    foreach pin $intr_pins {
        set pin_name [get_property NAME $pin]
        set irq_num [get_property CONFIG.C_IRQ_CONNECTION $pin]
        puts $int_file "$pin_name,$irq_num"
    }
}
close $int_file
puts "Generated interrupt map: $int_map_file"

# Generate clock configuration
set clock_file "clock_configuration.txt"
set clk_fh [open $clock_file w]

puts $clk_fh "Clock Configuration"
puts $clk_fh "=================="

set clocks [get_clocks]
foreach clock $clocks {
    set clk_name [get_property NAME $clock]
    set period [get_property PERIOD $clock]
    set freq [expr {1000.0 / $period}]
    
    puts $clk_fh "Clock: $clk_name"
    puts $clk_fh "  Period: ${period}ns"
    puts $clk_fh "  Frequency: ${freq}MHz"
    puts $clk_fh ""
    
    # Get clock sources
    set sources [get_clocks -of_objects [get_nets -of_objects [get_pins -of_objects [get_clocks $clk_name]]]]
    if {[llength $sources] > 0} {
        puts $clk_fh "  Sources:"
        foreach source $sources {
            puts $clk_fh "    - [get_property NAME $source]"
        }
        puts $clk_fh ""
    }
}
close $clk_fh
puts "Generated clock configuration: $clock_file"

# Generate memory map for software
set mem_map_file "memory_map.h"
set mem_fh [open $mem_map_file w]

puts $mem_fh "#ifndef MEMORY_MAP_H"
puts $mem_fh "#define MEMORY_MAP_H"
puts $mem_fh ""
puts $mem_fh "// Memory Map for Gunshot Detection System"
puts $mem_fh "// Auto-generated from Vivado"
puts $mem_fh ""
puts $mem_fh "// Base addresses"
puts $mem_fh "#define BASE_DDR         0x00100000"
puts $mem_fh ""

# Peripheral addresses
foreach addr_seg [get_bd_addr_segs -filter {NAME =~ "*Reg*"}] {
    set instance [get_property NAME $addr_seg]
    set base_addr [get_property OFFSET $addr_seg]
    set range [get_property RANGE $addr_seg]
    
    if {$base_addr != "" && $range != ""} {
        # Convert to macro-friendly name
        regsub -all {/} $instance "_" macro_name
        regsub -all {\\} $macro_name "_" macro_name
        regsub -all {\]} $macro_name "" macro_name
        regsub -all {\[} $macro_name "" macro_name
        set macro_name [string toupper $macro_name]
        
        puts $mem_fh "#define BASE_${macro_name} 0x[string range $base_addr 2 end]"
    }
}

puts $mem_fh ""
puts $mem_fh "// DMA buffer addresses"
puts $mem_fh "#define DMA_BUFFER_ADDR   0x01000000"
puts $mem_fh "#define DMA_BUFFER_SIZE   0x00100000  // 1MB"
puts $mem_fh ""
puts $mem_fh "// Audio buffer configuration"
puts $mem_fh "#define AUDIO_BUFFER_SIZE 0x00020000  // 128KB"
puts $mem_fh "#define MAX_SAMPLES       32768       // 2 seconds @ 16kHz"
puts $mem_fh ""
puts $mem_fh "#endif // MEMORY_MAP_H"
close $mem_fh
puts "Generated memory map header: $mem_map_file"

# Generate device tree overlay (for Linux if used)
set dts_file "gunshot_detection.dts"
set dts_fh [open $dts_fh w]

puts $dts_fh "/dts-v1/;"
puts $dts_fh "/plugin/;"
puts $dts_fh ""
puts $dts_fh "/ {"
puts $dts_fh "    fragment@0 {"
puts $dts_fh "        target = <&amba>;"
puts $dts_fh "        __overlay__ {"
puts $dts_fh "            gunshot_detector_0: gunshot_detector@43C10000 {"
puts $dts_fh "                compatible = \"xlnx,gunshot-detector-1.0\";"
puts $dts_fh "                reg = <0x43C10000 0x10000>;"
puts $dts_fh "                interrupt-parent = <&intc>;"
puts $dts_fh "                interrupts = <0 29 4>;"
puts $dts_fh "                clocks = <&clkc 15>;"
puts $dts_fh "                clock-names = \"ap_clk\";"
puts $dts_fh "            };"
puts $dts_fh "        };"
puts $dts_fh "    };"
puts $dts_fh "};"
close $dts_fh
puts "Generated device tree overlay: $dts_file"

# Create Vitis platform directory structure
file mkdir vitis_platform
file mkdir vitis_platform/hw
file mkdir vitis_platform/sw
file mkdir vitis_platform/boot

# Copy necessary files
file copy -force $xsa_file vitis_platform/hw/
file copy -force gunshot_detection_system.bit vitis_platform/hw/
file copy -force boot.bin vitis_platform/boot/
file copy -force $mem_map_file vitis_platform/sw/
file copy -force $addr_map_file vitis_platform/sw/
file copy -force $int_map_file vitis_platform/sw/

# Create platform README
set readme_file "vitis_platform/README.md"
set readme_fh [open $readme_file w]

puts $readme_fh "# Gunshot Detection Platform"
puts $readme_fh ""
puts $readme_fh "## Hardware Platform for Spartan Edge-7"
puts $readme_fh ""
puts $readme_fh "### Files:"
puts $readme_fh "- `hw/gunshot_detection_platform.xsa`: Hardware platform file"
puts $readme_fh "- `hw/gunshot_detection_system.bit`: FPGA bitstream"
puts $readme_fh "- `boot/boot.bin`: Boot image for SD card"
puts $readme_fh "- `sw/memory_map.h`: Memory map for software"
puts $readme_fh "- `sw/address_map.csv`: Address map"
puts $readme_fh "- `sw/interrupt_map.csv`: Interrupt map"
puts $readme_fh ""
puts $readme_fh "### Usage:"
puts $readme_fh "1. Create Vitis workspace"
puts $readme_fh "2. Create platform project from XSA file"
puts $readme_fh "3. Create application project"
puts $readme_fh "4. Build and run"
puts $readme_fh ""
puts $readme_fh "### Hardware Features:"
puts $readme_fh "- 100MHz system clock"
puts $readme_fh "- 3-channel SPI ADC interface"
puts $readme_fh "- 6-channel I2S microphone interface"
puts $readme_fh "- UART debug interface"
puts $readme_fh "- LED indicators"
puts $readme_fh "- Button inputs"
puts $readme_fh ""
puts $readme_fh "### Software Requirements:"
puts $readme_fh "- Vitis 2022.2 or later"
puts $readme_fh "- Xilinx device drivers"
puts $readme_fh "- ARM GCC toolchain"
close $readme_fh

puts "Created Vitis platform structure"

# Generate platform creation script for Vitis
set vitis_script "create_vitis_platform.tcl"
set vitis_fh [open $vitis_script w]

puts $vitis_fh "# Vitis Platform Creation Script"
puts $vitis_fh ""
puts $vitis_fh "# Set workspace"
puts $vitis_fh "setws ./vitis_workspace"
puts $vitis_fh ""
puts $vitis_fh "# Create platform from XSA"
puts $vitis_fh "platform create -name gunshot_detection_platform \\"
puts $vitis_fh "    -hw ./vitis_platform/hw/gunshot_detection_platform.xsa \\"
puts $vitis_fh "    -proc ps7_cortexa9_0 \\"
puts $vitis_fh "    -os standalone"
puts $vitis_fh ""
puts $vitis_fh "# Generate platform"
puts $vitis_fh "platform generate"
puts $vitis_fh ""
puts $vitis_fh "# Create application project"
puts $vitis_fh "app create -name gunshot_detection_app -platform gunshot_detection_platform -domain standalone_domain -template \"Empty Application\""
puts $vitis_fh ""
puts $vitis_fh "# Add source files"
puts $vitis_fh "importsources -name gunshot_detection_app -path ../firmware/src"
puts $vitis_fh ""
puts $vitis_fh "# Build application"
puts $vitis_fh "app build -name gunshot_detection_app"
puts $vitis_fh ""
puts $vitis_fh "puts \"Platform and application created successfully!\""
close $vitis_fh

exec chmod +x $vitis_script
puts "Generated Vitis platform script: $vitis_script"

puts "========================================"
puts "Hardware Platform Export Complete!"
puts "========================================"
puts "Generated files:"
puts "  - $xsa_file (Hardware platform)"
puts "  - Vitis platform directory structure"
puts "  - Memory map and configuration files"
puts "  - Vitis creation script"
puts "========================================"
puts "Next steps:"
puts "1. Open Vitis IDE"
puts "2. Run: source create_vitis_platform.tcl"
puts "3. Build software application"
puts "4. Program FPGA and run"
puts "========================================"

# Save and close project
save_project
close_project

puts "Project saved and closed"

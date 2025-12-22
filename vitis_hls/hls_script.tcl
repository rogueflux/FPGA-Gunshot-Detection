# HLS Script for Gunshot Detector Synthesis
# Target: Spartan-7 FPGA

# Set project name and part
set project_name "gunshot_detector"
set part "xc7s50csga324-1"
set clock_period 10  ;# 100 MHz clock
set solution_name "solution1"

# Create project
open_project -reset $project_name
set_top gunshot_detector_hls_complete

# Add source files
add_files "gunshot_detector_hls_complete.cpp"
add_files "gunshot_detector_hls_complete.h"
add_files "gunshot_weights.h"

# Add testbench
add_files -tb "gunshot_testbench.cpp"

# Set part and create solution
open_solution -reset $solution_name
set_part $part
create_clock -period $clock_period -name default

# Set optimization directives
config_compile -pipeline_loops 0
config_interface -m_axi_addr64
config_interface -register_io scalar_all

# Set array partitioning for performance
set_directive_array_partition -type cyclic -factor 4 -dim 1 "conv1d_layer" weights
set_directive_array_partition -type complete -dim 1 "conv1d_layer" bias
set_directive_array_partition -type cyclic -factor 8 -dim 1 "conv1d_layer" input

set_directive_unroll -factor 4 "conv1d_layer/CONV_OUTER"
set_directive_pipeline -II 1 "conv1d_layer/CONV_INNER"

set_directive_dataflow "gunshot_detector_hls_complete"
set_directive_interface -mode ap_ctrl_none "gunshot_detector_hls_complete"

# AXI Stream interfaces
set_directive_interface -mode axis -register -register_mode both "gunshot_detector_hls_complete" audio_in
set_directive_interface -mode axis -register -register_mode both "gunshot_detector_hls_complete" detection_out

# AXI Lite control interfaces
set_directive_interface -mode s_axilite -bundle ctrl "gunshot_detector_hls_complete" start
set_directive_interface -mode s_axilite -bundle ctrl "gunshot_detector_hls_complete" reset
set_directive_interface -mode s_axilite -bundle ctrl "gunshot_detector_hls_complete" ready
set_directive_interface -mode s_axilite -bundle ctrl "gunshot_detector_hls_complete" sample_count
set_directive_interface -mode s_axilite -bundle ctrl "gunshot_detector_hls_complete" return

# Set resource limits (for Spartan-7)
config_schedule -effort high -relax_ii_for_timing
config_bind -effort high
config_compile -name_max_length 256

# Synthesis options
csynth_design

# Export IP
export_design -format ip_catalog \
    -description "Real-time Gunshot Detection IP" \
    -vendor "gunshot_detection" \
    -library "hls" \
    -version "1.0" \
    -display_name "GunshotDetector"

# Generate reports
report_utilization
report_timing

# Run C simulation
if {[file exists "gunshot_testbench.cpp"]} {
    csim_design -clean
}

# Run Co-simulation (if RTL files exist)
if {[file exists "gunshot_detector_hls_complete_prj"]} {
    cosim_design -setup -rtl vhdl -tool modelsim
}

# Close project
close_project

puts "========================================="
puts "HLS Synthesis Complete"
puts "========================================="
puts "Project: $project_name"
puts "Part: $part"
puts "Clock: ${clock_period}ns (${expr{1000.0/$clock_period}} MHz)"
puts "Solution: $solution_name"
puts "========================================="
puts "Generated IP is ready for Vivado integration"
puts "========================================="

# Create Vivado export script
set export_script [open "export_to_vivado.tcl" w]
puts $export_script "#!/bin/tclsh"
puts $export_script "# Script to import HLS IP into Vivado"
puts $export_script ""
puts $export_script "set ip_repo_path \[file normalize \$project_name/$solution_name/impl/ip\]"
puts $export_script "set_property ip_repo_paths \$ip_repo_path \[current_project\]"
puts $export_script "update_ip_catalog"
puts $export_script ""
puts $export_script "# Add IP to block design"
puts $export_script "create_bd_cell -type ip -vlnv gunshot_detection:hls:gunshot_detector_hls_complete:1.0 gunshot_detector_0"
puts $export_script ""
puts $export_script "# Connect clock and reset"
puts $export_script "connect_bd_net \[get_bd_pins gunshot_detector_0/ap_clk\] \[get_bd_pins processing_system7_0/FCLK_CLK0\]"
puts $export_script "connect_bd_net \[get_bd_pins gunshot_detector_0/ap_rst_n\] \[get_bd_pins rst_ps7_0_100M/peripheral_aresetn\]"
puts $export_script ""
puts $export_script "# Connect AXI Stream interfaces"
puts $export_script "# audio_in: Connect to DMA or I2S receiver"
puts $export_script "# detection_out: Connect to interrupt controller or GPIO"
puts $export_script ""
puts $export_script "# Connect AXI Lite control"
puts $export_script "apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master \"/processing_system7_0/M_AXI_GP0\" Clk \"Auto\" } \[get_bd_intf_pins gunshot_detector_0/s_axi_ctrl\]"
close $export_script

puts "Export script created: export_to_vivado.tcl"

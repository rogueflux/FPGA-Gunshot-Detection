# Vivado Project Creation Script
# Gunshot Detection System for Spartan Edge-7

# Set project properties
set project_name "gunshot_detection_fpga"
set board_part "em.avnet.com:zedboard:part0:1.4"
set device_part "xc7z020clg484-1"
set design_name "gunshot_detection_system"

# Create project
create_project $project_name ./$project_name -part $device_part -force

# Set project properties
set_property board_part $board_part [current_project]
set_property target_language VHDL [current_project]
set_property simulator_language Mixed [current_project]
set_property default_lib work [current_project]

puts "Created project: $project_name"
puts "Target device: $device_part"
puts "Board: $board_part"

# Create 'sources_1' fileset
if {[string equal [get_filesets -quiet sources_1] ""]} {
    create_fileset -srcset sources_1
}

# Create 'constrs_1' fileset
if {[string equal [get_filesets -quiet constrs_1] ""]} {
    create_fileset -constrset constrs_1
}

# Create 'sim_1' fileset
if {[string equal [get_filesets -quiet sim_1] ""]} {
    create_fileset -simset sim_1
}

# Add constraint files
add_files -fileset constrs_1 -norecurse ../vivado/spartan_edge_accel_gunshot.xdc

puts "Added constraint file"

# Set constraint file properties
set file_obj [get_files -of_objects [get_filesets constrs_1] [list "*spartan_edge_accel_gunshot.xdc"]]
set_property -name "file_type" -value "XDC" -objects $file_obj

# Create block design
source ../vivado/block_design.tcl

# Generate the wrapper
make_wrapper -files [get_files ./$project_name/$project_name.srcs/sources_1/bd/$design_name/$design_name.bd] -top
add_files -norecurse ./$project_name/$project_name.srcs/sources_1/bd/$design_name/hdl/${design_name}_wrapper.vhd

# Update compile order
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

puts "Generated block design wrapper"

# Add HLS IP repository
set_property ip_repo_paths {
    ../vitis_hls/gunshot_detector_hls_complete/solution1/impl/ip
    ../vitis_hls/spi_adc_interface/solution1/impl/ip
} [current_project]
update_ip_catalog

puts "Added HLS IP repository"

# Create IP cores (if needed)
# Uncomment to create custom IP cores
# create_ip -name gunshot_detector_hls_complete -vendor xilinx.com -library hls -version 1.0 -module_name gunshot_detector_0
# create_ip -name spi_adc_interface -vendor xilinx.com -library hls -version 1.0 -module_name spi_adc_0

# Set synthesis strategy
set_property strategy "Vivado Synthesis Defaults" [get_runs synth_1]
set_property steps.synth_design.args.more_options "-verbose -mode out_of_context" [get_runs synth_1]

# Set implementation strategy
set_property strategy "Vivado Implementation Defaults" [get_runs impl_1]
set_property steps.opt_design.is_enabled true [get_runs impl_1]
set_property steps.opt_design.args.more_options "-verbose" [get_runs impl_1]
set_property steps.place_design.args.more_options "-verbose" [get_runs impl_1]
set_property steps.route_design.args.more_options "-verbose" [get_runs impl_1]
set_property steps.write_bitstream.args.bin_file true [get_runs impl_1]

# Configure report generation
set_property steps.synth_design.args.more_options "-report_strategy {NoTimingReports} -verbose" [get_runs synth_1]
set_property steps.opt_design.args.more_options "-verbose" [get_runs impl_1]
set_property steps.place_design.args.more_options "-verbose" [get_runs impl_1]
set_property steps.route_design.args.more_options "-verbose" [get_runs impl_1]

# Set project settings for Spartan-7 optimization
set_property generic { \
    CLOCK_PERIOD=10 \
    SAMPLE_RATE=16000 \
    NUM_CHANNELS=3 \
    USE_DSP48=1 \
    OPTIMIZE_FOR_SPEED=1 \
} [current_fileset]

puts "Set project optimization settings"

# Add source files from other directories
# Add Vitis HLS generated files
add_files -norecurse ../vitis_hls/gunshot_detector_hls_complete/solution1/impl/ip/hdl/vhdl/
add_files -norecurse ../vitis_hls/spi_adc_interface/solution1/impl/ip/hdl/vhdl/

# Add firmware source files
add_files -norecurse ../firmware/src/

puts "Added source files from HLS and firmware directories"

# Set top module
set_property top ${design_name}_wrapper [current_fileset]

# Save project
save_project

puts "========================================"
puts "Vivado Project Creation Complete!"
puts "========================================"
puts "Project Name: $project_name"
puts "Design Name: $design_name"
puts "Device: $device_part"
puts "Board: $board_part"
puts "========================================"
puts "Next steps:"
puts "1. Run synthesis: launch_runs synth_1"
puts "2. Run implementation: launch_runs impl_1"
puts "3. Generate bitstream: launch_runs impl_1 -to_step write_bitstream"
puts "4. Export hardware: write_hw_platform -fixed -include_bit -force -file ./${project_name}.xsa"
puts "========================================"

# Optional: Create a run script
set run_script [open "./run_project.tcl" w]
puts $run_script "#!/bin/tclsh"
puts $run_script "# Run script for $project_name"
puts $run_script ""
puts $run_script "open_project $project_name/$project_name.xpr"
puts $run_script ""
puts $run_script "# Reset runs"
puts $run_script "reset_run synth_1"
puts $run_script "reset_run impl_1"
puts $run_script ""
puts $run_script "# Run synthesis"
puts $run_script "launch_runs synth_1"
puts $run_script "wait_on_run synth_1"
puts $run_script ""
puts $run_script "# Run implementation"
puts $run_script "launch_runs impl_1 -to_step write_bitstream"
puts $run_script "wait_on_run impl_1"
puts $run_script ""
puts $run_script "# Generate reports"
puts $run_script "open_run impl_1"
puts $run_script "report_timing_summary -file timing_report.txt"
puts $run_script "report_utilization -file utilization_report.txt"
puts $run_script "report_power -file power_report.txt"
puts $run_script ""
puts $run_script "# Export hardware platform"
puts $run_script "write_hw_platform -fixed -include_bit -force -file ./${project_name}.xsa"
puts $run_script ""
puts $run_script "puts \"Project build complete!\""
close $run_script

exec chmod +x ./run_project.tcl

puts "Created run script: ./run_project.tcl"
puts "To build the project, run: vivado -mode batch -source run_project.tcl"

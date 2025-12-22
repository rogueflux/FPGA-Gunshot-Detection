# Implementation Script for Gunshot Detection System

# Open the project
open_project gunshot_detection_fpga/gunshot_detection_fpga.xpr

puts "========================================"
puts "Starting Implementation"
puts "========================================"
puts "Project: gunshot_detection_fpga"
puts "Target: Spartan-7 FPGA"
puts "========================================"

# Reset implementation run
reset_run impl_1
puts "Reset implementation run"

# Set implementation strategies
set_property strategy Performance_Explore [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE Explore [get_runs impl_1]

puts "Set implementation strategy: Performance_Explore"

# Set timing constraints
create_clock -name clk_100m -period 10.000 [get_ports clk_100m]
create_clock -name i2s_mclk -period 81.380 [get_ports i2s_mclk]

# Set false paths
set_false_path -from [get_ports {btn_user btn_reset}]
set_false_path -to [get_ports {led_status led_alert led_system}]

puts "Set timing constraints"

# Run synthesis if not already done
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "Running synthesis..."
    launch_runs synth_1
    wait_on_run synth_1
}

# Check synthesis results
if {[get_property STATUS [get_runs synth_1]] != "synth_design Complete!"} {
    puts "ERROR: Synthesis failed!"
    return 1
}

puts "Synthesis completed successfully"

# Generate utilization report after synthesis
open_run synth_1 -name synth_1
report_utilization -file synthesis_utilization_report.txt -hierarchical
report_timing_summary -file synthesis_timing_report.txt

puts "Generated synthesis reports"

# Run implementation
puts "Starting implementation..."
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

# Check implementation results
if {[get_property STATUS [get_runs impl_1]] != "write_bitstream Complete!"} {
    puts "ERROR: Implementation failed!"
    
    # Try to get error details
    open_run impl_1
    report_timing_summary -failfast -file timing_failure_report.txt
    report_utilization -file utilization_failure_report.txt
    
    return 1
}

puts "Implementation completed successfully"

# Open implemented design
open_run impl_1

# Generate implementation reports
puts "Generating implementation reports..."

report_timing_summary -file timing_summary_report.txt
report_timing -sort_by group -max_paths 100 -path_type summary -file detailed_timing_report.txt
report_utilization -file utilization_report.txt -hierarchical
report_power -file power_analysis_report.txt
report_clock_networks -file clock_network_report.txt
report_drc -file drc_report.txt
report_methodology -file methodology_report.txt

# Generate resource utilization by hierarchy
report_utilization -hierarchical -hierarchical_depth 4 -file hierarchical_utilization_report.txt

# Generate post-route timing analysis
report_timing -max_paths 50 -delay_type min_max -sort_by group -input_pins -routable_nets -file post_route_timing_report.txt

# Generate area optimization suggestions
report_utilization -file area_optimization_suggestions.txt

puts "Generated all reports"

# Check timing closure
set timing_paths [get_timing_paths -max_paths 1]
if {[llength $timing_paths] > 0} {
    set slack [get_property SLACK [lindex $timing_paths 0]]
    if {$slack < 0} {
        puts "WARNING: Timing violation detected! Slack = ${slack}ns"
        
        # Try timing optimization
        puts "Attempting timing optimization..."
        phys_opt_design -directive Explore
        route_design
        
        # Re-check timing
        set timing_paths [get_timing_paths -max_paths 1]
        set slack [get_property SLACK [lindex $timing_paths 0]]
        puts "After optimization: Slack = ${slack}ns"
    } else {
        puts "Timing closure achieved! Slack = ${slack}ns"
    }
}

# Check resource utilization
set lut_usage [get_property LUT [get_utilization]]
set ff_usage [get_property FF [get_utilization]]
set dsp_usage [get_property DSP [get_utilization]]
set bram_usage [get_property BRAM [get_utilization]]

puts "========================================"
puts "Resource Utilization Summary"
puts "========================================"
puts "LUTs: $lut_usage"
puts "Flip-Flops: $ff_usage"
puts "DSPs: $dsp_usage"
puts "BRAMs: $bram_usage"
puts "========================================"

# Check for overutilization
set max_luts 32600  # Spartan-7 S50 capacity
set max_dsps 120
set max_brams 120

if {$lut_usage > $max_luts * 0.8} {
    puts "WARNING: LUT utilization > 80%"
}
if {$dsp_usage > $max_dsps * 0.8} {
    puts "WARNING: DSP utilization > 80%"
}
if {$bram_usage > $max_brams * 0.8} {
    puts "WARNING: BRAM utilization > 80%"
}

# Generate design checkpoint
write_checkpoint -force implemented_design.dcp
puts "Saved design checkpoint: implemented_design.dcp"

# Export netlist for simulation
write_verilog -force -mode funcsim implemented_design_netlist.v
write_vhdl -force -mode funcsim implemented_design_netlist.vhd
puts "Exported simulation netlists"

# Generate bitstream properties
set_property BITSTREAM.CONFIG.CONFIGRATE 33 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS true [current_design]

puts "Set bitstream properties"

# Create programming files
write_bitstream -force gunshot_detection_system.bit
write_cfgmem -force -format bin -interface spix4 -size 16 -loadbit "up 0x0 gunshot_detection_system.bit" boot.bin

puts "========================================"
puts "Implementation Complete!"
puts "========================================"
puts "Generated files:"
puts "  - gunshot_detection_system.bit (Bitstream)"
puts "  - boot.bin (Boot image for SD card)"
puts "  - implemented_design.dcp (Design checkpoint)"
puts "  - Multiple report files (*_report.txt)"
puts "========================================"
puts "Reports generated:"
puts "  - Timing Summary"
puts "  - Utilization"
puts "  - Power Analysis"
puts "  - DRC"
puts "  - Methodology"
puts "========================================"

# Save project
save_project

puts "Project saved successfully"
puts "To program the FPGA:"
puts "1. Copy boot.bin to SD card"
puts "2. Insert SD card into Spartan Edge board"
puts "3. Set boot mode to SD card"
puts "4. Power cycle the board"
puts "========================================"

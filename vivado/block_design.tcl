# Create block design for Spartan Edge Accelerator

create_bd_design "gunshot_detection"

# Add Zynq Processing System
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" }  [get_bd_cells ps7]

# Add AXI DMA for audio data
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_audio
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {23}] [get_bd_cells axi_dma_audio]

# Add HLS IP core
create_bd_cell -type ip -vlnv xilinx.com:hls:gunshot_detector_hls_complete:1.0 gunshot_detector_ip

# Add FFT IP for feature extraction
create_bd_cell -type ip -vlnv xilinx.com:ip:xfft:9.1 xfft_0
set_property -dict [list CONFIG.implementation_options {pipelined_streaming_io} CONFIG.transform_length {256} CONFIG.target_clock_frequency {100} CONFIG.number_of_stages_using_block_ram_for_data_and_phase_factors {5}] [get_bd_cells xfft_0]

# Add I2S interface
create_bd_cell -type module -reference i2s_receiver i2s_rx_0

# Connect clocks
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_dma_audio/s_axi_lite_aclk]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins gunshot_detector_ip/ap_clk]

# Connect interrupts
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 irq_concat
connect_bd_net [get_bd_pins axi_dma_audio/mm2s_introut] [get_bd_pins irq_concat/In0]
connect_bd_net [get_bd_pins irq_concat/dout] [get_bd_pins ps7/IRQ_F2P]

# Connect AXI interfaces
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Master "/ps7/M_AXI_GP0" Clk "Auto" }  [get_bd_intf_pins axi_dma_audio/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config {Slave "/ps7/S_AXI_HP0" Clk "Auto" }  [get_bd_intf_pins axi_dma_audio/M_AXI_MM2S]

# Connect data path
connect_bd_intf_net [get_bd_intf_pins axi_dma_audio/M_AXIS_MM2S] [get_bd_intf_pins gunshot_detector_ip/audio_features]
connect_bd_net [get_bd_pins gunshot_detector_ip/detection] [get_bd_pins xlslice_0/Din]
connect_bd_net [get_bd_pins gunshot_detector_ip/confidence] [get_bd_pins xlslice_1/Din]

# Add GPIO for LEDs
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 gpio_leds
set_property -dict [list CONFIG.C_GPIO_WIDTH {2} CONFIG.C_ALL_OUTPUTS {1}] [get_bd_cells gpio_leds]

# Validate and save design
validate_bd_design
save_bd_design

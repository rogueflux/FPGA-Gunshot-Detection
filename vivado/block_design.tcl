# Vivado Block Design for Gunshot Detection System
# Spartan Edge-7 Accelerator Board

# Create block design
create_bd_design "gunshot_detection_system"
update_compile_order -fileset sources_1

puts "Creating block design for gunshot detection system..."

# ============================================================================
# 1. Add Zynq Processing System (PS7 for Spartan-7)
# ============================================================================
puts "Adding Zynq Processing System..."

create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7_0

# Apply board preset for Spartan Edge Accelerator
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP1 {1} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_TTC0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_GRP_CD_ENABLE {1} \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART1_UART1_IO {MIO 48 .. 49} \
    CONFIG.PCW_I2C0_PERIPHERAL_ENABLE {0} \
    CONFIG.PCW_GPIO_MIO_GPIO_ENABLE {1} \
    CONFIG.PCW_GPIO_MIO_GPIO_IO {MIO} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_DEFAULT_ACP_USER_VAL {1} \
    CONFIG.PCW_ACT_DCI_PERIPHERAL_FREQMHZ {10.158730} \
    CONFIG.PCW_ACT_QSPI_PERIPHERAL_FREQMHZ {200.000000} \
    CONFIG.PCW_ACT_ENET0_PERIPHERAL_FREQMHZ {125.000000} \
    CONFIG.PCW_ACT_SDIO_PERIPHERAL_FREQMHZ {50.000000} \
    CONFIG.PCW_ACT_UART_PERIPHERAL_FREQMHZ {100.000000} \
    CONFIG.PCW_ACT_WDT_PERIPHERAL_FREQMHZ {108.333336} \
    CONFIG.PCW_ACT_PCAP_PERIPHERAL_FREQMHZ {200.000000} \
    CONFIG.PCW_ACT_TPIU_PERIPHERAL_FREQMHZ {200.000000} \
    CONFIG.PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
    CONFIG.PCW_ACT_FPGA1_PERIPHERAL_FREQMHZ {10.000000} \
    CONFIG.PCW_ACT_FPGA2_PERIPHERAL_FREQMHZ {10.000000} \
    CONFIG.PCW_ACT_FPGA3_PERIPHERAL_FREQMHZ {10.000000} \
    CONFIG.PCW_ACT_TTC0_CLK0_PERIPHERAL_FREQMHZ {108.333336} \
    CONFIG.PCW_ACT_TTC0_CLK1_PERIPHERAL_FREQMHZ {108.333336} \
    CONFIG.PCW_ACT_TTC0_CLK2_PERIPHERAL_FREQMHZ {108.333336} \
] [get_bd_cells ps7_0]

# ============================================================================
# 2. Add AXI DMA for Audio Data Transfer
# ============================================================================
puts "Adding AXI DMA for audio data..."

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_audio
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {23} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_mm2s_burst_size {256} \
    CONFIG.c_s2mm_burst_size {256} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_include_s2mm_dre {1} \
] [get_bd_cells axi_dma_audio]

# ============================================================================
# 3. Add Gunshot Detector HLS IP Core
# ============================================================================
puts "Adding Gunshot Detector HLS IP..."

create_bd_cell -type ip -vlnv xilinx.com:hls:gunshot_detector_hls_complete:1.0 gunshot_detector_0

# ============================================================================
# 4. Add SPI Controller for ADC Interface
# ============================================================================
puts "Adding SPI Controller for ADC..."

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_quad_spi:3.2 axi_quad_spi_adc
set_property -dict [list \
    CONFIG.C_USE_STARTUP {0} \
    CONFIG.C_NUM_SS_BITS {3} \
    CONFIG.C_SCK_RATIO {8} \
    CONFIG.C_FIFO_DEPTH {256} \
    CONFIG.C_NUM_TRANSFER_BITS {8} \
    CONFIG.C_TYPE_OF_AXI4_INTERFACE {0} \
    CONFIG.C_UC_FAMILY {0} \
] [get_bd_cells axi_quad_spi_adc]

# ============================================================================
# 5. Add I2S Receiver (Optional for MEMS microphones)
# ============================================================================
puts "Adding I2S Receiver..."

create_bd_cell -type ip -vlnv xilinx.com:user:i2s_receiver:1.0 i2s_receiver_0
set_property -dict [list \
    CONFIG.NUM_CHANNELS {6} \
    CONFIG.DATA_WIDTH {24} \
    CONFIG.AXI_DATA_WIDTH {32} \
] [get_bd_cells i2s_receiver_0]

# ============================================================================
# 6. Add FFT IP for Feature Extraction
# ============================================================================
puts "Adding FFT IP for feature extraction..."

create_bd_cell -type ip -vlnv xilinx.com:ip:xfft:9.1 xfft_0
set_property -dict [list \
    CONFIG.implementation_options {pipelined_streaming_io} \
    CONFIG.transform_length {256} \
    CONFIG.target_clock_frequency {100} \
    CONFIG.throttle_scheme {realtime} \
    CONFIG.complex_mult_type {use_mults_resources} \
    CONFIG.butterfly_type {use_luts} \
    CONFIG.data_format {fixed_point} \
    CONFIG.phase_factor_width {16} \
    CONFIG.output_ordering {natural_order} \
    CONFIG.aresetn {true} \
    CONFIG.rounding_modes {convergent_rounding} \
    CONFIG.scaling_options {scaled} \
    CONFIG.accum_width {33} \
] [get_bd_cells xfft_0]

# ============================================================================
# 7. Add Clocking Wizard
# ============================================================================
puts "Adding Clocking Wizard..."

create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {100.000} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {12.288} \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {50.000} \
    CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {200.000} \
    CONFIG.USE_LOCKED {false} \
    CONFIG.USE_RESET {false} \
    CONFIG.CLKIN1_JITTER_PS {100.0} \
    CONFIG.MMCM_DIVCLK_DIVIDE {1} \
    CONFIG.MMCM_CLKIN1_PERIOD {10.0} \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F {81.375} \
    CONFIG.MMCM_CLKOUT1_DIVIDE {20} \
    CONFIG.MMCM_CLKOUT2_DIVIDE {5} \
    CONFIG.NUM_OUT_CLKS {3} \
    CONFIG.CLKOUT1_JITTER {312.069} \
    CONFIG.CLKOUT1_PHASE_ERROR {245.713} \
    CONFIG.CLKOUT2_JITTER {242.086} \
    CONFIG.CLKOUT2_PHASE_ERROR {245.713} \
    CONFIG.CLKOUT3_JITTER {173.936} \
    CONFIG.CLKOUT3_PHASE_ERROR {245.713} \
] [get_bd_cells clk_wiz_0]

# ============================================================================
# 8. Add GPIO for LEDs and Buttons
# ============================================================================
puts "Adding GPIO for LEDs..."

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_leds
set_property -dict [list \
    CONFIG.C_GPIO_WIDTH {8} \
    CONFIG.C_ALL_OUTPUTS {1} \
    CONFIG.C_DOUT_DEFAULT {0x00000001} \
] [get_bd_cells axi_gpio_leds]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_buttons
set_property -dict [list \
    CONFIG.C_GPIO_WIDTH {4} \
    CONFIG.C_ALL_INPUTS {1} \
    CONFIG.C_INTERRUPT_PRESENT {1} \
] [get_bd_cells axi_gpio_buttons]

# ============================================================================
# 9. Add Interrupt Controller
# ============================================================================
puts "Adding Interrupt Controller..."

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 axi_intc_0

# ============================================================================
# 10. Add System Reset
# ============================================================================
puts "Adding System Reset..."

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_100M

# ============================================================================
# 11. Add AXI Interconnect
# ============================================================================
puts "Adding AXI Interconnect..."

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_MI {8}] [get_bd_cells axi_interconnect_0]

# ============================================================================
# 12. Add AXI SmartConnect for High Performance
# ============================================================================
puts "Adding AXI SmartConnect..."

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_0]

# ============================================================================
# 13. Connect Clocks
# ============================================================================
puts "Connecting clocks..."

connect_bd_net [get_bd_pins ps7_0/FCLK_CLK0] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins i2s_receiver_0/i2s_mclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_dma_audio/s_axi_lite_aclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins gunshot_detector_0/ap_clk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_quad_spi_adc/ext_spi_clk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins xfft_0/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_gpio_leds/s_axi_aclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_gpio_buttons/s_axi_aclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_intc_0/s_axi_aclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/S00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M00_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M01_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M02_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M03_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M04_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M05_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M06_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins axi_interconnect_0/M07_ACLK]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out2] [get_bd_pins rst_ps7_100M/slowest_sync_clk]

# ============================================================================
# 14. Connect Resets
# ============================================================================
puts "Connecting resets..."

connect_bd_net [get_bd_pins ps7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_100M/ext_reset_in]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_dma_audio/axi_resetn]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins gunshot_detector_0/ap_rst_n]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_quad_spi_adc/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins i2s_receiver_0/axi_resetn]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins xfft_0/aresetn]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_gpio_leds/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_gpio_buttons/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_intc_0/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M00_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M01_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M02_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M03_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M04_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M05_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M06_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins axi_interconnect_0/M07_ARESETN]
connect_bd_net [get_bd_pins rst_ps7_100M/peripheral_aresetn] [get_bd_pins smartconnect_0/aresetn]

# ============================================================================
# 15. Connect AXI Interfaces
# ============================================================================
puts "Connecting AXI interfaces..."

# Connect PS7 to AXI Interconnect
connect_bd_intf_net [get_bd_intf_pins ps7_0/M_AXI_GP0] [get_bd_intf_pins axi_interconnect_0/S00_AXI]

# Connect AXI Interconnect to peripherals
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] [get_bd_intf_pins axi_dma_audio/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M01_AXI] [get_bd_intf_pins gunshot_detector_0/s_axi_ctrl]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M02_AXI] [get_bd_intf_pins axi_quad_spi_adc/AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M03_AXI] [get_bd_intf_pins i2s_receiver_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M04_AXI] [get_bd_intf_pins xfft_0/S_AXI_CONFIG]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M05_AXI] [get_bd_intf_pins axi_gpio_leds/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M06_AXI] [get_bd_intf_pins axi_gpio_buttons/S_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M07_AXI] [get_bd_intf_pins axi_intc_0/s_axi]

# Connect DMA to SmartConnect
connect_bd_intf_net [get_bd_intf_pins axi_dma_audio/M_AXI_MM2S] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_audio/M_AXI_S2MM] [get_bd_intf_pins smartconnect_0/S01_AXI]

# Connect SmartConnect to PS7 HP port
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins ps7_0/S_AXI_HP0]

# ============================================================================
# 16. Connect Data Path
# ============================================================================
puts "Connecting data path..."

# Connect DMA to Gunshot Detector (audio data)
connect_bd_intf_net [get_bd_intf_pins axi_dma_audio/M_AXIS_MM2S] [get_bd_intf_pins gunshot_detector_0/audio_in]

# Connect Gunshot Detector output to DMA (results)
create_bd_cell -type ip -vlnv xilinx.com:ip:axis_data_fifo:2.0 axis_data_fifo_0
set_property -dict [list CONFIG.FIFO_DEPTH {1024}] [get_bd_cells axis_data_fifo_0]

connect_bd_intf_net [get_bd_intf_pins gunshot_detector_0/detection_out] [get_bd_intf_pins axis_data_fifo_0/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins axis_data_fifo_0/M_AXIS] [get_bd_intf_pins axi_dma_audio/S_AXIS_S2MM]

# Connect I2S to DMA (optional audio input)
connect_bd_intf_net [get_bd_intf_pins i2s_receiver_0/M_AXIS] [get_bd_intf_pins axi_dma_audio/S_AXIS_S2MM]

# ============================================================================
# 17. Connect Interrupts
# ============================================================================
puts "Connecting interrupts..."

create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0
set_property -dict [list CONFIG.NUM_PORTS {5}] [get_bd_cells xlconcat_0]

connect_bd_net [get_bd_pins axi_dma_audio/mm2s_introut] [get_bd_pins xlconcat_0/In0]
connect_bd_net [get_bd_pins axi_dma_audio/s2mm_introut] [get_bd_pins xlconcat_0/In1]
connect_bd_net [get_bd_pins axi_quad_spi_adc/ip2intc_irpt] [get_bd_pins xlconcat_0/In2]
connect_bd_net [get_bd_pins axi_gpio_buttons/ip2intc_irpt] [get_bd_pins xlconcat_0/In3]
connect_bd_net [get_bd_pins axi_intc_0/irq] [get_bd_pins xlconcat_0/In4]

connect_bd_net [get_bd_pins xlconcat_0/dout] [get_bd_pins ps7_0/IRQ_F2P]

# ============================================================================
# 18. Connect External Interfaces
# ============================================================================
puts "Connecting external interfaces..."

# SPI Interface to external ADC
create_bd_port -dir O spi_sck
create_bd_port -dir O spi_mosi
create_bd_port -dir I spi_miso
create_bd_port -dir O -from 2 -to 0 spi_cs

connect_bd_net [get_bd_ports spi_sck] [get_bd_pins axi_quad_spi_adc/ext_spi_clk]
connect_bd_net [get_bd_ports spi_mosi] [get_bd_pins axi_quad_spi_adc/io0_o]
connect_bd_net [get_bd_ports spi_miso] [get_bd_pins axi_quad_spi_adc/io1_i]
connect_bd_net [get_bd_ports spi_cs] [get_bd_pins axi_quad_spi_adc/ss_o]

# I2S Interface (optional)
create_bd_port -dir O i2s_mclk
create_bd_port -dir O i2s_bclk
create_bd_port -dir O i2s_lrclk
create_bd_port -dir I -from 5 -to 0 i2s_data

connect_bd_net [get_bd_ports i2s_mclk] [get_bd_pins i2s_receiver_0/i2s_mclk]
connect_bd_net [get_bd_ports i2s_bclk] [get_bd_pins i2s_receiver_0/i2s_bclk]
connect_bd_net [get_bd_ports i2s_lrclk] [get_bd_pins i2s_receiver_0/i2s_lrclk]
connect_bd_net [get_bd_ports i2s_data] [get_bd_pins i2s_receiver_0/i2s_data]

# LEDs
create_bd_port -dir O -from 7 -to 0 led_outputs
connect_bd_net [get_bd_ports led_outputs] [get_bd_pins axi_gpio_leds/gpio_io_o]

# Buttons
create_bd_port -dir I -from 3 -to 0 btn_inputs
connect_bd_net [get_bd_ports btn_inputs] [get_bd_pins axi_gpio_buttons/gpio_io_i]

# UART (for debug)
create_bd_port -dir O uart_tx
create_bd_port -dir I uart_rx
connect_bd_net [get_bd_ports uart_tx] [get_bd_pins ps7_0/UART1_TX]
connect_bd_net [get_bd_ports uart_rx] [get_bd_pins ps7_0/UART1_RX]

# ============================================================================
# 19. Configure Address Map
# ============================================================================
puts "Configuring address map..."

assign_bd_address
set_property offset 0x43C00000 [get_bd_addr_segs {ps7_0/Data/SEG_axi_dma_audio_Reg}]
set_property range 64K [get_bd_addr_segs {ps7_0/Data/SEG_axi_dma_audio_Reg}]

set_property offset 0x43C10000 [get_bd_addr_segs {ps7_0/Data/SEG_gunshot_detector_0_Reg}]
set_property range 64K [get_bd_addr_segs {ps7_0/Data/SEG_gunshot_detector_0_Reg}]

set_property offset 0x43C20000 [get_bd_addr_segs {ps7_0/Data/SEG_axi_quad_spi_adc_Reg}]
set_property range 64K [get_bd_addr_segs {ps7_0/Data/SEG_axi_quad_spi_adc_Reg}]

set_property offset 0x43C30000 [get_bd_addr_segs {ps7_0/Data/SEG_i2s_receiver_0_Reg}]
set_property range 64K [get_bd_addr_segs {ps7_0/Data/SEG_i2s_receiver_0_Reg}]

set_property offset 0x43C40000 [get_bd_addr_segs {ps7_0/Data/SEG_xfft_0_Reg}]
set_property range 64K [get_bd_addr_segs {ps7_0/Data/SEG_xfft_0_Reg}]

set_property offset 0x43C50000 [get_bd_addr_segs {ps7_0/Data/SEG_axi_gpio_leds_Reg}]
set_property range 64K [get_bd_addr_segs {ps7_0/Data/SEG_axi_gpio_leds_Reg}]

set_property offset 0x43C60000 [get_bd_addr_segs {ps7_0/Data/SEG_axi_gpio_buttons_Reg}]
set_property range 64K [get_bd_addr_segs {ps7_0/Data/SEG_axi_gpio_buttons_Reg}]

set_property offset 0x43C70000 [get_bd_addr_segs {ps7_0/Data/SEG_axi_intc_0_Reg}]
set_property range 64K [get_bd_addr_segs {ps7_0/Data/SEG_axi_intc_0_Reg}]

# ============================================================================
# 20. Validate and Save Design
# ============================================================================
puts "Validating block design..."

validate_bd_design
save_bd_design

puts "========================================"
puts "Block Design Creation Complete!"
puts "========================================"
puts "Components added:"
puts "  - Zynq Processing System (PS7)"
puts "  - AXI DMA for audio data"
puts "  - Gunshot Detector HLS IP"
puts "  - SPI Controller for ADC"
puts "  - I2S Receiver (optional)"
puts "  - FFT for feature extraction"
puts "  - GPIO for LEDs and buttons"
puts "  - Interrupt controller"
puts "  - Clocking wizard"
puts "  - AXI Interconnect"
puts "  - SmartConnect"
puts "========================================"

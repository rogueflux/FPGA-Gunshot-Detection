# Spartan Edge Accelerator Board Constraints File
# Target: Spartan-7 FPGA (XC7S50CSGA324-1)
# Gunshot Detection System

##############################
# Clock Constraints
##############################

# Primary 100MHz clock from board
create_clock -name clk_100m -period 10.000 [get_ports clk_100m]
set_property PACKAGE_PIN T8 [get_ports clk_100m]
set_property IOSTANDARD LVCMOS33 [get_ports clk_100m]

# I2S Master Clock (12.288 MHz for 48kHz audio)
create_clock -name i2s_mclk -period 81.380 [get_ports i2s_mclk]
set_property PACKAGE_PIN R13 [get_ports i2s_mclk]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_mclk]

##############################
# SPI Interface for ADC (MAX11060)
##############################

# SPI Clock
set_property PACKAGE_PIN P14 [get_ports spi_sck]
set_property IOSTANDARD LVCMOS33 [get_ports spi_sck]
set_property DRIVE 12 [get_ports spi_sck]
set_property SLEW SLOW [get_ports spi_sck]

# SPI MOSI (FPGA -> ADC)
set_property PACKAGE_PIN N15 [get_ports spi_mosi]
set_property IOSTANDARD LVCMOS33 [get_ports spi_mosi]
set_property DRIVE 12 [get_ports spi_mosi]
set_property PULLUP true [get_ports spi_mosi]

# SPI MISO (ADC -> FPGA)
set_property PACKAGE_PIN M15 [get_ports spi_miso]
set_property IOSTANDARD LVCMOS33 [get_ports spi_miso]
set_property PULLUP true [get_ports spi_miso]

# SPI Chip Select
set_property PACKAGE_PIN L14 [get_ports spi_cs]
set_property IOSTANDARD LVCMOS33 [get_ports spi_cs]
set_property DRIVE 12 [get_ports spi_cs]

# SPI Additional Chip Selects (for multiple ADCs)
set_property PACKAGE_PIN K15 [get_ports spi_cs2]
set_property IOSTANDARD LVCMOS33 [get_ports spi_cs2]
set_property DRIVE 12 [get_ports spi_cs2]

set_property PACKAGE_PIN J14 [get_ports spi_cs3]
set_property IOSTANDARD LVCMOS33 [get_ports spi_cs3]
set_property DRIVE 12 [get_ports spi_cs3]

##############################
# I2S Audio Interface (Optional, for MEMS microphones)
##############################

# I2S Master Clock
set_property PACKAGE_PIN R13 [get_ports i2s_mclk]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_mclk]
set_property DRIVE 12 [get_ports i2s_mclk]

# I2S Bit Clock
set_property PACKAGE_PIN T13 [get_ports i2s_bclk]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_bclk]
set_property DRIVE 12 [get_ports i2s_bclk]

# I2S Word Clock (Left/Right)
set_property PACKAGE_PIN T12 [get_ports i2s_lrclk]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_lrclk]
set_property DRIVE 12 [get_ports i2s_lrclk]

# I2S Data Inputs (6 microphones)
set_property PACKAGE_PIN R12 [get_ports i2s_data[0]]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_data[0]]
set_property PULLUP true [get_ports i2s_data[0]]

set_property PACKAGE_PIN T11 [get_ports i2s_data[1]]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_data[1]]
set_property PULLUP true [get_ports i2s_data[1]]

set_property PACKAGE_PIN R11 [get_ports i2s_data[2]]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_data[2]]
set_property PULLUP true [get_ports i2s_data[2]]

set_property PACKAGE_PIN T10 [get_ports i2s_data[3]]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_data[3]]
set_property PULLUP true [get_ports i2s_data[3]]

set_property PACKAGE_PIN R10 [get_ports i2s_data[4]]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_data[4]]
set_property PULLUP true [get_ports i2s_data[4]]

set_property PACKAGE_PIN T9 [get_ports i2s_data[5]]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_data[5]]
set_property PULLUP true [get_ports i2s_data[5]]

##############################
# UART Interface (Debug/Communication)
##############################

# UART TX (FPGA -> PC)
set_property PACKAGE_PIN J18 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]
set_property DRIVE 12 [get_ports uart_tx]

# UART RX (PC -> FPGA)
set_property PACKAGE_PIN H18 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]
set_property PULLUP true [get_ports uart_rx]

##############################
# GPIO and LED Indicators
##############################

# Status LED (Green)
set_property PACKAGE_PIN P18 [get_ports led_status]
set_property IOSTANDARD LVCMOS33 [get_ports led_status]
set_property DRIVE 12 [get_ports led_status]

# Alert LED (Red) - Gunshot detected
set_property PACKAGE_PIN N17 [get_ports led_alert]
set_property IOSTANDARD LVCMOS33 [get_ports led_alert]
set_property DRIVE 12 [get_ports led_alert]

# System LED (Blue)
set_property PACKAGE_PIN M18 [get_ports led_system]
set_property IOSTANDARD LVCMOS33 [get_ports led_system]
set_property DRIVE 12 [get_ports led_system]

# User Button (Active Low)
set_property PACKAGE_PIN K18 [get_ports btn_user]
set_property IOSTANDARD LVCMOS33 [get_ports btn_user]
set_property PULLUP true [get_ports btn_user]

# Reset Button
set_property PACKAGE_PIN J17 [get_ports btn_reset]
set_property IOSTANDARD LVCMOS33 [get_ports btn_reset]
set_property PULLUP true [get_ports btn_reset]

##############################
# External Triggers and Alarms
##############################

# External Alarm Output
set_property PACKAGE_PIN H17 [get_ports alarm_out]
set_property IOSTANDARD LVCMOS33 [get_ports alarm_out]
set_property DRIVE 12 [get_ports alarm_out]

# External Trigger Input
set_property PACKAGE_PIN G17 [get_ports trigger_in]
set_property IOSTANDARD LVCMOS33 [get_ports trigger_in]
set_property PULLUP true [get_ports trigger_in]

##############################
# Memory Interface (DDR3)
##############################

# DDR3 Clock
create_clock -name ddr_clk -period 5.000 [get_ports ddr3_clk_p]
set_property PACKAGE_PIN G3 [get_ports ddr3_clk_p]
set_property IOSTANDARD SSTL15 [get_ports ddr3_clk_p]

set_property PACKAGE_PIN G2 [get_ports ddr3_clk_n]
set_property IOSTANDARD SSTL15 [get_ports ddr3_clk_n]

# DDR3 Address/Command
set_property PACKAGE_PIN D1 [get_ports ddr3_addr[0]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[0]]

set_property PACKAGE_PIN E3 [get_ports ddr3_addr[1]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[1]]

set_property PACKAGE_PIN D2 [get_ports ddr3_addr[2]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[2]]

set_property PACKAGE_PIN C2 [get_ports ddr3_addr[3]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[3]]

set_property PACKAGE_PIN A3 [get_ports ddr3_addr[4]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[4]]

set_property PACKAGE_PIN B3 [get_ports ddr3_addr[5]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[5]]

set_property PACKAGE_PIN E2 [get_ports ddr3_addr[6]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[6]]

set_property PACKAGE_PIN C1 [get_ports ddr3_addr[7]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[7]]

set_property PACKAGE_PIN F3 [get_ports ddr3_addr[8]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[8]]

set_property PACKAGE_PIN B2 [get_ports ddr3_addr[9]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[9]]

set_property PACKAGE_PIN F2 [get_ports ddr3_addr[10]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[10]]

set_property PACKAGE_PIN A2 [get_ports ddr3_addr[11]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[11]]

set_property PACKAGE_PIN E1 [get_ports ddr3_addr[12]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[12]]

set_property PACKAGE_PIN B1 [get_ports ddr3_addr[13]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_addr[13]]

# DDR3 Data (DQ)
set_property PACKAGE_PIN G1 [get_ports ddr3_dq[0]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dq[0]]

set_property PACKAGE_PIN H3 [get_ports ddr3_dq[1]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dq[1]]

set_property PACKAGE_PIN H2 [get_ports ddr3_dq[2]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dq[2]]

set_property PACKAGE_PIN J3 [get_ports ddr3_dq[3]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dq[3]]

set_property PACKAGE_PIN J2 [get_ports ddr3_dq[4]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dq[4]]

set_property PACKAGE_PIN J1 [get_ports ddr3_dq[5]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dq[5]]

set_property PACKAGE_PIN K3 [get_ports ddr3_dq[6]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dq[6]]

set_property PACKAGE_PIN K2 [get_ports ddr3_dq[7]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dq[7]]

# DDR3 Data Strobe (DQS)
set_property PACKAGE_PIN H1 [get_ports ddr3_dqs_p[0]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dqs_p[0]]

set_property PACKAGE_PIN G4 [get_ports ddr3_dqs_n[0]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_dqs_n[0]]

# DDR3 Control
set_property PACKAGE_PIN F1 [get_ports ddr3_ba[0]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_ba[0]]

set_property PACKAGE_PIN D3 [get_ports ddr3_ba[1]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_ba[1]]

set_property PACKAGE_PIN C3 [get_ports ddr3_ba[2]]
set_property IOSTANDARD SSTL15 [get_ports ddr3_ba[2]]

set_property PACKAGE_PIN A1 [get_ports ddr3_cas_n]
set_property IOSTANDARD SSTL15 [get_ports ddr3_cas_n]

set_property PACKAGE_PIN L3 [get_ports ddr3_cke]
set_property IOSTANDARD SSTL15 [get_ports ddr3_cke]

set_property PACKAGE_PIN M3 [get_ports ddr3_cs_n]
set_property IOSTANDARD SSTL15 [get_ports ddr3_cs_n]

set_property PACKAGE_PIN L1 [get_ports ddr3_odt]
set_property IOSTANDARD SSTL15 [get_ports ddr3_odt]

set_property PACKAGE_PIN M2 [get_ports ddr3_ras_n]
set_property IOSTANDARD SSTL15 [get_ports ddr3_ras_n]

set_property PACKAGE_PIN L2 [get_ports ddr3_reset_n]
set_property IOSTANDARD SSTL15 [get_ports ddr3_reset_n]

set_property PACKAGE_PIN M1 [get_ports ddr3_we_n]
set_property IOSTANDARD SSTL15 [get_ports ddr3_we_n]

##############################
# SD Card Interface (for boot)
##############################

# SD Card Clock
set_property PACKAGE_PIN T15 [get_ports sdio_clk]
set_property IOSTANDARD LVCMOS33 [get_ports sdio_clk]
set_property DRIVE 12 [get_ports sdio_clk]

# SD Card Command
set_property PACKAGE_PIN R14 [get_ports sdio_cmd]
set_property IOSTANDARD LVCMOS33 [get_ports sdio_cmd]
set_property PULLUP true [get_ports sdio_cmd]

# SD Card Data
set_property PACKAGE_PIN T14 [get_ports sdio_data[0]]
set_property IOSTANDARD LVCMOS33 [get_ports sdio_data[0]]
set_property PULLUP true [get_ports sdio_data[0]]

set_property PACKAGE_PIN R15 [get_ports sdio_data[1]]
set_property IOSTANDARD LVCMOS33 [get_ports sdio_data[1]]
set_property PULLUP true [get_ports sdio_data[1]]

set_property PACKAGE_PIN P15 [get_ports sdio_data[2]]
set_property IOSTANDARD LVCMOS33 [get_ports sdio_data[2]]
set_property PULLUP true [get_ports sdio_data[2]]

set_property PACKAGE_PIN N14 [get_ports sdio_data[3]]
set_property IOSTANDARD LVCMOS33 [get_ports sdio_data[3]]
set_property PULLUP true [get_ports sdio_data[3]]

##############################
# Timing Constraints
##############################

# SPI timing constraints
set_input_delay -clock [get_clocks clk_100m] -min -add_delay 2.0 [get_ports spi_miso]
set_input_delay -clock [get_clocks clk_100m] -max -add_delay 5.0 [get_ports spi_miso]

set_output_delay -clock [get_clocks clk_100m] -min -add_delay 1.0 [get_ports spi_sck]
set_output_delay -clock [get_clocks clk_100m] -max -add_delay 3.0 [get_ports spi_sck]

set_output_delay -clock [get_clocks clk_100m] -min -add_delay 1.0 [get_ports spi_mosi]
set_output_delay -clock [get_clocks clk_100m] -max -add_delay 3.0 [get_ports spi_mosi]

# I2S timing constraints
set_input_delay -clock [get_clocks i2s_mclk] -min -add_delay 1.0 [get_ports i2s_data*]
set_input_delay -clock [get_clocks i2s_mclk] -max -add_delay 4.0 [get_ports i2s_data*]

set_output_delay -clock [get_clocks i2s_mclk] -min -add_delay 1.0 [get_ports i2s_bclk]
set_output_delay -clock [get_clocks i2s_mclk] -max -add_delay 3.0 [get_ports i2s_bclk]

set_output_delay -clock [get_clocks i2s_mclk] -min -add_delay 1.0 [get_ports i2s_lrclk]
set_output_delay -clock [get_clocks i2s_mclk] -max -add_delay 3.0 [get_ports i2s_lrclk]

# UART timing
set_input_delay -clock [get_clocks clk_100m] -min -add_delay 1.0 [get_ports uart_rx]
set_input_delay -clock [get_clocks clk_100m] -max -add_delay 3.0 [get_ports uart_rx]

set_output_delay -clock [get_clocks clk_100m] -min -add_delay 1.0 [get_ports uart_tx]
set_output_delay -clock [get_clocks clk_100m] -max -add_delay 3.0 [get_ports uart_tx]

# False path declarations
set_false_path -from [get_ports btn_user]
set_false_path -from [get_ports btn_reset]
set_false_path -from [get_ports trigger_in]

# Clock groups
set_clock_groups -asynchronous \
    -group [get_clocks clk_100m] \
    -group [get_clocks i2s_mclk] \
    -group [get_clocks ddr_clk]

##############################
# Power Constraints
##############################

# Reduce drive strength for power savings
set_property DRIVE 8 [get_ports {led_status led_alert led_system}]
set_property SLEW SLOW [get_ports {spi_* i2s_*}]

# Disable unused pullups
set_property PULLTYPE NONE [get_ports {uart_tx alarm_out}]

##############################
# I/O Standard Summary
##############################

puts "========================================"
puts "Spartan Edge Accelerator Board Constraints"
puts "========================================"
puts "Total I/O pins constrained: [llength [get_ports]]"
puts "LVCMOS33 pins: [llength [get_ports -filter {IOSTANDARD == LVCMOS33}]]"
puts "SSTL15 pins: [llength [get_ports -filter {IOSTANDARD == SSTL15}]]"
puts "========================================"

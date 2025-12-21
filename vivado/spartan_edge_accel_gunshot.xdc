## Clock Definitions
create_clock -name clk_fpga -period 10.0 [get_ports clk_100MHz]
create_clock -name clk_i2s -period 20.833 [get_ports i2s_mclk]

## I2S Interface Pins
set_property PACKAGE_PIN U18 [get_ports i2s_mclk]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_mclk]
set_property DRIVE 12 [get_ports i2s_mclk]

set_property PACKAGE_PIN U17 [get_ports i2s_bclk]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_bclk]
set_property PULLDOWN TRUE [get_ports i2s_bclk]

set_property PACKAGE_PIN V17 [get_ports i2s_lrclk]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_lrclk]
set_property PULLDOWN TRUE [get_ports i2s_lrclk]

## I2S Data Pins (6 microphones)
set_property PACKAGE_PIN V16 [get_ports i2s_data[0]]
set_property PACKAGE_PIN U16 [get_ports i2s_data[1]]
set_property PACKAGE_PIN T17 [get_ports i2s_data[2]]
set_property PACKAGE_PIN T16 [get_ports i2s_data[3]]
set_property PACKAGE_PIN T15 [get_ports i2s_data[4]]
set_property PACKAGE_PIN R17 [get_ports i2s_data[5]]
set_property IOSTANDARD LVCMOS33 [get_ports i2s_data*]
set_property PULLDOWN TRUE [get_ports i2s_data*]

## Output Indicators
set_property PACKAGE_PIN N16 [get_ports alert_led]
set_property IOSTANDARD LVCMOS33 [get_ports alert_led]
set_property DRIVE 12 [get_ports alert_led]

set_property PACKAGE_PIN M16 [get_ports status_led]
set_property IOSTANDARD LVCMOS33 [get_ports status_led]
set_property DRIVE 12 [get_ports status_led]

## UART Interface
set_property PACKAGE_PIN K16 [get_ports uart_tx]
set_property PACKAGE_PIN J16 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx uart_rx]
set_property PULLUP TRUE [get_ports uart_rx]

## Timing Constraints
set_input_delay -clock [get_clocks clk_i2s] -min 1.0 [get_ports i2s_data*]
set_input_delay -clock [get_clocks clk_i2s] -max 3.0 [get_ports i2s_data*]
set_input_delay -clock [get_clocks clk_i2s] 0.5 [get_ports i2s_lrclk]

set_output_delay -clock [get_clocks clk_fpga] -min 0 [get_ports alert_led]
set_output_delay -clock [get_clocks clk_fpga] -max 1 [get_ports alert_led]

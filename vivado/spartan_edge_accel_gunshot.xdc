## Clock Definitions
create_clock -name clk_fpga -period 10.0 [get_ports clk_100MHz]

## SPI Interface Pins (for Microphone ADC)
set_property PACKAGE_PIN U18 [get_ports spi_sck]
set_property IOSTANDARD LVCMOS33 [get_ports spi_sck]
set_property DRIVE 12 [get_ports spi_sck]

set_property PACKAGE_PIN U17 [get_ports spi_mosi]
set_property IOSTANDARD LVCMOS33 [get_ports spi_mosi]

set_property PACKAGE_PIN V17 [get_ports spi_miso]
set_property IOSTANDARD LVCMOS33 [get_ports spi_miso]

set_property PACKAGE_PIN V16 [get_ports spi_cs]
set_property IOSTANDARD LVCMOS33 [get_ports spi_cs]

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

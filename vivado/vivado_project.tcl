# vivado_project.tcl
# Complete Vivado project creation for Gunshot Detection System
# Target: Spartan Edge Accelerator Board (Spartan-7 XC7S50)

# Set project parameters
set PROJECT_NAME "gunshot_detection_fpga"
set PROJECT_DIR "vivado_project"
set BOARD_PART "digilentinc.com:spartan-edge-accelerator:part0:1.0"
set DEVICE_PART "xc7s50-cpg236-1"
set TARGET_LANGUAGE "Verilog"

# Create project
create_project $PROJECT_NAME $PROJECT_DIR -part $DEVICE_PART -force

# Set project properties
set_property board_part $BOARD_PART [current_project]
set_property target_language $TARGET_LANGUAGE [current_project]
set_property default_lib work [current_project]
set_property simulator_language Mixed [current_project]
set_property ip_output_repo ./ip_repo [current_project]

# Create sources directory structure
file mkdir sources_1/sources
file mkdir sources_1/constraints
file mkdir sources_1/ip
file mkdir sim_1

# 1. Add RTL source files
# Create and add axi4lite_dut.v
set axi4lite_dut_content {
`timescale 1ns / 1ps

module axi4lite_dut(
    input wire clk,
    input wire reset_n,
    
    // AXI4-Lite Write Address Channel
    input wire [31:0] awaddr,
    input wire        awvalid,
    output reg        awready,
    
    // AXI4-Lite Write Data Channel
    input wire [31:0] wdata,
    input wire [3:0]  wstrb,
    input wire        wvalid,
    output reg        wready,
    
    // AXI4-Lite Write Response Channel
    output reg [1:0]  bresp,
    output reg        bvalid,
    input wire        bready,
    
    // AXI4-Lite Read Address Channel
    input wire [31:0] araddr,
    input wire        arvalid,
    output reg        arready,
    
    // AXI4-Lite Read Data Channel
    output reg [31:0] rdata,
    output reg [1:0]  rresp,
    output reg        rvalid,
    input wire        rready
);
    
    // Internal memory (4KB = 1024 x 32-bit words)
    reg [31:0] memory [0:1023];
    
    // Internal state machine states (one-hot encoding)
    parameter [2:0] IDLE       = 3'b001;
    parameter [2:0] WRITE_ADDR = 3'b010;
    parameter [2:0] WRITE_DATA = 3'b100;
    parameter [2:0] WRITE_RESP = 3'b001; // Reusing IDLE code
    parameter [2:0] READ_ADDR  = 3'b010; // Reusing WRITE_ADDR code
    parameter [2:0] READ_DATA  = 3'b100; // Reusing WRITE_DATA code
    
    reg [2:0] current_state;
    reg [2:0] next_state;
    
    // Internal registers
    reg [31:0] write_addr_reg;
    reg [31:0] read_addr_reg;
    reg [31:0] write_data_reg;
    reg [3:0]  write_strb_reg;
    
    // FSM state transition
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end
    
    // FSM next state logic
    always @(*) begin
        next_state = current_state;
        
        case (current_state)
            IDLE: begin
                if (awvalid) begin
                    next_state = WRITE_ADDR;
                end else if (arvalid) begin
                    next_state = READ_ADDR;
                end
            end
            
            WRITE_ADDR: begin
                next_state = WRITE_DATA;
            end
            
            WRITE_DATA: begin
                next_state = WRITE_RESP;
            end
            
            WRITE_RESP: begin
                if (bready) begin
                    next_state = IDLE;
                end
            end
            
            READ_ADDR: begin
                next_state = READ_DATA;
            end
            
            READ_DATA: begin
                if (rready) begin
                    next_state = IDLE;
                end
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    // FSM output logic
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            // Initialize outputs
            awready <= 1'b0;
            wready <= 1'b0;
            bresp <= 2'b00;
            bvalid <= 1'b0;
            arready <= 1'b0;
            rdata <= 32'h0;
            rresp <= 2'b00;
            rvalid <= 1'b0;
            
            // Initialize internal registers
            write_addr_reg <= 32'h0;
            read_addr_reg <= 32'h0;
            write_data_reg <= 32'h0;
            write_strb_reg <= 4'b0000;
            
            // Initialize memory
            integer i;
            for (i = 0; i < 1024; i = i + 1) begin
                memory[i] <= 32'h0;
            end
        end else begin
            // Default outputs
            awready <= 1'b0;
            wready <= 1'b0;
            bvalid <= 1'b0;
            arready <= 1'b0;
            rvalid <= 1'b0;
            
            case (current_state)
                IDLE: begin
                    // Ready to accept new transactions
                    if (awvalid) begin
                        awready <= 1'b1;
                        write_addr_reg <= awaddr;
                    end else if (arvalid) begin
                        arready <= 1'b1;
                        read_addr_reg <= araddr;
                    end
                end
                
                WRITE_ADDR: begin
                    // Accept write data
                    if (wvalid) begin
                        wready <= 1'b1;
                        write_data_reg <= wdata;
                        write_strb_reg <= wstrb;
                    end
                end
                
                WRITE_DATA: begin
                    // Write to memory with byte enables
                    if (write_addr_reg[31:12] == 20'h0) begin
                        // Valid address range (0x0000-0x0FFF)
                        reg [9:0] mem_addr;
                        mem_addr = write_addr_reg[11:2];
                        if (mem_addr < 1024) begin
                            // Apply byte enables
                            if (write_strb_reg[0])
                                memory[mem_addr][7:0] <= write_data_reg[7:0];
                            if (write_strb_reg[1])
                                memory[mem_addr][15:8] <= write_data_reg[15:8];
                            if (write_strb_reg[2])
                                memory[mem_addr][23:16] <= write_data_reg[23:16];
                            if (write_strb_reg[3])
                                memory[mem_addr][31:24] <= write_data_reg[31:24];
                        end
                        bresp <= 2'b00; // OKAY response
                    end else begin
                        bresp <= 2'b10; // SLVERR response
                    end
                    bvalid <= 1'b1;
                end
                
                WRITE_RESP: begin
                    // Keep response valid until accepted
                    bvalid <= 1'b1;
                end
                
                READ_ADDR: begin
                    // Read from memory
                    if (read_addr_reg[31:12] == 20'h0) begin
                        // Valid address range
                        reg [9:0] mem_addr;
                        mem_addr = read_addr_reg[11:2];
                        if (mem_addr < 1024) begin
                            rdata <= memory[mem_addr];
                            rresp <= 2'b00; // OKAY response
                        end else begin
                            rdata <= 32'h0;
                            rresp <= 2'b10; // SLVERR response
                        end
                    end else begin
                        rdata <= 32'h0;
                        rresp <= 2'b10; // SLVERR response
                    end
                    rvalid <= 1'b1;
                end
                
                READ_DATA: begin
                    // Keep data valid until accepted
                    rvalid <= 1'b1;
                end
            endcase
        end
    end
    
endmodule
}

set axi4lite_dut_file [open "sources_1/sources/axi4lite_dut.v" w]
puts $axi4lite_dut_file $axi4lite_dut_content
close $axi4lite_dut_file
add_files -norecurse sources_1/sources/axi4lite_dut.v

# Create and add audio processing modules
set i2s_receiver_content {
`timescale 1ns / 1ps

module i2s_receiver(
    input wire clk_100mhz,
    input wire reset_n,
    
    // I2S Interface
    input wire i2s_mclk,
    input wire i2s_bclk,
    input wire i2s_lrclk,
    input wire [5:0] i2s_data,  // 6 microphones
    
    // AXI Stream Output
    output reg [31:0] m_axis_tdata,
    output reg [5:0]  m_axis_tuser,  // Channel ID
    output reg        m_axis_tvalid,
    input  wire       m_axis_tready,
    output reg        m_axis_tlast
);
    
    // Internal signals
    reg [15:0] audio_samples [0:5];  // 6 channels
    reg [2:0]  bit_counter;
    reg [3:0]  sample_counter;
    reg        lrclk_d;
    reg [5:0]  shift_regs [0:5];
    reg        capturing;
    reg [1:0]  channel_id;
    
    // Detect LRCLK edges
    always @(posedge clk_100mhz) begin
        lrclk_d <= i2s_lrclk;
    end
    
    // Sample I2S data on BCLK falling edge
    always @(negedge i2s_bclk or negedge reset_n) begin
        if (!reset_n) begin
            for (integer i = 0; i < 6; i = i + 1) begin
                shift_regs[i] <= 6'b0;
            end
            bit_counter <= 3'd0;
            capturing <= 1'b0;
        end else begin
            // Shift in data for each channel
            for (integer i = 0; i < 6; i = i + 1) begin
                shift_regs[i] <= {shift_regs[i][4:0], i2s_data[i]};
            end
            
            bit_counter <= bit_counter + 1;
            
            if (bit_counter == 3'd5) begin  // 6 bits per channel
                bit_counter <= 3'd0;
                capturing <= 1'b1;
            end else begin
                capturing <= 1'b0;
            end
        end
    end
    
    // Store complete samples
    always @(posedge clk_100mhz or negedge reset_n) begin
        if (!reset_n) begin
            for (integer i = 0; i < 6; i = i + 1) begin
                audio_samples[i] <= 16'b0;
            end
            sample_counter <= 4'd0;
            channel_id <= 2'd0;
        end else if (capturing) begin
            // Store the shifted data
            for (integer i = 0; i < 6; i = i + 1) begin
                audio_samples[i] <= {audio_samples[i][9:0], shift_regs[i]};
            end
            
            sample_counter <= sample_counter + 1;
            
            // After 16 samples, prepare to output
            if (sample_counter == 4'd15) begin
                channel_id <= channel_id + 1;
            end
        end
    end
    
    // AXI Stream output logic
    always @(posedge clk_100mhz or negedge reset_n) begin
        if (!reset_n) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tdata <= 32'b0;
            m_axis_tuser <= 6'b0;
            m_axis_tlast <= 1'b0;
        end else begin
            if (sample_counter == 4'd15 && capturing) begin
                m_axis_tvalid <= 1'b1;
                m_axis_tdata <= {16'b0, audio_samples[channel_id]};
                m_axis_tuser <= (1'b1 << channel_id);
                m_axis_tlast <= (channel_id == 2'd5);
            end else if (m_axis_tready) begin
                m_axis_tvalid <= 1'b0;
            end
        end
    end
    
endmodule
}

set i2s_receiver_file [open "sources_1/sources/i2s_receiver.v" w]
puts $i2s_receiver_file $i2s_receiver_content
close $i2s_receiver_file
add_files -norecurse sources_1/sources/i2s_receiver.v

# Create and add FFT module
set fft_256_module_content {
`timescale 1ns / 1ps

module fft_256_module(
    input wire clk,
    input wire reset_n,
    input wire start,
    input wire signed [15:0] audio_in,
    output reg [31:0] magnitude_out,
    output reg done
);
    
    // FFT parameters
    parameter N = 256;
    parameter LOG2N = 8;
    
    // Twiddle factors ROM (simplified)
    reg [15:0] cos_rom [0:N-1];
    reg [15:0] sin_rom [0:N-1];
    
    // FFT memory
    reg signed [15:0] real_mem [0:N-1];
    reg signed [15:0] imag_mem [0:N-1];
    
    // State machine
    reg [2:0] state;
    reg [LOG2N-1:0] stage;
    reg [LOG2N-1:0] butterfly;
    reg [LOG2N-1:0] index;
    
    parameter IDLE = 3'b000;
    parameter LOAD = 3'b001;
    parameter PROCESS = 3'b010;
    parameter STORE = 3'b011;
    parameter DONE = 3'b100;
    
    // Initialize ROM with precomputed values
    initial begin
        for (integer i = 0; i < N; i = i + 1) begin
            cos_rom[i] = $cos(2.0 * 3.141592653589793 * i / N) * 32767;
            sin_rom[i] = $sin(2.0 * 3.141592653589793 * i / N) * 32767;
        end
    end
    
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            done <= 1'b0;
            magnitude_out <= 32'b0;
            for (integer i = 0; i < N; i = i + 1) begin
                real_mem[i] <= 16'b0;
                imag_mem[i] <= 16'b0;
            end
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        state <= LOAD;
                        index <= 0;
                    end
                end
                
                LOAD: begin
                    // Load audio samples (windowed)
                    real_mem[index] <= audio_in;
                    imag_mem[index] <= 16'b0;
                    
                    if (index == N-1) begin
                        state <= PROCESS;
                        stage <= 0;
                    end else begin
                        index <= index + 1;
                    end
                end
                
                PROCESS: begin
                    // Butterfly computation (simplified)
                    reg signed [31:0] temp_real, temp_imag;
                    reg signed [15:0] cos_val, sin_val;
                    
                    // Get twiddle factors
                    cos_val = cos_rom[butterfly];
                    sin_val = sin_rom[butterfly];
                    
                    // Perform butterfly operation
                    temp_real = (real_mem[index] * cos_val - imag_mem[index] * sin_val) >> 15;
                    temp_imag = (real_mem[index] * sin_val + imag_mem[index] * cos_val) >> 15;
                    
                    // Store results
                    real_mem[index] <= temp_real[15:0];
                    imag_mem[index] <= temp_imag[15:0];
                    
                    // Update indices
                    if (butterfly == N/2 - 1) begin
                        butterfly <= 0;
                        if (stage == LOG2N - 1) begin
                            state <= STORE;
                            index <= 0;
                        end else begin
                            stage <= stage + 1;
                        end
                    end else begin
                        butterfly <= butterfly + 1;
                    end
                end
                
                STORE: begin
                    // Compute magnitude
                    reg signed [31:0] mag_sq;
                    mag_sq = real_mem[index] * real_mem[index] + 
                             imag_mem[index] * imag_mem[index];
                    magnitude_out <= mag_sq;
                    
                    if (index == N-1) begin
                        state <= DONE;
                    end else begin
                        index <= index + 1;
                    end
                end
                
                DONE: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end
    
endmodule
}

set fft_256_module_file [open "sources_1/sources/fft_256_module.v" w]
puts $fft_256_module_file $fft_256_module_content
close $fft_256_module_file
add_files -norecurse sources_1/sources/fft_256_module.v

# Create and add gunshot detector top module
set gunshot_detector_top_content {
`timescale 1ns / 1ps

module gunshot_detector_top(
    input wire clk_100mhz,
    input wire reset_n,
    
    // I2S Microphone Interface (6 channels)
    input wire i2s_mclk,
    input wire i2s_bclk,
    input wire i2s_lrclk,
    input wire [5:0] i2s_data,
    
    // AXI4-Lite Control Interface
    // Write Address Channel
    input wire [31:0] s_axi_awaddr,
    input wire        s_axi_awvalid,
    output reg        s_axi_awready,
    
    // Write Data Channel
    input wire [31:0] s_axi_wdata,
    input wire [3:0]  s_axi_wstrb,
    input wire        s_axi_wvalid,
    output reg        s_axi_wready,
    
    // Write Response Channel
    output reg [1:0]  s_axi_bresp,
    output reg        s_axi_bvalid,
    input wire        s_axi_bready,
    
    // Read Address Channel
    input wire [31:0] s_axi_araddr,
    input wire        s_axi_arvalid,
    output reg        s_axi_arready,
    
    // Read Data Channel
    output reg [31:0] s_axi_rdata,
    output reg [1:0]  s_axi_rresp,
    output reg        s_axi_rvalid,
    input wire        s_axi_rready,
    
    // Detection Output
    output reg        gunshot_detected,
    output reg [7:0]  confidence,
    output reg [15:0] x_position,
    output reg [15:0] y_position,
    
    // Status LEDs
    output reg        led_alert,
    output reg        led_status
);
    
    // Internal signals
    wire [31:0] audio_data [0:5];
    wire [5:0]  audio_valid;
    wire [5:0]  audio_ready;
    
    wire [31:0] fft_magnitude [0:5];
    wire [5:0]  fft_done;
    
    wire [7:0]  nn_confidence;
    wire        nn_detection;
    
    // Control registers
    reg [31:0] control_reg;
    reg [31:0] threshold_reg;
    reg [31:0] status_reg;
    
    // I2S Receiver instances (6 channels)
    genvar i;
    generate
        for (i = 0; i < 6; i = i + 1) begin : i2s_receivers
            i2s_receiver i2s_rx (
                .clk_100mhz(clk_100mhz),
                .reset_n(reset_n),
                .i2s_mclk(i2s_mclk),
                .i2s_bclk(i2s_bclk),
                .i2s_lrclk(i2s_lrclk),
                .i2s_data(i2s_data[i]),
                .m_axis_tdata(audio_data[i]),
                .m_axis_tvalid(audio_valid[i]),
                .m_axis_tready(audio_ready[i])
            );
        end
    endgenerate
    
    // FFT modules (6 channels)
    generate
        for (i = 0; i < 6; i = i + 1) begin : fft_modules
            fft_256_module fft_inst (
                .clk(clk_100mhz),
                .reset_n(reset_n),
                .start(audio_valid[i]),
                .audio_in(audio_data[i][15:0]),
                .magnitude_out(fft_magnitude[i]),
                .done(fft_done[i])
            );
        end
    endgenerate
    
    // Feature extraction
    reg [31:0] feature_vector [0:2051];
    reg        feature_valid;
    integer    feature_idx;
    
    always @(posedge clk_100mhz or negedge reset_n) begin
        if (!reset_n) begin
            feature_valid <= 1'b0;
            feature_idx <= 0;
            for (integer j = 0; j < 2052; j = j + 1) begin
                feature_vector[j] <= 32'b0;
            end
        end else begin
            if (&fft_done) begin  // All FFTs complete
                // Extract statistical features (simplified)
                for (integer bin = 0; bin < 256; bin = bin + 1) begin
                    // Mean
                    feature_vector[bin*4] <= fft_magnitude[0][bin] + 
                                             fft_magnitude[1][bin] + 
                                             fft_magnitude[2][bin] + 
                                             fft_magnitude[3][bin] + 
                                             fft_magnitude[4][bin] + 
                                             fft_magnitude[5][bin];
                    feature_valid <= 1'b1;
                end
            end else begin
                feature_valid <= 1'b0;
            end
        end
    end
    
    // Neural Network Inference (placeholder)
    always @(posedge clk_100mhz or negedge reset_n) begin
        if (!reset_n) begin
            nn_confidence <= 8'b0;
            nn_detection <= 1'b0;
        end else if (feature_valid) begin
            // Simple threshold-based detection for now
            if (feature_vector[0] > threshold_reg) begin
                nn_detection <= 1'b1;
                nn_confidence <= feature_vector[0][15:8];
            end else begin
                nn_detection <= 1'b0;
                nn_confidence <= 8'b0;
            end
        end
    end
    
    // MUSIC Localization (simplified)
    always @(posedge clk_100mhz or negedge reset_n) begin
        if (!reset_n) begin
            x_position <= 16'b0;
            y_position <= 16'b0;
        end else if (nn_detection) begin
            // Simple DoA estimation based on channel differences
            reg [31:0] diff [0:5];
            for (integer i = 0; i < 6; i = i + 1) begin
                diff[i] = fft_magnitude[i] - fft_magnitude[(i+3)%6];
            end
            
            // Basic triangulation
            x_position <= diff[0] + diff[1] - diff[3] - diff[4];
            y_position <= diff[1] + diff[2] - diff[4] - diff[5];
        end
    end
    
    // Output assignments
    always @(posedge clk_100mhz or negedge reset_n) begin
        if (!reset_n) begin
            gunshot_detected <= 1'b0;
            confidence <= 8'b0;
            led_alert <= 1'b0;
            led_status <= 1'b0;
        end else begin
            gunshot_detected <= nn_detection;
            confidence <= nn_confidence;
            led_alert <= nn_detection;
            led_status <= 1'b1;  // Always on when system is running
        end
    end
    
    // AXI4-Lite Interface
    axi4lite_dut axi4lite_interface (
        .clk(clk_100mhz),
        .reset_n(reset_n),
        .awaddr(s_axi_awaddr),
        .awvalid(s_axi_awvalid),
        .awready(s_axi_awready),
        .wdata(s_axi_wdata),
        .wstrb(s_axi_wstrb),
        .wvalid(s_axi_wvalid),
        .wready(s_axi_wready),
        .bresp(s_axi_bresp),
        .bvalid(s_axi_bvalid),
        .bready(s_axi_bready),
        .araddr(s_axi_araddr),
        .arvalid(s_axi_arvalid),
        .arready(s_axi_arready),
        .rdata(s_axi_rdata),
        .rresp(s_axi_rresp),
        .rvalid(s_axi_rvalid),
        .rready(s_axi_rready)
    );
    
    // Control register mapping
    always @(posedge clk_100mhz) begin
        if (s_axi_wvalid && s_axi_wready) begin
            case (s_axi_awaddr[7:0])
                8'h00: control_reg <= s_axi_wdata;
                8'h04: threshold_reg <= s_axi_wdata;
            endcase
        end
        
        // Update status register
        status_reg <= {16'b0, confidence, 7'b0, gunshot_detected};
    end
    
    // Read responses
    always @(posedge clk_100mhz) begin
        if (s_axi_arvalid && s_axi_arready) begin
            case (s_axi_araddr[7:0])
                8'h00: s_axi_rdata <= control_reg;
                8'h04: s_axi_rdata <= threshold_reg;
                8'h08: s_axi_rdata <= status_reg;
                8'h0C: s_axi_rdata <= {16'b0, x_position};
                8'h10: s_axi_rdata <= {16'b0, y_position};
                default: s_axi_rdata <= 32'hDEADBEEF;
            endcase
        end
    end
    
endmodule
}

set gunshot_detector_top_file [open "sources_1/sources/gunshot_detector_top.v" w]
puts $gunshot_detector_top_file $gunshot_detector_top_content
close $gunshot_detector_top_file
add_files -norecurse sources_1/sources/gunshot_detector_top.v

# 2. Add constraints file
set constraints_content {
## Clock Definitions
create_clock -name clk_100mhz -period 10.0 [get_ports clk_100mhz]

## I2S Interface Pins (Spartan Edge Accelerator Board)
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
set_property PACKAGE_PIN V16 [get_ports {i2s_data[0]}]
set_property PACKAGE_PIN U16 [get_ports {i2s_data[1]}]
set_property PACKAGE_PIN T17 [get_ports {i2s_data[2]}]
set_property PACKAGE_PIN T16 [get_ports {i2s_data[3]}]
set_property PACKAGE_PIN T15 [get_ports {i2s_data[4]}]
set_property PACKAGE_PIN R17 [get_ports {i2s_data[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {i2s_data[*]}]
set_property PULLDOWN TRUE [get_ports {i2s_data[*]}]

## Output LEDs
set_property PACKAGE_PIN N16 [get_ports led_alert]
set_property IOSTANDARD LVCMOS33 [get_ports led_alert]
set_property DRIVE 12 [get_ports led_alert]

set_property PACKAGE_PIN M16 [get_ports led_status]
set_property IOSTANDARD LVCMOS33 [get_ports led_status]
set_property DRIVE 12 [get_ports led_status]

## AXI4-Lite Interface (PMOD or GPIO)
set_property PACKAGE_PIN L17 [get_ports {s_axi_awaddr[0]}]
set_property PACKAGE_PIN K17 [get_ports {s_axi_awaddr[1]}]
set_property PACKAGE_PIN J17 [get_ports {s_axi_awaddr[2]}]
set_property PACKAGE_PIN J18 [get_ports {s_axi_awaddr[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {s_axi_awaddr[*]}]

set_property PACKAGE_PIN H17 [get_ports s_axi_awvalid]
set_property PACKAGE_PIN G17 [get_ports s_axi_awready]
set_property PACKAGE_PIN F17 [get_ports {s_axi_wdata[0]}]
set_property PACKAGE_PIN E17 [get_ports {s_axi_wdata[1]}]
set_property PACKAGE_PIN D17 [get_ports {s_axi_wdata[2]}]
set_property PACKAGE_PIN C17 [get_ports {s_axi_wdata[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {s_axi_wdata[*]}]

## Timing Constraints
set_input_delay -clock [get_clocks clk_100mhz] -min 1.0 [get_ports i2s_*]
set_input_delay -clock [get_clocks clk_100mhz] -max 3.0 [get_ports i2s_*]

set_output_delay -clock [get_clocks clk_100mhz] -min 0 [get_ports led_*]
set_output_delay -clock [get_clocks clk_100mhz] -max 1 [get_ports led_*]

## False Paths
set_false_path -from [get_ports i2s_mclk]
set_false_path -from [get_ports i2s_bclk]
set_false_path -from [get_ports i2s_lrclk]
}

set constraints_file [open "sources_1/constraints/spartan_edge_constraints.xdc" w]
puts $constraints_file $constraints_content
close $constraints_file
add_files -fileset constrs_1 -norecurse sources_1/constraints/spartan_edge_constraints.xdc

# 3. Add simulation files
set tb_top_content {
`timescale 1ns / 1ps

module tb_top();
    
    // Clock and reset
    reg clk_100mhz;
    reg reset_n;
    
    // I2S signals
    reg i2s_mclk;
    reg i2s_bclk;
    reg i2s_lrclk;
    reg [5:0] i2s_data;
    
    // AXI4-Lite signals
    reg [31:0] s_axi_awaddr;
    reg        s_axi_awvalid;
    wire       s_axi_awready;
    reg [31:0] s_axi_wdata;
    reg [3:0]  s_axi_wstrb;
    reg        s_axi_wvalid;
    wire       s_axi_wready;
    wire [1:0] s_axi_bresp;
    wire       s_axi_bvalid;
    reg        s_axi_bready;
    reg [31:0] s_axi_araddr;
    reg        s_axi_arvalid;
    wire       s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;
    
    // Output signals
    wire       gunshot_detected;
    wire [7:0] confidence;
    wire [15:0] x_position;
    wire [15:0] y_position;
    wire       led_alert;
    wire       led_status;
    
    // Testbench variables
    integer test_pass = 0;
    integer test_fail = 0;
    
    // Instantiate DUT
    gunshot_detector_top dut (
        .clk_100mhz(clk_100mhz),
        .reset_n(reset_n),
        
        .i2s_mclk(i2s_mclk),
        .i2s_bclk(i2s_bclk),
        .i2s_lrclk(i2s_lrclk),
        .i2s_data(i2s_data),
        
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        
        .gunshot

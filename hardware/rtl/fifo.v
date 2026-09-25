// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

module fifo #(
    parameter WIDTH = 1,
    parameter DEPTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire wr_en,
    input  wire rd_en,
    input  wire [WIDTH-1:0] din,
    output reg  [WIDTH-1:0] dout,
    output wire wr_ready,
    output wire full,
    output wire empty,
    output wire underflow
);

    reg [WIDTH-1:0] mem [0:DEPTH-1] ;
    localparam integer PTR_WIDTH = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam integer COUNT_WIDTH = $clog2(DEPTH + 1);
    reg [PTR_WIDTH-1:0] wr_ptr;
    reg [PTR_WIDTH-1:0] rd_ptr;
    reg [COUNT_WIDTH-1:0] count;
    localparam [PTR_WIDTH-1:0] LAST_PTR = PTR_WIDTH'(DEPTH - 1);

    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    // A write may replace an entry consumed on the same edge even when the
    // FIFO starts full. If it starts empty, simultaneous read/write bypasses
    // din into dout. This registered bypass preserves the decoder's existing
    // one-cycle input staging without requiring whole-tile preloading.
    wire stored_read = rd_en && !empty;
    // wr_ready simplified from `!full || stored_read` to `!full` to break
    // the mm_efficient UNOPTFLAT stall/rd_en/wr_ready loop (see the same
    // comment in src/act_fifo.v for the full trace and characterization).
    // Bypass path is independent and unaffected.
    assign wr_ready = !full;
    wire write_fire = wr_en && wr_ready;
    wire bypass_fire = rd_en && empty && write_fire;
    wire store_fire = write_fire && !bypass_fire;
    assign underflow = rd_en && empty && !write_fire;

    function automatic [PTR_WIDTH-1:0] next_ptr;
        input [PTR_WIDTH-1:0] ptr;
        begin
            next_ptr = (ptr == LAST_PTR) ? {PTR_WIDTH{1'b0}} : ptr + 1'b1;
        end
    endfunction

`ifndef SYNTHESIS
    // With the stall-backpressure fix in mm_efficient, rd_en can never be
    // asserted on an empty FIFO. Any underflow here means the stall gating
    // is buggy and stale operands are entering the PE — the exact silent
    // corruption the fix was written to prevent. Fail loudly.
    always @(posedge clk) begin
        if (rst && underflow)
            $fatal(1,
                   "FIFO underflow at %m t=%0t (stall gating bug: rd_en=1 on empty)",
                   $time);
    end
`endif

    // ------------------------------------------------------------------
    // Storage array + output register. Deliberately NO reset.
    //
    // A block RAM cannot be asynchronously reset, and neither can a compiled
    // SRAM, so a memory sitting inside an async-reset process cannot be
    // inferred as one; synthesis would dissolve every FIFO into flip-flops. Splitting the array and its
    // output register into their own reset-free process is what both
    // technologies actually want.
    //
    // Safety: nothing reads `dout` unless the FIFO reports data, and `count`
    // -- which is what `empty` is derived from -- IS still reset below. So
    // stale contents after reset are unreachable. `initial` keeps simulation
    // X-free; ASIC synthesis ignores it, matching real SRAM power-up.
    initial dout = {WIDTH{1'b0}};
    always @(posedge clk) begin
        if (bypass_fire)
            dout <= din;
        else if (stored_read)
            dout <= mem[rd_ptr];

        if (store_fire)
            mem[wr_ptr] <= din;
    end

    // Pointers and occupancy keep their reset: these are control state, and
    // they are what makes the unreset array above unreachable.
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            count  <= 0;
        end else begin
            case ({store_fire, stored_read})
              2'b11: begin
                wr_ptr <= next_ptr(wr_ptr);
                rd_ptr <= next_ptr(rd_ptr);
              end
              2'b10: begin
                wr_ptr <= next_ptr(wr_ptr);
                count  <= count + 1;
              end
              2'b01: begin
                rd_ptr <= next_ptr(rd_ptr);
                count  <= count - 1;
              end
              default: count <= count;
            endcase
        end
    end

endmodule

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

module act_fifo #(
    parameter WIDTH = 1,
    parameter DEPTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire wr_en,
    input  wire rd_en,
    input [3:0] precision,
    input  wire [WIDTH-1:0] din,
    output reg  [WIDTH-1:0] dout,
    output wire wr_ready,
    output wire full,
    output wire empty,
    output wire underflow,
    // High for the one cycle after a stored_read or bypass_fire has
    // landed fresh data on `dout`. Consumer (systolic PE) uses this to
    // qualify its MAC commit so an unchanged (stale) dout is not treated
    // as a valid operand.
    output wire act_out_valid
);

    reg [WIDTH-1:0] mem [0:DEPTH-1] ;
    localparam integer PTR_WIDTH = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    localparam integer COUNT_WIDTH = $clog2(DEPTH + 1);
    reg [PTR_WIDTH-1:0] wr_ptr;
    reg [PTR_WIDTH-1:0] rd_ptr;
    reg [COUNT_WIDTH-1:0] count;
    localparam [PTR_WIDTH-1:0] LAST_PTR = PTR_WIDTH'(DEPTH - 1);
    
    reg [3:0] precision_count;
    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    wire replay_read = rd_en && (precision_count == 0);
    wire stored_read = replay_read && !empty;
    // wr_ready simplified from `!full || stored_read` to `!full` to break
    // the pre-existing UNOPTFLAT combinational loop through the mm_efficient
    // stall qualifier:
    //   stall → rd_en → replay_read → stored_read → wr_ready →
    //         → activation_write_ready → adapter.activation_ready →
    //         → wr_en_act → act_write_fire → stall
    // The `stored_read` clause used to let the streamer write on the exact
    // cycle a read fires on a FULL FIFO (the slot the read vacates at
    // end-of-cycle would be filled by the write on the same edge). It was
    // an optimization worth ~1 cycle of streamer throughput at prefill drain
    // start. Removing it delays the streamer's first post-prefill write by
    // one cycle in the exact-full-and-reading case; steady-state throughput
    // is unchanged (streamer 1/cycle, array 1/P cycles per row). MAC-order
    // and MAC-count per row are unchanged so bit-identity holds. Bypass path
    // (rd_en && empty && write_fire → dout <= din) is independent of this
    // and unaffected.
    assign wr_ready = !full;
    wire write_fire = wr_en && wr_ready;
    wire bypass_fire = replay_read && empty && write_fire;
    wire store_fire = write_fire && !bypass_fire;
    assign underflow = replay_read && empty && !write_fire;

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
                   "act_fifo underflow at %m t=%0t (stall gating bug: rd_en=1 on empty)",
                   $time);
    end
`endif

    // Storage array + output register, with NO reset -- see the comment
    // in fifo.v. A block RAM (and a compiled SRAM) cannot be
    // asynchronously reset, so a memory inside an async-reset process is not
    // inferrable and synthesis dissolves it into flip-flops. `count` is still
    // reset below, so the unreset contents are unreachable.
    initial dout = {WIDTH{1'b0}};
    always @(posedge clk) begin
        if (bypass_fire)
            dout <= din;
        else if (stored_read)
            dout <= mem[rd_ptr];

        if (store_fire)
            mem[wr_ptr] <= din;
    end

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

    always @(posedge clk or negedge rst) begin
        if (!rst) precision_count <= 0;
        else if (rd_en) begin
            precision_count <= (precision_count < precision-1)? precision_count + 1 : 0;
        end

    end

    // act_out_valid: registered "the value now on dout is fresh from the
    // last delivery" flag. Aligns with dout timing (dout updates on the
    // clock edge after stored_read | bypass_fire). Stays low across
    // cycles where no delivery occurred, so downstream can tell a
    // held-over stale dout from a newly-loaded fresh dout.
    reg act_out_valid_r;
    always @(posedge clk or negedge rst) begin
        if (!rst)
            act_out_valid_r <= 1'b0;
        else
            act_out_valid_r <= stored_read | bypass_fire;
    end
    assign act_out_valid = act_out_valid_r;

endmodule

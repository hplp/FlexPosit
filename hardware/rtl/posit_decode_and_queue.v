// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

module posit_decode_and_queue #(
    parameter integer DATA_WIDTH = 6,   // signed Posit8/es2 exponent fits
    parameter integer DEPTH      = 8,   // one complete max-precision transaction
    parameter integer MAN_WIDTH  = 5,   // maximum Posit8 fraction field
    parameter integer MAX_ES     = 2,
    parameter integer K_WIDTH    = 16
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    stall,       // freeze on array stall

    // serialized posit bitstream + cycle strobes
    input  wire                    w,
    input  wire                    valid,
    input  wire                    cycle0,
    input  wire                    cycle1,
    input  wire                    cycleBeforeLast,
    input  wire                    cycleLast,
    input  wire [3:0]              precision,
    input  wire [1:0]              posit_es,
    // Per-tile K. The decoder self-gates raw-bit consumption at
    // K*precision so it can never drive the weight FIFO past its
    // written budget. When k_reg==0 the gate is disabled (see
    // effective_valid below) so standalone TBs that don't drive a
    // real K keep working.
    input  wire [K_WIDTH-1:0]      k_reg,

    // FIFO output (one beat when re=1)
    output wire [DATA_WIDTH-1:0]   data_out,
    output wire                    out_valid,
    output reg                     fifo_col_re, // Feeding FIfo
    output wire                     next_valid // Feeding next decoder
);
    // Raw-bit counter. Bumps every cycle the decoder actually consumes
    // a raw bit (effective_valid && !stall). Reset on !valid so the
    // count survives an intra-tile stall but zeros between tiles.
    // Width K_WIDTH+3 holds MAX_K * max-precision (128*8=1024) with
    // headroom.
    reg [K_WIDTH+3:0] raw_bits_consumed;
    wire [K_WIDTH+3:0] raw_bits_budget =
        {{3{1'b0}}, k_reg} *
        {{(K_WIDTH-1+3){1'b0}}, precision};

    // Gate raw-bit consumption at K*precision. k_reg==0 disables the
    // gate (standalone TBs). Once count reaches the budget,
    // effective_valid drops so the raw weight FIFO stops being read
    // even if cycle_gen keeps ticking. Information flow is one-way:
    // k_reg -> budget compare -> decoder internal state.
    wire effective_valid =
        valid && ((k_reg == {K_WIDTH{1'b0}}) ||
                  (raw_bits_consumed < raw_bits_budget));

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            raw_bits_consumed <= {(K_WIDTH+4){1'b0}};
        else if (!valid)
            raw_bits_consumed <= {(K_WIDTH+4){1'b0}};
        else if (!stall && effective_valid)
            raw_bits_consumed <= raw_bits_consumed +
                                 {{(K_WIDTH+3){1'b0}}, 1'b1};
    end

`ifndef SYNTHESIS
    // Passive equality assertion. At the falling edge of `valid` (tile
    // boundary) the raw-bit count must equal K*precision. A firing
    // here means the gate's underlying assumption (K*precision-bounded
    // consumption per tile) is violated.
    reg valid_prev;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_prev <= 1'b0;
        else
            valid_prev <= valid;
    end
    always @(posedge clk) begin
        // Guard by k_reg != 0 so standalone TBs that don't drive k_reg
        // (single_pe_es_tb, posit_queue_precision_tb) can tie it to 0
        // and this assertion becomes inert for them.
        if (rst_n && valid_prev && !valid &&
            raw_bits_consumed != {(K_WIDTH+4){1'b0}} &&
            k_reg != {K_WIDTH{1'b0}}) begin
            if (raw_bits_consumed !== raw_bits_budget) begin
                $fatal(1,
                    "posit_decode_and_queue %m: raw-bit count %0d != K*precision %0d (K=%0d P=%0d)",
                    raw_bits_consumed, raw_bits_budget, k_reg, precision);
            end
        end
    end
`endif
    // -------------------------------
    // Decode incoming posit
    // -------------------------------
    wire                   d_sign;
    wire [DATA_WIDTH-1:0]  d_exp_w;
    wire [MAN_WIDTH-1:0]   d_man_w;
    wire [DATA_WIDTH-1:0]  d_man_w_pack;
    wire                   d_zero;
    reg                    _valid;

    assign next_valid = _valid;

    posit_decoder #(
        .DATA_WIDTH (DATA_WIDTH),
        .MAN_WIDTH  (MAN_WIDTH),
        .MAX_ES     (MAX_ES)
    ) u_dec (
        .clk       (clk),
        .rst_n     (rst_n),
        .stall     (stall),
        .w         (w),
        .valid     (effective_valid),
        .cycle0    (cycle0),
        .cycle1    (cycle1),
        .cycleLast (cycleLast),
        .posit_es  (posit_es),
        .sign      (d_sign),
        .exp_out   (d_exp_w),
        .man       (d_man_w),
        .zero      (d_zero)
    );

    // Pack 1-bit fields to DATA_WIDTH
    wire [DATA_WIDTH-1:0] sign_pack = {{(DATA_WIDTH-1){1'b0}}, d_sign};
    wire [DATA_WIDTH-1:0] zero_pack = {{(DATA_WIDTH-1){1'b0}}, d_zero};
    assign d_man_w_pack = {d_man_w, {(DATA_WIDTH-MAN_WIDTH){1'b0}}};

    // FIFO input mapping:
    //  - cycle1: sign is written while the raw Posit is being decoded
    //  - following cycle0: the completed zero/exp/man fields are written
    wire [DATA_WIDTH-1:0] fifo_in0 =
        cycle1    ? sign_pack :
        cycle0 ? zero_pack :
                    {DATA_WIDTH{1'b0}}; // don't care when not writing

    // -------------------------------
    // Start the decoded output stream after the first complete raw Posit.
    // Thereafter the level is refreshed at each raw cycleLast boundary.
    // -------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            _valid        <= 1'b0;
            fifo_col_re <= 1'b0;
        end else if (!stall) begin
            if (cycleBeforeLast) begin
                fifo_col_re <= effective_valid;
            end
            if (cycleLast) begin
                _valid <= effective_valid;
            end
        end
    end

    // -------------------------------
    // FIFO
    // -------------------------------
    beat_fifo_burst #(
        .DATA_WIDTH (DATA_WIDTH),
        .DEPTH      (DEPTH)
    ) u_fifo (
        .clk        (clk),
        .rst_n      (rst_n),
        .stall      (stall),
        .valid      (effective_valid),
        .cycle0     (cycle0),
        .cycle1     (cycle1),
        .cycleLast  (cycleLast),
        .precision  (precision),
        .data_in0   (fifo_in0),       // sign or completed zero flag
        .data_in1   (d_exp_w),        // completed signed exponent
        .data_in2   (d_man_w_pack),   // completed packed fraction
        .re         (_valid),
        .data_out   (data_out),
        .out_valid  (out_valid)
    );

endmodule


module posit_decoder #(
    parameter DATA_WIDTH = 6,
    parameter FIFO_DETPH = 8,
    parameter MAN_WIDTH = 5,
    parameter MAX_ES = 2
)(
    input clk,
    input rst_n,
    input stall,       // freeze on array stall
    input w,
    input valid,
    input cycle0,
    input cycle1,
    input cycleLast,
    input [1:0] posit_es,
    output reg  sign, // sign or zero
    output reg  [DATA_WIDTH-1:0]   exp_out, // exp
    output reg  [MAN_WIDTH-1:0]   man, // mantissa
    output reg zero
);

reg regime_done;
reg regime_sign;
reg [1:0] exp_count;
reg [$clog2(MAN_WIDTH+1)-1:0] man_count;

wire [DATA_WIDTH-1:0] one = {{(DATA_WIDTH-1){1'b0}}, 1'b1};
wire [1:0] effective_es = (posit_es > MAX_ES) ? MAX_ES[1:0] : posit_es;
wire [DATA_WIDTH-1:0] regime_step = one << effective_es;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sign <= 0;
        zero <= 0;
        man <= 0;
        regime_done <= 0;
        regime_sign <= 0;
        exp_count <= 0;
        man_count <= 0;
        exp_out <= 0;
    end
    else if (!stall && valid) begin
        if (cycle0) begin
            sign <= w;
            zero <= !w;
            man <= 0;
            regime_done <= 0;
            regime_sign <= 0;
            exp_count <= 0;
            man_count <= 0;
            exp_out <= 0;
        end
        else if (cycle1) begin
            zero <= !w && zero;
            regime_sign <= w;
            regime_done <= 0;
            exp_count <= 0;
            man_count <= 0;
            // First regime bit establishes k=0 for a run of ones and
            // k=-1 for a run of zeros. Each k step is scaled by 2^es.
            exp_out <= w ? {DATA_WIDTH{1'b0}} : -regime_step;
        end
        else begin
            zero <= !w && zero;
            if (!regime_done) begin
                if (w != regime_sign) begin
                    // Regime terminator is consumed but is not an exponent bit.
                    regime_done <= 1'b1;
                end
                else begin
                    exp_out <= regime_sign ? (exp_out + regime_step)
                                           : (exp_out - regime_step);
                end
            end
            else if (exp_count < effective_es) begin
                // Exponent bits arrive MSB-first. Missing low exponent bits
                // at the end of a short posit are implicitly zero.
                if (w)
                    exp_out <= exp_out + (one << (effective_es - exp_count - 1'b1));
                exp_count <= exp_count + 1'b1;
            end
            else if (man_count < MAN_WIDTH) begin
                // Fraction bits arrive MSB-first and remain left-aligned for
                // the PE's serial shift-add path. Indexed capture is required:
                // shifting the whole register here discards earlier fraction
                // bits when more than one true fraction bit is present.
                man[MAN_WIDTH-1-man_count] <= w;
                man_count <= man_count + 1'b1;
            end
        end
    end
end

endmodule

`timescale 1ns/1ps

module beat_fifo_burst #(
    parameter integer DATA_WIDTH = 3,
    parameter integer DEPTH      = 8,
    parameter integer MAX_PRECISION = 8
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     stall,       // freeze on array stall
    input  wire                     valid,
    input  wire                     cycle0,
    input  wire                     cycle1,
    input  wire                     cycleLast,
    input  wire [3:0]               precision,
    input  wire [DATA_WIDTH-1:0]    data_in0,
    input  wire [DATA_WIDTH-1:0]    data_in1,
    input  wire [DATA_WIDTH-1:0]    data_in2,
    input  wire                     re,
    output reg  [DATA_WIDTH-1:0]    data_out,
    output reg                      out_valid
    );
    
    localparam integer PTR_WIDTH = (DEPTH > 1) ? $clog2(DEPTH) : 1;
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [PTR_WIDTH-1:0] wr_ptr, rd_ptr;
    integer fill_index;

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && cycle0 && re && precision > DEPTH)
            $fatal(
                1,
                "decoded-beat FIFO depth %0d is smaller than precision %0d",
                DEPTH,
                precision
            );
    end
`endif

    // ------------------------------------------------------------------
    // Storage array + output register, in their own process with NO reset.
    // Same rationale as src/fifo.v: a block RAM (and a compiled SRAM) cannot
    // be asynchronously reset, so a memory inside an async-reset process is
    // not inferrable. It also clears
    //
    //   [Synth 8-7137] Register mem_reg / data_out_reg in module
    //       beat_fifo_burst has both Set and reset with same priority.
    //       This may cause simulation mismatches.
    //
    // which came from mixing the zero-fill writes below with the async reset
    // in one process. That warning is a genuine sim-vs-synth divergence risk.
    //
    // Safety: wr_ptr, rd_ptr and out_valid are all still reset below, and
    // nothing consumes data_out except behind out_valid.
    initial data_out = {DATA_WIDTH{1'b0}};
    always @(posedge clk) begin
        if (!stall) begin
            if (re)
                data_out <= mem[rd_ptr];

            if (cycle1&valid) begin
                mem[wr_ptr] <= data_in0;
            end
            else if (cycle0&re) begin
                mem[wr_ptr] <= data_in1;
                mem[(wr_ptr+1) % DEPTH] <= data_in2;
                // PE protocol for any P:
                //   beat 0 sign, beat 1 exponent, beat 2 packed fraction,
                //   optional zero fillers, final beat explicit zero flag.
                // The PE shifts the packed fraction internally on filler
                // cycles, preserving one serial shift-add step per cycle.
                for (fill_index = 2;
                     fill_index < MAX_PRECISION-1;
                     fill_index = fill_index + 1) begin
                    if (fill_index < (precision-2))
                        mem[(wr_ptr+fill_index) % DEPTH] <= {DATA_WIDTH{1'b0}};
                    else if (fill_index == (precision-2))
                        mem[(wr_ptr+fill_index) % DEPTH] <= data_in0;
                end
            end
        end
    end

    // Pointers and the valid flag keep their async reset. These are the
    // control state that makes the unreset array above unreachable.
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            out_valid <= 0;
        end
        else if (!stall) begin
            out_valid <= re;
            if (re)
                rd_ptr <= (rd_ptr + 1) % DEPTH;

            if (cycle1&valid)
                wr_ptr <= (wr_ptr + 1) % DEPTH;
            else if (cycle0&re)   // here re means out_valid
                wr_ptr <= (wr_ptr + precision - 1) % DEPTH;
        end
endmodule

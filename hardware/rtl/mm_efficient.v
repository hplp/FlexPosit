// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

module mm #(
    parameter integer ACT_EBITS      = 4, // kept for futurae activation wiring
    parameter integer ACT_MBITS      = 3, // kept for future activation wiring
    parameter integer EXP_SUM_WIDTH  = 6,
    parameter integer MAN_SUM_WIDTH  = 5,
    parameter integer DATA_WIDTH     = 6,
    parameter integer DEPTH          = 8,
    parameter integer MAN_WIDTH      = 5,
    parameter integer POSIT_EXP_BIAS = 24,
    parameter integer ADDW           = 10,  // widened for MUL_GUARD=3 shift-add path
    parameter ACC_WIDTH = 32,
    parameter N         = 2,
    parameter K         = 2,
    parameter ACT_WIDTH = ACT_EBITS+ACT_MBITS+1,
    parameter integer SCALE_WIDTH = 5,
    parameter ACT_FIFO_DEPTH = 32,
    parameter W_FIFO_DEPTH = 32,
    // Ship-out FP tuple widths (must match systolic defaults)
    parameter integer ACC_EXPW       = 7,
    parameter integer ACC_MANW       = 12,
    parameter integer GRS            = 3,
    parameter integer NORM_EXPW      = ACC_EXPW + 4,
    parameter integer NORM_MANW      = ACC_MANW + GRS,
    parameter integer K_WIDTH        = 16,
    // FP16 ship-out (see bfp_to_fp16). Lane is IEEE 754 binary16.
    // Derived from GRS: FP16_BIAS_DELTA = (ACT_BIAS + ACT_MBITS + GRS + POSIT_EXP_BIAS) - 15
    parameter integer FP16_BIAS_DELTA = 19 + GRS,
    parameter integer OUT_LANE_W     = 16,
    parameter integer OUT_BUS_W      = N * OUT_LANE_W
)(
    input wire clk,
    input wire rst,
    input wire active,
    // "Delivery still in flight" from the activation streamer. Keeps
    // the stall qualifier live during the trailing period so the
    // FIFO-underflow gate holds while activations are still arriving
    // at L>=2. Comes strictly from the frontend; the stall qualifier
    // never receives array-side feedback.
    input wire streamer_pending,
    input wire [K_WIDTH-1:0] k_reg,   // per-tile K, forwarded to systolic
    input wire [3:0] precision,
    input wire [1:0] posit_es,
    // Per-output-column signed power-of-two exponent.
    // Column c is multiplied by 2^col_scale[c].
    input wire signed [SCALE_WIDTH-1:0] col_scale [N-1:0],
    input wire [ACT_WIDTH-1:0] act_din [N-1:0],
    input wire w_din [N-1:0],
    input wire wr_en_act,
    input wire wr_en_w,
    output wire activation_write_ready,
    output wire weight_write_ready,
    output wire activation_fifos_empty,
    output wire weight_fifos_empty,
    output wire stall_out,
    output reg  ingress_error,
    output wire done,
    output wire compute_done,          // 1-cycle pulse when compute finishes (pre-ship)
    // Stallable ship-out interface after compute completes.
    output wire                       out_valid,
    input  wire                       out_ready,
    output wire [$clog2(N)-1:0]       out_row,
    output wire [OUT_BUS_W-1:0]       out_data
);

    wire [ACT_WIDTH-1:0] act_fifo_out [N-1:0];
    wire [N-1:0] act_fifo_valid;
    wire w_fifo_out [N-1:0];
    wire [N-1:0] active_row;
    wire [N-1:0] active_column;
    wire [N-1:0] act_fifo_ready;
    wire [N-1:0] act_fifo_empty;
    wire [N-1:0] act_fifo_underflow;
    wire [N-1:0] w_fifo_ready;
    wire [N-1:0] w_fifo_empty;
    wire [N-1:0] w_fifo_underflow;
    reg active_cfg_d;
    reg [3:0] precision_reg;

    assign activation_write_ready = &act_fifo_ready;
    assign weight_write_ready = &w_fifo_ready;
    assign activation_fifos_empty = &act_fifo_empty;
    assign weight_fifos_empty = &w_fifo_empty;
    wire act_write_fire = wr_en_act && activation_write_ready;
    wire weight_write_fire = wr_en_w && weight_write_ready;

    // Backpressure. If the systolic array wants to pop an operand this cycle
    // (its active_row[i] or active_column[i] is high) and the corresponding
    // FIFO is empty AND no simultaneous write is bypassing new data through,
    // freeze every synchronous element of the compute pipeline for that
    // cycle. Gating is per-cycle and global: any single empty demand stalls
    // the whole array so row/column alignment stays coherent.
    //
    // Qualified by `active || streamer_pending`. `active` is S_RUN from
    // the controller. `streamer_pending` is the activation-streamer's
    // replay-busy signal and stays high while a delivery is still in
    // flight, keeping the stall gate live during the trailing period
    // when the array is finishing MACs whose activations are still
    // being delivered late at L>=2. Information flow is one-way:
    // streamer -> stall -> array. Nothing from the array feeds back
    // into the qualifier (any such feedback path produced a delta-cycle
    // hang in both iverilog and Verilator during earlier attempts).
    //
    // The `!*_write_fire` qualifier preserves the FIFO's zero-latency
    // bypass path (fifo.v: rd_en && empty && write_fire => dout <= din).
    // Without it, gating rd_en converts bypass into a store-then-read and
    // delivers the first beat one cycle late — off-by-one desyncs compute
    // even at L=1, where bypass fires on the S_RUN entry cycle.
    wire stall = (active || streamer_pending) &&
                 ((|(active_row & act_fifo_empty) & !act_write_fire) |
                  (|(active_column & w_fifo_empty) & !weight_write_fire));
    assign stall_out = stall;

    // Sticky interface status. All rows/columns accept atomically; a rejected
    // vector or an attempted read without data is never allowed to become a
    // silent partial transaction.
    always @(posedge clk or negedge rst) begin
        if (!rst)
            ingress_error <= 1'b0;
        else if ((wr_en_act && !activation_write_ready) ||
                 (wr_en_w && !weight_write_ready) ||
                 (|act_fifo_underflow) ||
                 (|w_fifo_underflow))
            ingress_error <= 1'b1;
    end

    // Hold the activation replay cadence for the complete tile. The systolic
    // block independently latches the same external precision at its start
    // edge for serializer/decoder control.
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            active_cfg_d <= 1'b0;
            precision_reg <= 4'd4;
        end else begin
            active_cfg_d <= active;
            if (active && !active_cfg_d)
                precision_reg <= precision;
        end
    end

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : row_fifos
            act_fifo #(.WIDTH(ACT_WIDTH), .DEPTH(ACT_FIFO_DEPTH)) act_fifo_inst (
                .clk(clk),
                .rst(rst),
                .precision(precision_reg),
                .wr_en(act_write_fire),
                .rd_en(active_row[i] && !stall),
                .din(act_din[i]),
                .dout(act_fifo_out[i]),
                .wr_ready(act_fifo_ready[i]),
                .full(),
                .empty(act_fifo_empty[i]),
                .underflow(act_fifo_underflow[i]),
                .act_out_valid(act_fifo_valid[i])
            );

            fifo #(.WIDTH(1), .DEPTH(W_FIFO_DEPTH)) w_fifo_inst (
                .clk(clk),
                .rst(rst),
                .wr_en(weight_write_fire),
                .rd_en(active_column[i] && !stall),
                .din(w_din[i]),
                .dout(w_fifo_out[i]),
                .wr_ready(w_fifo_ready[i]),
                .full(),
                .empty(w_fifo_empty[i]),
                .underflow(w_fifo_underflow[i])
            );
        end
    endgenerate

    systolic #(
        .ACT_EBITS(ACT_EBITS),
        .ACT_MBITS(ACT_MBITS),
        .EXP_SUM_WIDTH(EXP_SUM_WIDTH),
        .ACC_EXPW(ACC_EXPW),
        .ACC_MANW(ACC_MANW),
        .GRS(GRS),
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(DEPTH),
        .MAN_WIDTH(MAN_WIDTH),
        .POSIT_EXP_BIAS(POSIT_EXP_BIAS),
        .ADDW(ADDW),
        .ACC_WIDTH(ACC_WIDTH),
        .N(N),
        .SCALE_WIDTH(SCALE_WIDTH),
        .NORM_EXPW(NORM_EXPW),
        .NORM_MANW(NORM_MANW),
        .FP16_BIAS_DELTA(FP16_BIAS_DELTA),
        .OUT_LANE_W(OUT_LANE_W),
        .OUT_BUS_W(OUT_BUS_W),
        .K_WIDTH(K_WIDTH)
    ) systolic_inst (
        .clk(clk),
        .rst(rst),
        .stall(stall),
        .active(active),
        .k_reg_in(k_reg),
        .precision(precision),
        .posit_es(posit_es),
        .col_scale(col_scale),
        .act_in(act_fifo_out),
        .act_valid_in(act_fifo_valid),
        .w_in(w_fifo_out),
        .done(done),
        .compute_done_out(compute_done),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_row(out_row),
        .out_data(out_data),
        .active_row(active_row),
        .active_column(active_column)
    );

endmodule

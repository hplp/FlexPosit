// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

// Combinational power-of-two rescale for the normalized ship-out tuple.
// scale is signed two's complement:
//     value_out = value_in * 2^scale
// No mantissa datapath is needed unless the exponent saturates.
module bfp_pow2_rescale #(
    parameter integer EXP_WIDTH   = 10,
    parameter integer MAN_WIDTH   = 15,
    parameter integer SCALE_WIDTH = 5
)(
    input  wire                          in_sign,
    input  wire [EXP_WIDTH-1:0]          in_exp,
    input  wire [MAN_WIDTH-1:0]          in_man,
    input  wire signed [SCALE_WIDTH-1:0] scale,
    output wire                          out_sign,
    output wire [EXP_WIDTH-1:0]          out_exp,
    output wire [MAN_WIDTH-1:0]          out_man
);
    localparam integer SUM_WIDTH = EXP_WIDTH + 2;

    wire input_zero =
        (in_exp == {EXP_WIDTH{1'b0}}) &&
        (in_man == {MAN_WIDTH{1'b0}});

    wire signed [SUM_WIDTH-1:0] exp_extended =
        $signed({2'b00, in_exp});
    wire signed [SUM_WIDTH-1:0] scale_extended =
        {{(SUM_WIDTH-SCALE_WIDTH){scale[SCALE_WIDTH-1]}}, scale};
    wire signed [SUM_WIDTH-1:0] scaled_exp_sum =
        exp_extended + scale_extended;
    wire signed [SUM_WIDTH-1:0] max_exponent =
        $signed({2'b00, {EXP_WIDTH{1'b1}}});

    wire underflow = scaled_exp_sum < 0;
    wire overflow  = scaled_exp_sum > max_exponent;

    assign out_sign = input_zero || underflow ? 1'b0 : in_sign;
    assign out_exp =
        input_zero || underflow ? {EXP_WIDTH{1'b0}} :
        overflow                ? {EXP_WIDTH{1'b1}} :
                                  scaled_exp_sum[EXP_WIDTH-1:0];
    assign out_man =
        input_zero || underflow ? {MAN_WIDTH{1'b0}} :
        overflow                ? {MAN_WIDTH{1'b1}} :
                                  in_man;
endmodule

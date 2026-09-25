// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

// ============================================================================
// bfp_to_fp16
//
// Converts the normalized+rescaled ship-out tuple to IEEE 754 binary16 (FP16).
// Combinational. Ship-out format on input:
//   value = (-1)^sign * (1 + in_man / 2^IN_MANW) * 2^(in_exp - BIAS_OFFSET)
// zero is encoded as in_exp == 0 && in_man == 0 (sign forced 0 upstream).
//
// Output IEEE 754 half:
//   value = (-1)^sign * (1 + fp16_man / 2^10) * 2^(fp16_exp - 15)
// Policy:
//   - Round-to-nearest-even (RNE) from IN_MANW-bit fraction to 10-bit
//   - Saturate to max normal (exp=30, man=all-ones) on overflow (never emit Inf)
//   - Flush to zero on underflow (never emit subnormals)
//   - Never emit NaN (upstream never produces NaN)
// ============================================================================
module bfp_to_fp16 #(
    parameter integer IN_EXPW     = 11,
    parameter integer IN_MANW     = 15,
    // BIAS_DELTA = BIAS_OFFSET_INTERNAL - FP16_EXP_BIAS
    //   = (ACT_BIAS + ACT_MBITS + GRS + POSIT_EXP_BIAS) - 15
    //   = (7 + 3 + 3 + 24) - 15 = 22 at defaults
    parameter integer BIAS_DELTA  = 22
)(
    input  wire                 in_sign,
    input  wire [IN_EXPW-1:0]   in_exp,
    input  wire [IN_MANW-1:0]   in_man,
    output wire [15:0]          fp16
);
    localparam integer FP16_MANW = 10;
    localparam integer FP16_EXPW = 5;
    localparam integer FP16_MAX_BIASED_EXP = 30;  // 31 is reserved for Inf/NaN
    localparam integer ROUND_SHIFT = IN_MANW - FP16_MANW;  // 5 at defaults

    // Signed exponent arithmetic to catch under/overflow cleanly.
    localparam integer EXPS_W = IN_EXPW + 2;

    wire input_zero =
        (in_exp == {IN_EXPW{1'b0}}) &&
        (in_man == {IN_MANW{1'b0}});

    // Rebias to FP16 biased exponent, before rounding-induced carry.
    wire signed [EXPS_W-1:0] in_exp_ext =
        $signed({{2{1'b0}}, in_exp});
    wire signed [EXPS_W-1:0] delta_ext =
        $signed({{(EXPS_W - $clog2(BIAS_DELTA+1)){1'b0}},
                 BIAS_DELTA[$clog2(BIAS_DELTA+1)-1:0]});
    wire signed [EXPS_W-1:0] fp16_exp_pre_round = in_exp_ext - delta_ext;

    // RNE rounding from IN_MANW to FP16_MANW.
    wire [FP16_MANW-1:0] man_top = in_man[IN_MANW-1 -: FP16_MANW];
    wire round_bit = in_man[ROUND_SHIFT-1];
    wire sticky_bit = |in_man[ROUND_SHIFT-2:0];
    wire round_up = round_bit & (sticky_bit | man_top[0]);
    wire [FP16_MANW:0] man_rounded = {1'b0, man_top} + {{FP16_MANW{1'b0}}, round_up};
    wire man_carry = man_rounded[FP16_MANW];
    wire [FP16_MANW-1:0] man_after_round = man_rounded[FP16_MANW-1:0];

    // Rounding carry bumps the exponent by 1 (mantissa becomes all zeros).
    wire signed [EXPS_W-1:0] fp16_exp_post =
        fp16_exp_pre_round + {{(EXPS_W-1){1'b0}}, man_carry};

    wire underflow = (fp16_exp_post <= 0);
    wire overflow  = (fp16_exp_post >= FP16_MAX_BIASED_EXP + 1);

    wire                 out_sign;
    wire [FP16_EXPW-1:0] out_exp;
    wire [FP16_MANW-1:0] out_man;

    assign out_sign = (input_zero | underflow) ? 1'b0 : in_sign;
    assign out_exp  =
        (input_zero | underflow) ? {FP16_EXPW{1'b0}} :
        overflow                 ? FP16_MAX_BIASED_EXP[FP16_EXPW-1:0] :
                                   fp16_exp_post[FP16_EXPW-1:0];
    assign out_man  =
        (input_zero | underflow) ? {FP16_MANW{1'b0}} :
        overflow                 ? {FP16_MANW{1'b1}} :
                                   man_after_round;

    assign fp16 = {out_sign, out_exp, out_man};
endmodule

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

// ============================================================================
// bfp_normalize
//
// Ship-out normalization for a pe_bfp_acc tap. The accumulator now carries a
// GRS-extended widened mantissa: {in_sign, in_man} is an (ACC_MANW+GRS+1)-bit
// two's-complement value whose ULP is 2^-GRS relative to the ACC_MANW-integer
// view. This block finds the true leading 1 across the widened magnitude,
// left-shifts so the leading 1 sits at bit position ACC_MANW+GRS, drops the
// implicit-1, and folds the shift amount into a widened out_exp.
//
// Numeric identity preserved end-to-end:
//   value = signed({in_sign,in_man}) * 2^(in_exp - INTERNAL_BIAS - GRS)
//         = (-1)^out_sign
//           * (1 + out_man / 2^(ACC_MANW+GRS))
//           * 2^(out_exp - INTERNAL_BIAS - GRS)
// so the external decoder uses one fixed BIAS_OFFSET = INTERNAL_BIAS + GRS.
//
// Zero input yields all-zero output.
// ============================================================================
module bfp_normalize #(
    parameter integer ACC_EXPW  = 5,
    parameter integer ACC_MANW  = 12,
    parameter integer GRS       = 3,
    parameter integer MAN_TOT   = ACC_MANW + GRS,          // widened mantissa width
    parameter integer NORM_EXPW = ACC_EXPW + 4             // absorbs up to +MAN_TOT shift
)(
    input  wire                   in_sign,
    input  wire [ACC_EXPW-1:0]    in_exp,
    input  wire [MAN_TOT-1:0]     in_man,        // {in_sign,in_man} is 2's-comp (MAN_TOT+1) bit
    output wire                   out_sign,
    output wire [NORM_EXPW-1:0]   out_exp,
    output wire [MAN_TOT-1:0]     out_man         // implicit-leading-1 dropped
);
    localparam integer LPW = $clog2(MAN_TOT+2);  // width to hold lead_pos in [0, MAN_TOT]

    // Sign-extend to (MAN_TOT+2) bits so negation of the -2^MAN_TOT corner
    // case fits without overflow.
    wire signed [MAN_TOT+1:0] sv_ext = {{2{in_sign}}, in_man};
    wire signed [MAN_TOT+1:0] mag_sx = in_sign ? -sv_ext : sv_ext;
    wire        [MAN_TOT:0]   mag    = mag_sx[MAN_TOT:0];  // (MAN_TOT+1)-bit unsigned magnitude

    // Priority encoder: find highest set bit in mag (positions 0..MAN_TOT).
    integer  i;
    reg [LPW-1:0] lead_pos;
    reg           is_zero;
    always @* begin
        lead_pos = {LPW{1'b0}};
        is_zero  = 1'b1;
        for (i = MAN_TOT; i >= 0; i = i - 1) begin
            if (is_zero && mag[i]) begin
                lead_pos = i[LPW-1:0];
                is_zero  = 1'b0;
            end
        end
    end

    // Left-shift so the leading 1 sits at bit MAN_TOT, then drop it.
    wire [MAN_TOT:0]   shifted  = mag << (MAN_TOT[LPW-1:0] - lead_pos);
    wire [MAN_TOT-1:0] man_norm = shifted[MAN_TOT-1:0];

    // Widen exponent, add the shift amount.
    wire [NORM_EXPW-1:0] in_exp_ext = {{(NORM_EXPW-ACC_EXPW){1'b0}}, in_exp};
    wire [NORM_EXPW-1:0] exp_norm   = in_exp_ext + {{(NORM_EXPW-LPW){1'b0}}, lead_pos};

    assign out_sign = is_zero ? 1'b0                : in_sign;
    assign out_exp  = is_zero ? {NORM_EXPW{1'b0}}   : exp_norm;
    assign out_man  = is_zero ? {MAN_TOT{1'b0}}     : man_norm;
endmodule

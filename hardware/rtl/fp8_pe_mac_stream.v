// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

// ============================================================================
// Small signed comb adder (W-bit + W-bit -> (W+1)-bit)
// ============================================================================
module shared_adder_comb #(
  parameter integer W = 8
)(
  input  wire [W-1:0] a,
  input  wire [W-1:0] b,
  output wire [W:0]   y
);
  assign y = {a[W-1], a} + {b[W-1], b};
endmodule

// ============================================================================
// Block-floating accumulator (unsigned magnitude input, explicit sign)
// - GRS precision below the original mantissa LSB
// - 1-cycle input capture on in_valid (mul_done)
// - Compute/accumulate uses ONLY registered inputs
// - acc_valid pulses HIGH for exactly 1 cycle on commit (the cycle after capture)
// - Signed exponent alignment uses round-to-nearest-even
// ============================================================================
module pe_bfp_acc #(
  parameter integer IN_EXPW   = 5,   // width of in_exp
  parameter integer IN_MANW   = 3,   // logical integer mantissa width
  parameter integer MUL_GUARD = 0,   // sub-integer guard bits below in_mag LSB
  parameter integer ACC_EXPW  = 4,   // accumulator exponent width (block floating)
  parameter integer ACC_MANW  = 3,   // exposed accumulator mantissa width (integer)
  parameter integer GRS       = 3    // guard/round/sticky fractional bits below LSB
)(
  input  wire                              clk,
  input  wire                              rst_n,
  input  wire                              stall,       // freeze all state on array stall
  input  wire                              clear_acc,   // pulse to clear accumulator

  input  wire                              in_valid,    // 1-cycle pulse (mul_done)
  input  wire                              in_zero,     // zero-product token; commit without changing state
  input  wire                              in_sign,     // 1=negative term
  input  wire [IN_EXPW-1:0]                in_exp,      // unbiased exponent
  input  wire [IN_MANW+1+MUL_GUARD:0]      in_mag,      // UNSIGNED magnitude: (IN_MANW+2) integer bits + MUL_GUARD sub-integer bits

  output reg                      acc_sign,    // accumulator sign (MSB of widened mantissa)
  output reg  [ACC_EXPW-1:0]      acc_exp,     // accumulator block exponent
  output reg  [ACC_MANW+GRS-1:0]  acc_man,     // widened magnitude (ACC_MANW int bits + GRS frac)
  output reg                      acc_valid    // 1-cycle pulse when a new sum is committed
);
  // -----------------------------
  // One-cycle input capture stage
  // -----------------------------
  reg                              r_in_zero;
  reg                              r_in_sign;
  reg [IN_EXPW-1:0]                r_in_exp;
  reg [IN_MANW+1+MUL_GUARD:0]      r_in_mag;
  reg                              proc_en;   // delayed in_valid (commit enable)

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_in_zero <= 1'b0;
      r_in_sign <= 1'b0;
      r_in_exp  <= {IN_EXPW{1'b0}};
      r_in_mag  <= {(IN_MANW+2+MUL_GUARD){1'b0}};
      proc_en   <= 1'b0;
    end else if (!stall) begin
      proc_en <= in_valid;  // arm commit for next cycle
      if (in_valid) begin
        r_in_zero <= in_zero;
        r_in_sign <= in_sign;
        r_in_exp  <= in_exp;
        r_in_mag  <= in_mag;
      end
    end
  end

  // -----------------------------
  // Widened internal state.
  //   EXT_W = ACC_MANW + GRS. Widened mantissa is (EXT_W+1)-bit signed 2's comp:
  //     [EXT_W]         = sign
  //     [EXT_W-1:GRS]   = ACC_MANW integer magnitude bits (matching pre-GRS view)
  //     [GRS-1:0]       = guard/round/sticky fractional bits below original LSB
  // Numeric value: value = signed(acc_m_ext) * 2^(acc_exp - INTERNAL_BIAS - GRS).
  // The +GRS scale is folded into the output decode bias (see bfp_normalize.v).
  // -----------------------------
  localparam integer WEXT     = ACC_MANW;
  localparam integer EXT_W    = ACC_MANW + GRS;
  localparam integer INW      = IN_MANW + 2 + MUL_GUARD;
  // The product magnitude carries MUL_GUARD sub-integer bits below the
  // classical FP8 LSB. Insertion into the accumulator shifts left by
  // (GRS - MUL_GUARD) so those sub-integer bits sit at (or below) the
  // accumulator's GRS boundary. When MUL_GUARD == GRS the shift is 0.
  localparam integer INS_GRS  = GRS - MUL_GUARD;

  reg  signed [EXT_W:0] acc_m_ext;
  reg                   acc_zero;

  // Widen input term to (EXT_W+1) bits, then left-shift by INS_GRS. The
  // arithmetic shift-form avoids a zero-width replicate `{0{1'b0}}` when
  // MUL_GUARD == GRS (INS_GRS == 0), which some tools mis-handle.
  wire signed [EXT_W:0] in_mag_wide =
      $signed({{(EXT_W+1-INW){1'b0}}, r_in_mag}) <<< INS_GRS;
  wire signed [EXT_W:0] in_term_signed = r_in_sign ? -in_mag_wide : in_mag_wide;

  // Exponent difference acc - in (signed)
  wire signed [ACC_EXPW:0] acc_exp_ext = $signed({1'b0, acc_exp});
  wire signed [ACC_EXPW:0] in_exp_ext  = $signed({{(ACC_EXPW-IN_EXPW+1){1'b0}}, r_in_exp});
  wire signed [ACC_EXPW:0] exp_diff_ai = acc_exp_ext - in_exp_ext; // + => acc>=in
  wire [ACC_EXPW-1:0] in_exp_resized = ACC_EXPW'(r_in_exp);

  // Signed right shift with round-to-nearest-even on magnitude. Treating the
  // sticky bit as a numeric LSB made every tiny negative term contribute -1
  // accumulator unit after a large exponent cancellation. RNE is symmetric:
  // sub-half-ULP terms disappear and ties round to an even retained LSB.
  function signed [EXT_W:0] arith_shr_rne;
    input signed [EXT_W:0] val;
    input integer          sh;
    integer i;
    reg                    val_sign;
    reg [EXT_W:0]          magnitude;
    reg [EXT_W:0]          quotient;
    reg                    guard;
    reg                    sticky;
    begin
      if (sh <= 0) begin
        arith_shr_rne = val;
      end else begin
        val_sign = val[EXT_W];
        magnitude = val_sign ? (~val + 1'b1) : val;
        quotient = (sh > (EXT_W+1)) ? {(EXT_W+1){1'b0}}
                                     : (magnitude >> sh);
        guard = 1'b0;
        sticky = 1'b0;
        for (i = 0; i <= EXT_W; i = i + 1) begin
          if (i == (sh-1))
            guard = magnitude[i];
          else if (i < (sh-1))
            sticky = sticky | magnitude[i];
        end
        if (guard && (sticky || quotient[0]))
          quotient = quotient + 1'b1;
        arith_shr_rne = val_sign ? -$signed(quotient)
                                 :  $signed(quotient);
      end
    end
  endfunction

  // Align to larger exponent
  reg  [ACC_EXPW-1:0]   work_exp;
  reg  signed [EXT_W:0] acc_aligned, in_aligned;

  always @* begin
    if (acc_zero) begin
      work_exp    = in_exp_resized;
      acc_aligned = { (EXT_W+1){1'b0} };
      in_aligned  = in_term_signed;
    end else if (exp_diff_ai >= 0) begin
      // acc's exp >= in's: shift in down (into GRS bits, precision preserved)
      work_exp    = acc_exp;
      acc_aligned = acc_m_ext;
      in_aligned  = arith_shr_rne(in_term_signed, exp_diff_ai);
    end else begin
      // in's exp > acc's: shift acc down
      work_exp    = in_exp_resized;
      acc_aligned = arith_shr_rne(acc_m_ext, -exp_diff_ai);
      in_aligned  = in_term_signed;
    end
  end

  // Add with 1 sign-extension bit, then 1-bit signed RNE renormalize on
  // overflow. With a one-bit shift the dropped bit is an exact half-way bit;
  // increment only when the retained magnitude is odd (ties to even).
  wire signed [EXT_W+1:0] sum_ext =
      $signed({acc_aligned[EXT_W], acc_aligned}) +
      $signed({in_aligned [EXT_W], in_aligned });

  wire ovf = (sum_ext[EXT_W+1] != sum_ext[EXT_W]);
  wire                    sum_ext_sign = sum_ext[EXT_W+1];
  wire [EXT_W+1:0]        sum_ext_mag =
      sum_ext_sign ? (~sum_ext + 1'b1) : sum_ext;
  wire [EXT_W+1:0]        sum_half_q = sum_ext_mag >> 1;
  wire                    sum_half_round = sum_ext_mag[0] & sum_half_q[0];
  wire [EXT_W+1:0]        sum_half_rne_mag =
      sum_half_q + {{(EXT_W+1){1'b0}}, sum_half_round};
  wire signed [EXT_W+1:0] sum_half_rne =
      sum_ext_sign ? -$signed(sum_half_rne_mag)
                   :  $signed(sum_half_rne_mag);
  wire signed [EXT_W:0]   sum_norm =
      ovf ? sum_half_rne[EXT_W:0]
          : sum_ext[EXT_W:0];
  wire [ACC_EXPW-1:0]     exp_norm =
      ovf ? (work_exp + {{(ACC_EXPW-1){1'b0}},1'b1}) : work_exp;

  wire res_zero = (sum_norm == { (EXT_W+1){1'b0} });
  wire res_sign = sum_norm[EXT_W];

  // Extract widened magnitude view (all bits below the sign).
  wire [EXT_W-1:0] acc_man_view = sum_norm[EXT_W-1 : 0];

  // State update (commit on proc_en) + 1-cycle acc_valid pulse.
  // clear_acc is intentionally NOT stall-gated. It comes from start_pulse
  // (active rising edge) which uses ungated config-latch state, so it can
  // fire on the very cycle stall is asserted. Gating it there dropped the
  // clear and let the next tile inherit the previous tile's accumulator.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc_sign  <= 1'b0;
      acc_exp   <= {ACC_EXPW{1'b0}};
      acc_man   <= {(ACC_MANW+GRS){1'b0}};
      acc_m_ext <= { (EXT_W+1){1'b0} };
      acc_valid <= 1'b0;
      acc_zero  <= 1'b1;
    end else if (clear_acc) begin
      acc_sign  <= 1'b0;
      acc_exp   <= {ACC_EXPW{1'b0}};
      acc_man   <= {(ACC_MANW+GRS){1'b0}};
      acc_m_ext <= { (EXT_W+1){1'b0} };
      acc_zero  <= 1'b1;
      acc_valid <= 1'b0;
    end else if (!stall) begin
      if (in_valid) begin
        acc_valid <= 1'b0;
      end else if (proc_en) begin
        if (!r_in_zero) begin
          acc_sign  <= res_zero ? 1'b0 : res_sign;
          acc_exp   <= res_zero ? {ACC_EXPW{1'b0}} : exp_norm;
          acc_man   <= res_zero ? {(ACC_MANW+GRS){1'b0}} : acc_man_view;
          acc_m_ext <= sum_norm;
          acc_zero  <= res_zero;
        end
        // A zero product is still a completed pipeline transaction. Preserve
        // the accumulator verbatim, but pulse valid so array completion timing
        // remains identical when the final product of a dot product is zero.
        acc_valid <= 1'b1;
      end else begin
        acc_valid <= 1'b0;
      end
    end
  end
endmodule



// ============================================================================
// PIPE — tiny register slice forwarding _valid/_data_in/_act
// - Latches act only when cycleLast=1
// ============================================================================
module pe_pipe #(
  parameter integer ACT_EBITS   = 4,
  parameter integer ACT_MBITS   = 3,
  parameter integer DATA_WIDTH  = 3
)(
  input  wire                         clk,
  input  wire                         rst_n,
  input  wire                         stall,
  input  wire                         valid,
  input  wire                         act_valid,
  input  wire                         cycleLast,
  input  wire [DATA_WIDTH-1:0]        data_in,
  input  wire [ACT_EBITS+ACT_MBITS:0] act,

  output reg                          o_valid,
  output reg                          o_act_valid,
  output reg  [DATA_WIDTH-1:0]        o_data_in,
  output reg  [ACT_EBITS+ACT_MBITS:0] o_act
);
  // Same act delivery-window tracker as in pe_fp8_compute. Set on any
  // act_valid pulse in the current cycleLast window; reset on cycleLast
  // to the current-cycle input value. This yields the "was there a
  // delivery in this window" level signal, which we latch as o_act_valid
  // (so downstream PEs east see the correct freshness for the act value
  // we're propagating, not a stray zero from the FIFO's 1-cycle pulse).
  reg act_seen;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      act_seen <= 1'b0;
    else if (!stall) begin
      if (cycleLast)
        act_seen <= act_valid;
      else if (act_valid)
        act_seen <= 1'b1;
    end
  end
  wire effective_act_valid = act_valid | act_seen;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      o_valid     <= 1'b0;
      o_act_valid <= 1'b0;
      o_data_in   <= {DATA_WIDTH{1'b0}};
      o_act       <= { (ACT_EBITS+ACT_MBITS+1){1'b0} };
    end else if (!stall) begin
      o_valid   <= valid;
      o_data_in <= data_in;
      if (cycleLast) begin
        o_act       <= act;
        o_act_valid <= effective_act_valid;
      end
    end
  end
endmodule

// ============================================================================
// COMPUTE — FP8-like MUL (pure compute; no pass-through regs inside)
// - Keeps cycle0/1/2/Last protocol
// - Outputs out_sign/out_exp_sum_r/out_val_r and mul_done
// - Preserves original gating (valid | _valid) via pipe_valid
// ============================================================================
module pe_fp8_compute #(
  parameter integer ACT_EBITS      = 4,
  parameter integer ACT_MBITS      = 3,    // fp8 mantissa width
  parameter integer MUL_GUARD      = 3,    // sub-integer guard bits below FP8 significand LSB
  parameter integer EXP_SUM_WIDTH  = 6,    // exponent sum width produced at cycle1
  parameter integer DATA_WIDTH     = 6,    // signed Posit8/es2 exponent width
  parameter integer ADDW           = 10,   // signed adder width; must cover widened mantissa sum
  parameter integer POSIT_EXP_BIAS = 24    // keeps P<=8/es<=2 exponents non-negative
)(
    input  wire                                 clk,
    input  wire                                 rst_n,
    input  wire                                 stall,

    // control / timing
    input  wire                                 valid,
    input  wire                                 pipe_valid,   // from pe_pipe (aka _valid)
    // act_valid is used only to gate mul_done.
    // A MAC only commits when BOTH operand streams are architecturally
    // fresh at cycleLast — matching the weight path (which already
    // required `valid`). The intermediate compute regs (shifted_act etc.)
    // still update on `valid` alone; they may compute a garbage product
    // from a stale act, but with mul_done gated the accumulator never
    // sees that product.
    input  wire                                 act_valid,
    input  wire                                 cycle0,
    input  wire                                 cycle1,
    input  wire                                 cycle2,
    input  wire                                 cycleLast,

    // data in (bit-serial decoded beat + activation)
    input  wire [DATA_WIDTH-1:0]                data_in,
    input  wire [ACT_EBITS+ACT_MBITS:0]         act,   // {sign, exp, man}

    // product completion
    output reg                                  mul_done,
    output reg                                  out_zero,

    // product fields (per product)
    output reg                                  out_sign,
    output reg  [EXP_SUM_WIDTH-1:0]             out_exp_sum_r,   // valid at cycle1
    output reg  [ACT_MBITS+1+MUL_GUARD:0]       out_val_r        // unsigned magnitude with MUL_GUARD sub-integer bits
);

  // Widened significand width used inside the shift-add loop. The FP8 activation
  // has ACT_MBITS+1 integer bits (implicit-1 for normals); MUL_GUARD extra low
  // bits let progressive right shifts by 4 or 5 produce nonzero contributions
  // that a 4-bit significand would truncate to zero.
  localparam integer MUL_MAG_W = ACT_MBITS + 1 + MUL_GUARD;
  // Local shared adder (mantissa work)
  reg  signed [ADDW-1:0] add_a, add_b;
  wire signed [ADDW:0]   add_y;
  reg  signed [ADDW:0]   add_y_q;

  reg [MUL_MAG_W-1:0]    shifted_act;    // implicit 1 prefixed, widened by MUL_GUARD
  reg [DATA_WIDTH-1:0]   shifted_man;    // running weight mantissa shift

  wire [ACT_EBITS-1:0] act_exp_field =
      act[ACT_EBITS+ACT_MBITS-1 : ACT_MBITS];
  wire [ACT_MBITS-1:0] act_man_field = act[ACT_MBITS-1:0];
  wire act_exp_zero = (act_exp_field == {ACT_EBITS{1'b0}});
  wire act_exp_reserved = &act_exp_field;
  wire act_is_zero = act_exp_zero && (act_man_field == {ACT_MBITS{1'b0}});

  // FP8 policy used by the reference model:
  //   exp=0, man=0 : zero
  //   exp=0, man!=0: subnormal (effective exponent=1, no hidden bit)
  //   exp=all-ones : saturate to max finite (exp=all-ones-1, man=all-ones)
  wire [ACT_EBITS-1:0] act_exp_effective =
      act_exp_reserved ? ({ACT_EBITS{1'b1}} - 1'b1) :
      act_exp_zero     ? {{(ACT_EBITS-1){1'b0}}, 1'b1} :
                         act_exp_field;
  // Widened significand: the classical (ACT_MBITS+1) integer bits followed by
  // MUL_GUARD zero low bits. For the reserved (max-finite) case, saturation
  // keeps the top (ACT_MBITS+1) bits at all-ones and pads the guard bits with
  // zero so the numeric value is (2^(ACT_MBITS+1)-1) << 0 in the widened
  // representation, i.e. still 15 in classical FP8 units.
  wire [MUL_MAG_W-1:0] act_temp =
      act_exp_reserved ? { {(ACT_MBITS+1){1'b1}}, {MUL_GUARD{1'b0}} } :
                         { ~act_exp_zero, act_man_field, {MUL_GUARD{1'b0}} };
  localparam [ADDW-1:0] POSIT_EXP_BIAS_W = POSIT_EXP_BIAS[ADDW-1:0];

  shared_adder_comb #(.W(ADDW)) U_ADD (
    .a(add_a[ADDW-1:0]),
    .b(add_b[ADDW-1:0]),
    .y(add_y)
  );

  // Combinational adder inputs (depend on cycle phase)
  always @* begin
    if (cycle0) begin
      add_a = {ADDW{1'b0}};
      add_b = {ADDW{1'b0}};
    end else if (cycle1) begin
      // Exponent sum path. A fixed internal posit bias keeps every P=4,
      // es<=2 product exponent non-negative without changing its value.
      add_a = {{(ADDW-ACT_EBITS){1'b0}}, act_exp_effective}
              + POSIT_EXP_BIAS_W;
      add_b = {{(ADDW - DATA_WIDTH){data_in[DATA_WIDTH-1]}}, data_in};
    end else if (cycle2) begin
      // mantissa path: first bit uses current LSB of data_in
      add_a = {{(ADDW-MUL_MAG_W){1'b0}}, act_temp};
      add_b = data_in[DATA_WIDTH-1] ? {{(ADDW-MUL_MAG_W){1'b0}}, shifted_act}
                                    : {ADDW{1'b0}};
    end else begin
      // later mantissa bits use shifted_man MSB
      add_a = add_y_q[ADDW-1:0]; // accumulate running sum
      add_b = shifted_man[DATA_WIDTH-1] ? {{(ADDW-MUL_MAG_W){1'b0}}, shifted_act}
                                        : {ADDW{1'b0}};
    end
  end

  // act delivery-window tracker. `act_valid` from the FIFO is a
  // one-cycle pulse (registered from stored_read), which does not align
  // in general with pe_cycLast (FIFO's precision_count and cycle_gen's
  // cnt run on independent phases). Track "did a delivery happen in the
  // interval (prev_cycleLast, current_cycleLast]" as a set-hold flag,
  // reset on cycleLast with the current-cycle input value so that a
  // delivery landing exactly at cycleLast counts for the next window.
  reg act_seen;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      act_seen <= 1'b0;
    else if (!stall) begin
      if (cycleLast)
        act_seen <= act_valid;
      else if (act_valid)
        act_seen <= 1'b1;
    end
  end
  wire effective_act_valid = act_valid | act_seen;

  // mul_done pulse on final beat (combinational). Both operands must be
  // architecturally valid at cycleLast — weight-side via `valid` (from
  // decoder), activation-side via `effective_act_valid` (any delivery
  // in the current cycleLast window).
  wire mul_done_temp = valid & effective_act_valid & cycleLast;

  // Registers for product fields — (valid | _valid) gating
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_exp_sum_r <= {EXP_SUM_WIDTH{1'b0}};
      out_val_r     <= {(ACT_MBITS+2+MUL_GUARD){1'b0}};
      out_sign      <= 1'b0;
      add_y_q       <= { (ADDW+1){1'b0} };
      shifted_act   <= { MUL_MAG_W{1'b0} };
      shifted_man   <= { DATA_WIDTH{1'b0} };
      mul_done      <= 1'b0;
      out_zero      <= 1'b0;
    end else if (!stall && (valid | pipe_valid)) begin
      mul_done <= mul_done_temp;
      if (valid) begin
        add_y_q <= add_y;
        if (cycle0) begin
          out_sign    <= act[ACT_EBITS+ACT_MBITS] ^ data_in[0]; // sign XOR
          shifted_act <= act_temp;
        end else if (cycle1) begin
          out_exp_sum_r <= add_y[EXP_SUM_WIDTH-1:0];
          shifted_act   <= shifted_act >>> 1;
        end else if (cycle2) begin
          shifted_act <= shifted_act >>> 1;
          shifted_man <= data_in << 1;
          out_val_r   <= add_y[ACT_MBITS+1+MUL_GUARD:0];
        end else begin
          // Continue shifting the fraction captured on cycle2. Reloading from
          // data_in here happened to work for Posit4/5, but filler beats then
          // erased the third and later fraction bits at P>=6.
          shifted_man <= shifted_man << 1;
          shifted_act <= shifted_act >>> 1;
          out_val_r   <= add_y[ACT_MBITS+1+MUL_GUARD:0];
          if (cycleLast) begin
            // The decoder emits its explicit zero flag as the final beat.
            out_zero <= data_in[0] | act_is_zero;
          end
        end
      end
    end
  end
endmodule

// ============================================================================
// TOP: FP8-like PE (stream) with integrated accumulator
// - Instantiates pe_pipe + pe_fp8_compute + pe_bfp_acc
// ============================================================================
module fp8_pe_mac_stream #(
  parameter integer ACT_EBITS      = 4,
  parameter integer ACT_MBITS      = 3,    // fp8 mantissa width
  parameter integer EXP_SUM_WIDTH  = 6,    // exponent sum width produced at cycle1
  parameter integer DATA_WIDTH     = 6,    // decoded Posit field width
  parameter integer ADDW           = 10,   // exponent/shift-add adder width; must cover widened mantissa sum
  parameter integer POSIT_EXP_BIAS = 24,
  // Accumulator view/precision
  parameter integer ACC_EXPW       = 7,
  parameter integer ACC_MANW       = 12,   // integer bits of accumulator mantissa
  parameter integer GRS            = 3,    // guard/round/sticky bits below LSB
  parameter integer MUL_GUARD      = GRS   // sub-integer guard bits; must satisfy MUL_GUARD <= GRS
)(
    input  wire                                 clk,
    input  wire                                 rst_n,
    input  wire                                 stall,           // freeze all state on array stall
    input  wire                                 acc_clear,       // pulse to clear accumulator before a new dot product

    // pipeline handshakes
    input  wire                                 valid,
    input  wire                                 cycle0,
    input  wire                                 cycle1,
    input  wire                                 cycle2,
    input  wire                                 cycleLast,

    // data in (bit-serial decoded beat + activation)
    input  wire [DATA_WIDTH-1:0]                data_in,
    input  wire [ACT_EBITS+ACT_MBITS:0]         act,             // {sign, exp, man}
    input  wire                                 act_valid,       // 1 iff `act` is fresh from FIFO

    // pass-through to next PE
    output wire                                 _valid,
    output wire [ACT_EBITS+ACT_MBITS:0]         _act,
    output wire                                 _act_valid,
    output wire [DATA_WIDTH-1:0]                _data_in,

    // product completion
    output wire                                 mul_done,

    // product fields (per product)
    output wire                                 out_sign,
    output wire [EXP_SUM_WIDTH-1:0]             out_exp_sum_r,   // valid at cycle1
    output wire [ACT_MBITS+1+MUL_GUARD:0]       out_val_r,       // unsigned magnitude with MUL_GUARD sub-integer bits

    // accumulator view (widened by GRS below the original LSB)
    output wire                         acc_valid,
    output wire                         acc_sign,
    output wire [ACC_EXPW-1:0]          acc_exp,
    output wire [ACC_MANW+GRS-1:0]      acc_man
);

  // 1) PIPE: register slice for pass-through
  pe_pipe #(
    .ACT_EBITS  (ACT_EBITS),
    .ACT_MBITS  (ACT_MBITS),
    .DATA_WIDTH (DATA_WIDTH)
  ) U_PIPE (
    .clk         (clk),
    .rst_n       (rst_n),
    .stall       (stall),
    .valid       (valid),
    .act_valid   (act_valid),
    .cycleLast   (cycleLast),
    .data_in     (data_in),
    .act         (act),
    .o_valid     (_valid),
    .o_act_valid (_act_valid),
    .o_data_in   (_data_in),
    .o_act       (_act)
  );

  // 2) COMPUTE: FP8-like multiply (no internal pass-through regs)
  wire product_zero;
  pe_fp8_compute #(
    .ACT_EBITS     (ACT_EBITS),
    .ACT_MBITS     (ACT_MBITS),
    .MUL_GUARD     (MUL_GUARD),
    .EXP_SUM_WIDTH (EXP_SUM_WIDTH),
    .DATA_WIDTH    (DATA_WIDTH),
    .ADDW          (ADDW),
    .POSIT_EXP_BIAS(POSIT_EXP_BIAS)
  ) U_COMPUTE (
    .clk         (clk),
    .rst_n       (rst_n),
    .stall       (stall),
    .valid       (valid),
    .pipe_valid  (_valid),     // preserve original (valid | _valid) gating
    .act_valid   (act_valid),  // unregistered — same cadence as `act`
    .cycle0      (cycle0),
    .cycle1      (cycle1),
    .cycle2      (cycle2),
    .cycleLast   (cycleLast),
    .data_in     (data_in),
    .act         (act),
    .mul_done    (mul_done),
    .out_zero    (product_zero),
    .out_sign    (out_sign),
    .out_exp_sum_r(out_exp_sum_r),
    .out_val_r   (out_val_r)
  );

  // 3) ACCUMULATOR: block-floating accumulate one term per product
  pe_bfp_acc #(
    .IN_EXPW   (EXP_SUM_WIDTH),
    .IN_MANW   (ACT_MBITS),                 // integer bits of out_val_r
    .MUL_GUARD (MUL_GUARD),                 // sub-integer guard bits below LSB
    .ACC_EXPW  (ACC_EXPW),
    .ACC_MANW  (ACC_MANW),
    .GRS       (GRS)
  ) U_ACC (
    .clk       (clk),
    .rst_n     (rst_n),
    .stall     (stall),
    .clear_acc (acc_clear),
    .in_valid  (mul_done),       // accumulate once per product
    .in_zero   (product_zero),
    .in_sign   (out_sign),
    .in_exp    (out_exp_sum_r),
    .in_mag    (out_val_r),
    .acc_sign  (acc_sign),
    .acc_exp   (acc_exp),
    .acc_man   (acc_man),
    .acc_valid (acc_valid)
  );
endmodule

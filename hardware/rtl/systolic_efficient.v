// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

module systolic #(
    parameter integer ACT_EBITS      = 4, // kept for futurae activation wiring
    parameter integer ACT_MBITS      = 3, // kept for future activation wiring
    parameter integer ACC_EXPW       = 7,
    parameter integer ACC_MANW       = 12,
    parameter integer GRS            = 3,     // guard/round/sticky in accumulator
    parameter integer DATA_WIDTH     = 6,   // decoded Posit8 field width
    parameter integer DEPTH          = 8,   // max decoded transaction length
    parameter integer MAN_WIDTH      = 5,   // max Posit8 fraction bits
    parameter integer POSIT_EXP_BIAS = 24,
    parameter ACC_WIDTH = 32,
    parameter N         = 2,
    parameter integer EXP_SUM_WIDTH  = 6,
    parameter integer ADDW           = 10,  // widened for MUL_GUARD=3 shift-add path
    parameter ACT_WIDTH = ACT_EBITS+ACT_MBITS+1,
    parameter integer SCALE_WIDTH    = 5,
    // Ship-out (normalized) FP tuple widths per PE.
    // Mantissa carries the widened (ACC_MANW+GRS)-bit form so bfp_normalize can
    // rescue precision from the GRS bits that pe_bfp_acc preserved during alignment.
    parameter integer NORM_EXPW      = ACC_EXPW + 4,           // absorbs up to +ACC_MANW+GRS shift
    parameter integer NORM_MANW      = ACC_MANW + GRS,         // widened mantissa
    // FP16 ship-out: bfp_to_fp16 converts the normalized+rescaled tuple to
    // IEEE 754 half. BIAS_DELTA = (ACT_BIAS + ACT_MBITS + GRS + POSIT_EXP_BIAS)
    //   - FP16_EXP_BIAS = (7 + 3 + GRS + 24) - 15 = 19 + GRS.
    parameter integer FP16_BIAS_DELTA = 19 + GRS,
    parameter integer OUT_LANE_W     = 16,
    parameter integer OUT_BUS_W      = N * OUT_LANE_W,
    // Width of the per-row completion counters. Sized to hold MAX_K
    // exactly; must match K_WIDTH in mm_efficient and any upstream
    // controller.
    parameter integer K_WIDTH        = 16
)(
    input                   clk,
    input                   rst,
    // FIFO-underflow backpressure. High for one cycle when any operand the
    // array wants to consume this cycle is not yet available. Freezes every
    // synchronous register in the compute pipeline so row/column alignment
    // stays coherent. Config latches (active_reg / precision_reg /
    // posit_es_reg / col_scale_reg) are intentionally NOT gated: they only
    // update on start_pulse which cannot fire during compute, and gating
    // them risks losing the acc_clear pulse if stall ever coincided with
    // active rising (defensive; today's flow never asserts stall pre-start).
    input                   stall,
    input                   active,
    // Per-tile K from the controller. Drives per-row completion
    // thresholds (row_done, row_final_done) and forwards to the
    // posit decoder for its K-bounded raw-bit consumption gate.
    input [K_WIDTH-1:0]     k_reg_in,
    input [3:0]             precision,
    input [1:0]             posit_es,
    input wire signed [SCALE_WIDTH-1:0] col_scale [N-1:0],
    input [ACT_WIDTH-1:0]   act_in [N-1:0],
    // Per-row activation-valid, parallel to act_in. Registered
    // stored_read | bypass_fire pulse from each per-row activation
    // FIFO; propagates east through the PE array and gates each PE's
    // MAC commit (mul_done requires act_valid at cycleLast) so a stale
    // FIFO `dout` never contributes to a MAC.
    input [N-1:0]           act_valid_in,
    input                   w_in [N-1:0],
    output                  done,
    output                  compute_done_out,   // 1-cycle pulse when compute finishes (pre-ship)
    // Ship-out interface (streams N rows, one per cycle, after compute completes)
    output wire             out_valid,
    input  wire             out_ready,
    output wire [$clog2(N)-1:0] out_row,
    output wire [OUT_BUS_W-1:0] out_data,
    output [N-1:0]          active_row,
    output [N-1:0]          active_column
);

    // Internal signals
    wire                 fifo_col_re [0:N];
    // Forward declaration: all_rows_done is consumed by the cycle_gen
    // instantiation below but assigned further down, next to row_final_done;
    // some synthesis tools reject use-before-declaration.
    wire                 all_rows_done;
    wire                 decoder_valid_out [0:N];
    wire                 first_row_pe_valid_out [0:N];
    wire [DATA_WIDTH-1:0] decoder_data_out [0:N];
    wire [ACT_WIDTH-1:0] pe_act_out [0:N][0:N];
    wire                 pe_act_valid_out [0:N][0:N];
    wire [DATA_WIDTH-1:0] pe_data_out   [0:N][0:N];
    wire                 pe_valid [0:N][0:N];
    wire [N*N-1:0]       pe_done;

    // Per-row FIFO-read counter. Increments on act_valid_in[i]
    // (registered stored_read | bypass_fire from act_fifo[i]) —
    // "one activation was actually delivered to PE(i, 0)." Reaching K
    // means row i's FIFO has been drained of its K writes and no more
    // reads should fire, so active_row[i] can drop.
    //
    // NB: counting pe_done[i*N + 0] or pe_done[i*N + (N-1)] instead
    // lags stored_read by 2..N cycles. That lag is enough for
    // precision_count to wrap and fire another stored_read on the
    // now-empty FIFO before the counter reaches K — the exact
    // underflow this gate is meant to prevent.
    reg [K_WIDTH-1:0] row_commit_count [0:N-1];
    integer rci;

    // Per-row LAST-PE commit counter. Increments on pe_done[i*N + (N-1)]
    // (acc_valid pulse of PE(i, N-1)). Reaching K means row i's dot
    // product is final and safe for ship-out to read; all rows
    // reaching K defines compute completion (all_rows_done below),
    // independent of the pipeline-drain condition the old semantic
    // relied on.
    reg [K_WIDTH-1:0] row_final_commit_count [0:N-1];
    integer rfci;

    reg active_reg;
    reg [3:0] precision_reg;
    reg [1:0] posit_es_reg;
    reg signed [SCALE_WIDTH-1:0] col_scale_reg [0:N-1];
    integer cfg_col;
    // One cycle at the start of every operation. This clears stale dot-product
    // state without changing the serialized decode or PE compute schedule.
    wire start_pulse = active && !active_reg;

    // Per-PE accumulator taps (used only during ship-out).
    // acc_man is the widened (ACC_MANW+GRS)-bit magnitude view.
    wire                       acc_sign_arr [0:N-1][0:N-1];
    wire [ACC_EXPW-1:0]        acc_exp_arr  [0:N-1][0:N-1];
    wire [ACC_MANW+GRS-1:0]    acc_man_arr  [0:N-1][0:N-1];
    
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            active_reg  <= 0;
            precision_reg <= 4'd4;
            posit_es_reg <= 0;
            for (cfg_col = 0; cfg_col < N; cfg_col = cfg_col + 1)
                col_scale_reg[cfg_col] <= {SCALE_WIDTH{1'b0}};
            for (rci = 0; rci < N; rci = rci + 1)
                row_commit_count[rci] <= {K_WIDTH{1'b0}};
            // row_final_commit_count MUST be reset here, not only on
            // start_pulse. start_pulse is too late: the descriptor write lands
            // k_reg_in before `active` rises, and row_final_done compares this
            // counter against k_reg_in, so counts left at K by the PREVIOUS run
            // satisfy completion the instant a non-zero K arrives -- while the
            // array is idle. The spurious ship-out that follows eventually
            // underflows a weight FIFO.
            //
            // Without this reset only the first compute run after power-on
            // works. The sibling counter
            // row_commit_count was already reset for exactly this reason.
            for (rfci = 0; rfci < N; rfci = rfci + 1)
                row_final_commit_count[rfci] <= {K_WIDTH{1'b0}};
        end
        else begin
            active_reg <= active;
            // Configuration is stable for the full operation even if the
            // software-visible input changes while the array is running.
            if (start_pulse) begin
                precision_reg <= precision;
                posit_es_reg <= posit_es;
                for (cfg_col = 0; cfg_col < N; cfg_col = cfg_col + 1)
                    col_scale_reg[cfg_col] <= col_scale[cfg_col];
            end
            // Per-row read counter. Bumps on act_valid_in[i] (one
            // activation delivered to PE(i,0) from FIFO i) while
            // active=1. Held at 0 outside S_RUN so a stale count from
            // the previous tile can't leak into the next tile's
            // row_in_progress and hold active_row up on drained FIFOs.
            if (!active) begin
                for (rci = 0; rci < N; rci = rci + 1)
                    row_commit_count[rci] <= {K_WIDTH{1'b0}};
            end
            else begin
                for (rci = 0; rci < N; rci = rci + 1)
                    if (act_valid_in[rci])
                        row_commit_count[rci] <=
                            row_commit_count[rci] + {{(K_WIDTH-1){1'b0}}, 1'b1};
            end
            // Per-row LAST-PE commit counter. Bumps on
            // pe_done[i*N + (N-1)], the acc_valid pulse of PE(i, N-1).
            // Reset on start_pulse so a fresh tile starts from 0.
            if (start_pulse) begin
                for (rfci = 0; rfci < N; rfci = rfci + 1)
                    row_final_commit_count[rfci] <= {K_WIDTH{1'b0}};
            end
            else begin
                for (rfci = 0; rfci < N; rfci = rfci + 1)
                    if (pe_done[rfci*N + (N-1)])
                        row_final_commit_count[rfci] <=
                            row_final_commit_count[rfci] + {{(K_WIDTH-1){1'b0}}, 1'b1};
            end
        end
    end

    // Wires for strobes
    wire cyc0, cyc1, cyc2, cycLast, cycBeforeLast;
    wire pe_cyc0, pe_cyc1, pe_cyc2, pe_cycLast;

    // Instantiate the cycle generator
    cycle_gen #(
        .MAX_PREC(8)  // maximum supported precision
    ) u_cyc (
        .clk         (clk),
        .rst         (rst),          // active-low reset
        .stall       (stall),         // freeze counting during array stall
        .start_valid (active_reg),       // global start
        .done_clear  (done),         // clear when bottom-right PE asserts done
        .activation_done (all_rows_done),
        .precision   (precision_reg),

        .cycle0      (cyc0),
        .cycle1      (cyc1),
        .cycle2      (cyc2),
        .cycleBeforeLast (cycBeforeLast),
        .cycleLast   (cycLast),

        .pe_cycle0   (pe_cyc0),
        .pe_cycle1   (pe_cyc1),
        .pe_cycle2   (pe_cyc2),
        .pe_cycleLast(pe_cycLast)
    );

    // row i uses bit i (i-cycle delayed)
    reg [N-1:0] cyc0_sr, cyc1_sr, cyc2_sr, cycLast_sr;

    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            cyc0_sr    <= {N{1'b0}};
            cyc1_sr    <= {N{1'b0}};
            cyc2_sr    <= {N{1'b0}};
            cycLast_sr <= {N{1'b0}};
        end else if (!stall) begin
            cyc0_sr    <= {cyc0_sr[N-2:0],    pe_cyc0   };
            cyc1_sr    <= {cyc1_sr[N-2:0],    pe_cyc1   };
            cyc2_sr    <= {cyc2_sr[N-2:0],    pe_cyc2   };
            cycLast_sr <= {cycLast_sr[N-2:0], pe_cycLast};
        end
    end
    
    genvar i, j;
    generate 
      for (j = 0; j < N; j = j + 1) begin: decoder_loop
        wire decoder_valid;
        assign decoder_valid = (j==0)? active_reg : decoder_valid_out[j-1];
        posit_decode_and_queue #(
          .DATA_WIDTH(DATA_WIDTH),
          .DEPTH(DEPTH),
          .MAN_WIDTH(MAN_WIDTH),
          .MAX_ES(2),
          .K_WIDTH(K_WIDTH)
        ) posit_decoder_and_fifo(
          .clk(clk),
          .rst_n(rst),
          .stall(stall),
          .w(w_in[j]),
          .valid(decoder_valid),
          .cycle0(cyc0),
          .cycle1(cyc1),
          .cycleBeforeLast(cycBeforeLast),
          .cycleLast(cycLast),
          .precision(precision_reg),
          .posit_es(posit_es_reg),
          .k_reg(k_reg_in),
          .data_out(decoder_data_out[j]),
          .out_valid(first_row_pe_valid_out[j]),
          .fifo_col_re(fifo_col_re[j]),
          .next_valid(decoder_valid_out[j])
        );

      end
    endgenerate

    generate
        for (i = 0; i < N; i = i + 1) begin : row
            for (j = 0; j < N; j = j + 1) begin : col
                wire local_valid;
                wire [DATA_WIDTH-1:0] local_data_in;
                wire [ACT_WIDTH-1:0] local_act;
                wire local_act_valid;
                assign local_valid = (i==0)? first_row_pe_valid_out[j] : pe_valid[i-1][j];
                assign local_data_in = (i==0)? decoder_data_out[j] : pe_data_out[i-1][j];
                assign local_act = (j==0)? act_in[i] : pe_act_out[i][j-1];
                // Activation-valid mirrors act's east propagation: FIFO for
                // column 0, previous PE's _act_valid for other columns.
                assign local_act_valid = (j==0)? act_valid_in[i]
                                              : pe_act_valid_out[i][j-1];
                wire pe_cyc0_temp, pe_cyc1_temp, pe_cyc2_temp, pe_cycLast_temp;
                assign pe_cyc0_temp = (i==0)? pe_cyc0 : cyc0_sr[i-1];
                assign pe_cyc1_temp = (i==0)? pe_cyc1 : cyc1_sr[i-1];
                assign pe_cyc2_temp = (i==0)? pe_cyc2 : cyc2_sr[i-1];
                assign pe_cycLast_temp = (i==0)? pe_cycLast : cycLast_sr[i-1];

                fp8_pe_mac_stream #(
                    .ACT_EBITS(ACT_EBITS),
                    .ACT_MBITS(ACT_MBITS),
                    .EXP_SUM_WIDTH(EXP_SUM_WIDTH),
                    .ACC_EXPW(ACC_EXPW),
                    .ACC_MANW(ACC_MANW),
                    .GRS(GRS),
                    .DATA_WIDTH(DATA_WIDTH),
                    .ADDW(ADDW),
                    .POSIT_EXP_BIAS(POSIT_EXP_BIAS)
                ) pe(
                    .clk(clk),
                    .rst_n(rst),
                    .stall(stall),
                    .valid(local_valid),
                    .cycle0(pe_cyc0_temp),
                    .cycle1(pe_cyc1_temp),
                    .cycle2(pe_cyc2_temp),
                    .cycleLast(pe_cycLast_temp),
                    .data_in(local_data_in),
                    .act(local_act),
                    .act_valid(local_act_valid),
                    ._valid(pe_valid[i][j]),
                    ._act(pe_act_out[i][j]),
                    ._act_valid(pe_act_valid_out[i][j]),
                    ._data_in(pe_data_out[i][j]),
                    .acc_valid(pe_done[i*N+j]),
                    .acc_clear(start_pulse),
                    .acc_sign(acc_sign_arr[i][j]),
                    .acc_exp (acc_exp_arr [i][j]),
                    .acc_man (acc_man_arr [i][j])
                );

            end
        end
    endgenerate

    reg [N-1:0] active_shift_reg;
    reg first_act_valid;
    always @(posedge clk or negedge rst)
        if (!rst) begin
            active_shift_reg <= 0;
            first_act_valid <= 0;
        end
        else if (!stall) begin
            active_shift_reg <= {first_act_valid, active_shift_reg[N-1:1]};
            if (cycLast) first_act_valid <= active_reg;
        end

    // Per-row completion gates the shift-reg drain. Baseline
    // active_row from the shift register drops on a fixed cycle count
    // independent of MAC commits. With mul_done gated by act_valid,
    // some cycleLasts may not commit and the baseline would drop
    // row i's demand before row i had committed K MACs. This gate
    // holds active_row[i] high while row i is in progress:
    // count > 0 AND count < K. Inert at L=1 (K MACs commit naturally
    // during S_RUN so row_done fires before the baseline drops).
    wire [N-1:0] row_done;
    wire [N-1:0] row_in_progress;
    genvar rd_row;
    generate
        for (rd_row = 0; rd_row < N; rd_row = rd_row + 1) begin : gen_row_done
            assign row_done[rd_row] =
                (row_commit_count[rd_row] >= k_reg_in) &&
                (k_reg_in != {K_WIDTH{1'b0}});
            assign row_in_progress[rd_row] =
                (row_commit_count[rd_row] != {K_WIDTH{1'b0}}) &&
                !row_done[rd_row];
        end
    endgenerate

    genvar rr, cc;
    generate
        for (rr = 0; rr < N; rr = rr + 1) begin : gen_active_row
            wire baseline_active_row =
                (rr==0)? first_act_valid : active_shift_reg[N-rr];
            assign active_row[rr] = baseline_active_row | row_in_progress[rr];
        end
        for (cc = 0; cc < N; cc = cc + 1) begin : gen_active_column
            assign active_column[cc] = (cc == 0) ? active : fifo_col_re[cc-1];
        end
    endgenerate
    // Level signal; rises when every row's PE(N-1) has committed K
    // MACs. Includes the current-cycle pe_done pulse in the per-row K
    // test so the rising edge aligns with the same posedge the counter
    // would update. Independent of pipeline-drain condition: "compute
    // is done when every row has finished its K MACs" regardless of
    // whether weight-side `valid` has drained.
    wire [N-1:0] row_final_done;
    genvar rfd;
    generate
        for (rfd = 0; rfd < N; rfd = rfd + 1) begin : gen_row_final_done
            assign row_final_done[rfd] =
                ((row_final_commit_count[rfd] +
                  {{(K_WIDTH-1){1'b0}}, pe_done[rfd*N + (N-1)]}) >= k_reg_in) &&
                (k_reg_in != {K_WIDTH{1'b0}});
        end
    endgenerate
    assign all_rows_done = &row_final_done;

    // compute_done as a rising-edge pulse of all_rows_done. all_rows_done
    // is a level that stays high from last-row-K-commit until the next
    // tile's start_pulse resets the counters — leaving it as-is would
    // re-trigger ship-out every cycle SH_IDLE re-enters, until start_pulse.
    // The pulse form matches the legacy compute_done semantics that the
    // ship-out FSM (and controller S_WAIT_COMPUTE) were written against.
    reg all_rows_done_prev;
    always @(posedge clk or negedge rst) begin
        if (!rst)
            all_rows_done_prev <= 1'b0;
        else
            all_rows_done_prev <= all_rows_done;
    end
    wire compute_done = all_rows_done & ~all_rows_done_prev;
    assign compute_done_out = compute_done;

    // ------------------------------------------------------------------
    // Ship-out FSM + drain pipeline
    //   SH_IDLE       : waiting for compute_done
    //   SH_SHIP       : hand rows to the stage register (one per accepted cycle)
    //   SH_DONE       : wait for stage to drain, then pulse `done` and return
    //
    // FSM drives internal (emit_valid, emit_row) which select acc_{sign,exp,
    // man}_arr[emit_row][*]. Each column is normalized+rescaled combinationally
    // and captured into a one-entry stage register. The FP16 conversion runs
    // from the stage register (post-pipeline), so the critical path is split
    // into (mux -> normalize -> rescale) and (fp16 convert -> port). Adds one
    // cycle of drain fill latency; steady-state throughput is unchanged.
    // ------------------------------------------------------------------
    localparam integer LGN_SAFE = (N > 1) ? $clog2(N) : 1;
    localparam [1:0] SH_IDLE = 2'd0, SH_SHIP = 2'd1, SH_DONE = 2'd2;
    reg [1:0]          sh_state;
    reg                done_r;
    reg                compute_done_seen;
    reg                emit_valid;
    reg [LGN_SAFE-1:0] emit_row;
    assign             done = done_r;

    // Stage register (between rescale and FP16 convert). One-entry with a
    // valid/ready handshake; accepts new data whenever empty or the current
    // entry was accepted by the consumer.
    reg                 drain_stage_valid;
    reg [LGN_SAFE-1:0]  drain_stage_row;
    reg                 drain_stage_sign [0:N-1];
    reg [NORM_EXPW-1:0] drain_stage_exp  [0:N-1];
    reg [NORM_MANW-1:0] drain_stage_man  [0:N-1];

    wire drain_stage_ready = !drain_stage_valid || out_ready;

    // Combinational drain (per-column) at module scope so the stage register
    // can capture all N tuples together.
    wire                 drain_scaled_sign [0:N-1];
    wire [NORM_EXPW-1:0] drain_scaled_exp  [0:N-1];
    wire [NORM_MANW-1:0] drain_scaled_man  [0:N-1];

    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            sh_state          <= SH_IDLE;
            emit_valid        <= 1'b0;
            emit_row          <= {LGN_SAFE{1'b0}};
            done_r            <= 1'b0;
            compute_done_seen <= 1'b0;
        end else begin
            done_r     <= 1'b0;
            emit_valid <= 1'b0;
            case (sh_state)
                SH_IDLE: begin
                    if (compute_done) compute_done_seen <= 1'b1;
                    if (compute_done | compute_done_seen) begin
                        sh_state          <= SH_SHIP;
                        emit_row          <= {LGN_SAFE{1'b0}};
                        emit_valid        <= 1'b1;
                        compute_done_seen <= 1'b0;
                    end
                end
                SH_SHIP: begin
                    emit_valid <= 1'b1;
                    if (drain_stage_ready) begin
                        if (emit_row == (N-1)) begin
                            emit_valid <= 1'b0;
                            sh_state   <= SH_DONE;
                        end else begin
                            emit_row <= emit_row + 1'b1;
                        end
                    end
                end
                SH_DONE: begin
                    emit_valid <= 1'b0;
                    if (!drain_stage_valid) begin
                        done_r   <= 1'b1;
                        sh_state <= SH_IDLE;
                    end
                end
                default: sh_state <= SH_IDLE;
            endcase
        end
    end

    integer drc;
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            drain_stage_valid <= 1'b0;
            drain_stage_row   <= {LGN_SAFE{1'b0}};
            for (drc = 0; drc < N; drc = drc + 1) begin
                drain_stage_sign[drc] <= 1'b0;
                drain_stage_exp[drc]  <= {NORM_EXPW{1'b0}};
                drain_stage_man[drc]  <= {NORM_MANW{1'b0}};
            end
        end else if (drain_stage_ready) begin
            drain_stage_valid <= emit_valid;
            drain_stage_row   <= emit_row;
            for (drc = 0; drc < N; drc = drc + 1) begin
                drain_stage_sign[drc] <= drain_scaled_sign[drc];
                drain_stage_exp[drc]  <= drain_scaled_exp[drc];
                drain_stage_man[drc]  <= drain_scaled_man[drc];
            end
        end
    end

    // Module outputs driven by the pipeline stage.
    assign out_valid = drain_stage_valid;
    assign out_row   = drain_stage_row;

    // Row mux + per-column normalize/rescale (feeds the stage register).
    // FP16 convert runs from the stage register (post-pipeline).
    // Rescale is a signed exponent adjustment, so it adds no multiplier.
    genvar cn;
    generate
        for (cn = 0; cn < N; cn = cn + 1) begin : gen_ship_col
            wire                       sel_sign;
            wire [ACC_EXPW-1:0]        sel_exp;
            wire [ACC_MANW+GRS-1:0]    sel_man;
            assign sel_sign = acc_sign_arr[emit_row][cn];
            assign sel_exp  = acc_exp_arr [emit_row][cn];
            assign sel_man  = acc_man_arr [emit_row][cn];

            wire                  norm_sign;
            wire [NORM_EXPW-1:0]  norm_exp;
            wire [NORM_MANW-1:0]  norm_man;
            bfp_normalize #(
                .ACC_EXPW (ACC_EXPW),
                .ACC_MANW (ACC_MANW),
                .GRS      (GRS),
                .NORM_EXPW(NORM_EXPW)
            ) u_norm (
                .in_sign (sel_sign),
                .in_exp  (sel_exp),
                .in_man  (sel_man),
                .out_sign(norm_sign),
                .out_exp (norm_exp),
                .out_man (norm_man)
            );

            wire                  scaled_sign;
            wire [NORM_EXPW-1:0]  scaled_exp;
            wire [NORM_MANW-1:0]  scaled_man;
            bfp_pow2_rescale #(
                .EXP_WIDTH  (NORM_EXPW),
                .MAN_WIDTH  (NORM_MANW),
                .SCALE_WIDTH(SCALE_WIDTH)
            ) u_rescale (
                .in_sign (norm_sign),
                .in_exp  (norm_exp),
                .in_man  (norm_man),
                .scale   (col_scale_reg[cn]),
                .out_sign(scaled_sign),
                .out_exp (scaled_exp),
                .out_man (scaled_man)
            );

            // Expose scaled_* per column so the stage register can capture
            // all N tuples together.
            assign drain_scaled_sign[cn] = scaled_sign;
            assign drain_scaled_exp[cn]  = scaled_exp;
            assign drain_scaled_man[cn]  = scaled_man;

            // Convert to IEEE 754 binary16 from the pipelined stage.
            wire [15:0] fp16_lane;
            bfp_to_fp16 #(
                .IN_EXPW   (NORM_EXPW),
                .IN_MANW   (NORM_MANW),
                .BIAS_DELTA(FP16_BIAS_DELTA)
            ) u_to_fp16 (
                .in_sign(drain_stage_sign[cn]),
                .in_exp (drain_stage_exp[cn]),
                .in_man (drain_stage_man[cn]),
                .fp16   (fp16_lane)
            );

            assign out_data[cn*OUT_LANE_W +: OUT_LANE_W] = fp16_lane;
        end
    endgenerate

endmodule

// -----------------------------------------------------------------------------
// cycle_gen.v  (Verilog-2001)
// - Latches counting on start_valid=1, clears on done_clear=1
// - Emits cycle0 / cycle1 / cycle2 / cycleLast each precision-length burst
// - Also emits 1-cycle delayed versions for PEs: pe_cycle*
// - rst is active-low
// -----------------------------------------------------------------------------
module cycle_gen #(
    parameter integer MAX_PREC = 8
)(
    input  wire        clk,
    input  wire        rst,          // active-low
    input  wire        stall,        // freeze counting while array is stalled
    input  wire        start_valid,  // pulse/high to start counting
    input  wire        done_clear,   // pulse/high to stop counting
    // Activation-side completion halt. High while all rows have
    // committed K MACs at PE(N-1) (same signal that drives compute_done's
    // rising edge). eff_done_clear ORs this with the ship-out `done`
    // input so cycle_gen halts as soon as the last useful MAC commits
    // rather than waiting for ship-out to complete.
    input  wire        activation_done,
    input  wire [3:0]  precision,    // 1..MAX_PREC

    // For decoders/queues
    output wire        cycle0,
    output wire        cycle1,
    output wire        cycle2,
    output wire        cycleBeforeLast,
    output wire        cycleLast,

    // For PEs (1-cycle delayed)
    output reg         pe_cycle0,
    output reg         pe_cycle1,
    output reg         pe_cycle2,
    output reg         pe_cycleLast
);
    reg       counting;
    reg [2:0] cnt;

    // Halt on activation_done in addition to the ship-out done pulse.
    // activation_done stays high from last-K-MAC through next-tile
    // start_pulse (row_final_commit_count reset), so it's a level
    // source for eff_done_clear. done_clear takes priority over
    // start_valid in the FSM below, so the fresh tile only starts
    // counting once activation_done has dropped -- which happens the
    // cycle row_final_commit_count is reset by start_pulse.
    wire eff_done_clear = done_clear | activation_done;

    // counting FSM
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            counting <= 1'b0;
        end else if (!stall) begin
            if (eff_done_clear) begin
                counting <= 1'b0;
            end else if (start_valid) begin
                counting <= 1'b1;
            end
        end
    end

    // free-run 0..precision-1 while counting (starts immediately on start_valid)
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            cnt <= 3'd0;
        end else if (!stall) begin
            if (eff_done_clear) begin
                // A new operation must restart on serialized bit 0. Keeping the
                // previous modulo-P phase corrupted column 0 on the second job.
                cnt <= 3'd0;
            end else if (counting | start_valid) begin
                if (cnt == (precision - 1))
                    cnt <= 3'd0;
                else
                    cnt <= cnt + 3'd1;
            end
        end
    end

    // decoder strobes (same cycle)
    assign cycle0    = (counting | start_valid) & (cnt == 3'd0);
    assign cycle1    = counting & (cnt == 3'd1);
    assign cycle2    = counting & (cnt == 3'd2);
    assign cycleLast = counting & (cnt == (precision - 1));
    assign cycleBeforeLast = counting & (cnt == (precision - 2));

    // PE strobes (1-cycle delayed)
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            pe_cycle0    <= 1'b0;
            pe_cycle1    <= 1'b0;
            pe_cycle2    <= 1'b0;
            pe_cycleLast <= 1'b0;
        end else if (!stall) begin
            pe_cycle0    <= cycle0;
            pe_cycle1    <= cycle1;
            pe_cycle2    <= cycle2;
            pe_cycleLast <= cycleLast;
        end
    end
endmodule

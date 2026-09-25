// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

// Array-level testbench: one GEMM tile through the N x N FlexPosit array.
//
//   out[i][j] = 2^scale[j] * sum_k act[i][k] * w[j][k]
//
// act[i][k]  FP8 E4M3 activation codes          build/act.mem    (i*K + k)
// w[j][k]    Posit(P, ES) weight codes, one     build/w.mem      ((j*K + k)*P + bit)
//            serialized bit per line, MSB first,
//            sign-magnitude (SerialPosit) order
// scale[j]   signed 5-bit per-column exponent   build/scale.mem  (j)
//
// Each FP16 output lane is printed as "LANE <i> <j> <hex>" for
// scripts/run_array_test.py to compare against model/flexposit_rtl_exact.py.
module array_tb;

    parameter integer N  = 4;
    parameter integer K  = 16;
    parameter integer P  = 4;
    parameter integer ES = 1;

    localparam integer ACT_WIDTH  = 8;
    localparam integer OUT_LANE_W = 16;
    localparam integer OUT_BUS_W  = N * OUT_LANE_W;

    reg clk, rst, active;
    reg [3:0] precision;
    reg [1:0] posit_es;
    reg signed [4:0] col_scale [N-1:0];

    wire done;
    wire compute_done;
    wire                 ship_valid;
    wire [$clog2(N)-1:0] ship_row;
    wire [OUT_BUS_W-1:0] ship_data;

    reg [ACT_WIDTH-1:0] act_mem   [0:N*K-1];
    reg                 w_mem     [0:N*K*P-1];
    reg [4:0]           scale_mem [0:N-1];

    reg [ACT_WIDTH-1:0] act_din [N-1:0];
    reg                 w_din   [N-1:0];
    reg wr_en_act, wr_en_w;

    always #5 clk = ~clk;

    mm #(
        .N(N),
        .K(K),
        // Sized for one full tile: K activations and K*P weight bits per lane.
        .ACT_FIFO_DEPTH(256),
        .W_FIFO_DEPTH(1024)
    ) dut (
        .clk(clk),
        .rst(rst),
        .active(active),
        .streamer_pending(1'b0),
        .k_reg(K[15:0]),
        .precision(precision),
        .posit_es(posit_es),
        .col_scale(col_scale),
        .act_din(act_din),
        .w_din(w_din),
        .wr_en_act(wr_en_act),
        .wr_en_w(wr_en_w),
        .done(done),
        .compute_done(compute_done),
        .out_valid(ship_valid),
        .out_ready(1'b1),
        .out_row(ship_row),
        .out_data(ship_data)
    );

    integer k, r, p, c;
    initial begin
        if ($test$plusargs("VCD")) begin
            $dumpfile("build/array_tb.vcd");
            $dumpvars(0, array_tb);
        end

        $readmemh("build/act.mem", act_mem);
        $readmemh("build/w.mem", w_mem);
        $readmemh("build/scale.mem", scale_mem);

        clk = 0;
        rst = 1;
        active = 0;
        wr_en_act = 0;
        wr_en_w = 0;
        precision = P[3:0];
        posit_es = ES[1:0];
        for (c = 0; c < N; c = c + 1)
            col_scale[c] = $signed(scale_mem[c]);

        #10 rst = 0;
        #10 rst = 1;

        // Fill the per-row activation FIFOs.
        wr_en_act = 1;
        for (k = 0; k < K; k = k + 1) begin
            for (r = 0; r < N; r = r + 1)
                act_din[r] = act_mem[r*K + k];
            #10;
        end
        wr_en_act = 0;
        #10;

        // Fill the per-column weight FIFOs, one serialized bit per cycle.
        wr_en_w = 1;
        for (k = 0; k < K; k = k + 1) begin
            for (p = 0; p < P; p = p + 1) begin
                for (c = 0; c < N; c = c + 1)
                    w_din[c] = w_mem[(c*K + k)*P + p];
                #10;
            end
        end
        wr_en_w = 0;
        #20;

        active = 1;
        #(10*K*P);
        active = 0;
        wait (done === 1'b1);
        @(negedge clk);
        $finish;
    end

    initial begin
        repeat (K*P + K*(P+1) + 30*N + 300) @(posedge clk);
        $fatal(1, "Timeout (N=%0d K=%0d P=%0d ES=%0d)", N, K, P, ES);
    end

    // Ship-out: one row per cycle; each row carries N FP16 lanes.
    integer lane;
    always @(posedge clk) begin
        if (ship_valid)
            for (lane = 0; lane < N; lane = lane + 1)
                $display("LANE %0d %0d %h", ship_row, lane,
                         ship_data[lane*OUT_LANE_W +: OUT_LANE_W]);
    end

endmodule

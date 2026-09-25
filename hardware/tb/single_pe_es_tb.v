// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Yimin Gao
// FlexPosit: https://github.com/hplp/FlexPosit
// If you use this in academic work, please cite:
//   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
//   Inference Accelerators," MICRO 2026 (to appear).

`timescale 1ns/1ps

module single_pe_es_tb #(
    parameter integer P = 4
);
    localparam integer ACT_EBITS      = 4;
    localparam integer ACT_MBITS      = 3;
    localparam integer MUL_GUARD      = 3;   // matches new fp8_pe_mac_stream default
    localparam integer DATA_WIDTH     = 6;
    localparam integer MAN_WIDTH      = 5;
    localparam integer EXP_SUM_WIDTH  = 6;
    localparam integer ADDW           = 10;
    localparam integer ACC_EXPW       = 7;
    localparam integer ACC_MANW       = 12;
    localparam integer GRS            = 3;
    localparam integer POSIT_EXP_BIAS = 24;
    localparam [3:0] PRECISION = P;

    reg clk;
    reg rst_n;
    reg start_valid;
    reg raw_valid;
    reg raw_w;
    reg [1:0] posit_es;

    reg running;
    reg [2:0] phase;
    wire cycle0    = (running | start_valid) && (phase == 3'd0);
    wire cycle1    = running && (phase == 3'd1);
    wire cycle2    = running && (phase == 3'd2);
    wire cycleLast = running && (phase == (P-1));
    wire cycleBeforeLast = running && (phase == (P-2));

    reg pe_cycle0;
    reg pe_cycle1;
    reg pe_cycle2;
    reg pe_cycleLast;

    always #5 clk = ~clk;

    // Local copy of cycle_gen timing: decoder strobes now, PE strobes one
    // cycle later. Reset between cases keeps every product independent.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running      <= 1'b0;
            phase        <= 3'd0;
            pe_cycle0    <= 1'b0;
            pe_cycle1    <= 1'b0;
            pe_cycle2    <= 1'b0;
            pe_cycleLast <= 1'b0;
        end
        else begin
            if (start_valid)
                running <= 1'b1;
            if (running | start_valid) begin
                if (phase == (P-1))
                    phase <= 3'd0;
                else
                    phase <= phase + 1'b1;
            end

            pe_cycle0    <= cycle0;
            pe_cycle1    <= cycle1;
            pe_cycle2    <= cycle2;
            pe_cycleLast <= cycleLast;
        end
    end

    wire [DATA_WIDTH-1:0] decoded_data;
    wire decoded_valid;

    posit_decode_and_queue #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(8),
        .MAN_WIDTH(MAN_WIDTH),
        .MAX_ES(2)
    ) u_decode (
        .clk(clk),
        .rst_n(rst_n),
        .w(raw_w),
        .valid(raw_valid),
        .cycle0(cycle0),
        .cycle1(cycle1),
        .cycleBeforeLast(cycleBeforeLast),
        .cycleLast(cycleLast),
        .stall(1'b0),   // standalone tb never stalls; unconnected floats
                        // to z and X-poisons the !stall-gated registers.
        .precision(PRECISION),
        .posit_es(posit_es),
        .k_reg(16'd0),  // 0 disables the K-budget assertion (standalone tb)
        .data_out(decoded_data),
        .out_valid(decoded_valid),
        .fifo_col_re(),
        .next_valid()
    );

    reg [ACT_EBITS+ACT_MBITS:0] activation;
    wire pipe_valid;
    wire [ACT_EBITS+ACT_MBITS:0] pipe_act;
    wire [DATA_WIDTH-1:0] pipe_data;
    wire mul_done;
    wire product_sign;
    wire [EXP_SUM_WIDTH-1:0] product_exp;
    wire [ACT_MBITS+1+MUL_GUARD:0] product_mag;
    wire acc_valid;
    wire acc_sign;
    wire [ACC_EXPW-1:0] acc_exp;
    wire [ACC_MANW+GRS-1:0] acc_man;

    fp8_pe_mac_stream #(
        .ACT_EBITS(ACT_EBITS),
        .ACT_MBITS(ACT_MBITS),
        .MUL_GUARD(MUL_GUARD),
        .EXP_SUM_WIDTH(EXP_SUM_WIDTH),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDW(ADDW),
        .POSIT_EXP_BIAS(POSIT_EXP_BIAS),
        .ACC_EXPW(ACC_EXPW),
        .ACC_MANW(ACC_MANW),
        .GRS(GRS)
    ) u_pe (
        .clk(clk),
        .rst_n(rst_n),
        .stall(1'b0),      // see above
        .acc_clear(1'b0),
        .valid(decoded_valid),
        // This tb drives `activation` directly as a register, so the
        // activation operand is always architecturally fresh. Tying
        // act_valid high reproduces the pre-act-valid-gate mul_done
        // semantics this tb was written against; unconnected it floats
        // to z and every MAC commit resolves to X.
        .act_valid(1'b1),
        .cycle0(pe_cycle0),
        .cycle1(pe_cycle1),
        .cycle2(pe_cycle2),
        .cycleLast(pe_cycleLast),
        .data_in(decoded_data),
        .act(activation),
        ._valid(pipe_valid),
        ._act(pipe_act),
        ._data_in(pipe_data),
        .mul_done(mul_done),
        .out_sign(product_sign),
        .out_exp_sum_r(product_exp),
        .out_val_r(product_mag),
        .acc_valid(acc_valid),
        .acc_sign(acc_sign),
        .acc_exp(acc_exp),
        .acc_man(acc_man)
    );

    integer tests;
    integer failures;

    task automatic expected_product;
        input  [P-1:0] code;
        input  [1:0] es;
        input  [7:0] act_code;
        output reg expected_zero;
        output reg expected_sign;
        output integer expected_exp;
        output integer expected_mag;

        reg [P-1:0] magnitude;
        integer act_exp_field;
        integer act_man_field;
        integer act_exp_effective;
        integer act_sig;
        integer act_zero;
        integer idx;
        integer regime_bit;
        integer run;
        integer k;
        integer exponent;
        integer exponent_bits;
        integer fraction_shift;
        begin
            act_exp_field = act_code[6:3];
            act_man_field = act_code[2:0];
            act_zero = (act_exp_field == 0) && (act_man_field == 0);

            if (act_exp_field == 0) begin
                act_exp_effective = 1;
                act_sig = act_man_field;
            end
            else if (act_exp_field == 15) begin
                // Reserved exponent policy: saturate to max-finite E4M3.
                act_exp_effective = 14;
                act_sig = 15;
            end
            else begin
                act_exp_effective = act_exp_field;
                act_sig = 8 + act_man_field;
            end

            expected_zero = (code == {P{1'b0}}) || act_zero;
            expected_sign = code[P-1] ^ act_code[7];
            expected_exp = 0;
            expected_mag = 0;

            if (code != {P{1'b0}}) begin
                magnitude = code[P-1] ? ((~code) + 1'b1) : code;
                idx = P-2;
                regime_bit = magnitude[idx];
                run = 0;
                while ((idx >= 0) && (magnitude[idx] == regime_bit)) begin
                    run = run + 1;
                    idx = idx - 1;
                end
                k = regime_bit ? (run - 1) : -run;

                if (idx >= 0)
                    idx = idx - 1; // regime terminator

                exponent = 0;
                exponent_bits = 0;
                while ((exponent_bits < es) && (idx >= 0)) begin
                    exponent = (exponent << 1) | magnitude[idx];
                    exponent_bits = exponent_bits + 1;
                    idx = idx - 1;
                end
                exponent = exponent << (es - exponent_bits);
                expected_exp = act_exp_effective + POSIT_EXP_BIAS
                               + k * (1 << es) + exponent;

                // Match the PE's bit-serial shift-and-add behavior exactly:
                // each posit fraction bit adds a right-shifted widened
                // activation significand, with truncation at each shift.
                // With MUL_GUARD extra low bits, the internal significand is
                // act_sig << MUL_GUARD, so shifts up to (MUL_GUARD + integer
                // bits) can produce nonzero contributions.
                expected_mag = act_sig << MUL_GUARD;
                fraction_shift = 1;
                while ((idx >= 0) && (fraction_shift <= MAN_WIDTH)) begin
                    if (magnitude[idx])
                        expected_mag = expected_mag +
                            ((act_sig << MUL_GUARD) >> fraction_shift);
                    fraction_shift = fraction_shift + 1;
                    idx = idx - 1;
                end
            end
        end
    endtask

    task automatic run_case;
        input [P-1:0] code;
        input [1:0] es;
        input [7:0] act_code;

        reg [P-1:0] stream_code;
        reg expected_zero;
        reg expected_sign;
        integer expected_exp;
        integer expected_mag;
        integer bit_idx;
        integer timeout;
        reg signed [ACC_MANW+GRS:0] expected_acc_value;
        begin
            expected_product(
                code, es, act_code, expected_zero, expected_sign,
                expected_exp, expected_mag
            );

            // The production memory writer separates the original posit sign
            // and two's-complements the remaining magnitude stream.
            stream_code = code[P-1]
                ? {1'b1, ((~code[P-2:0]) + 1'b1)}
                : code;

            rst_n       = 1'b0;
            start_valid = 1'b0;
            raw_valid   = 1'b0;
            raw_w       = 1'b0;
            posit_es    = es;
            activation  = act_code;
            repeat (2) @(negedge clk);
            rst_n = 1'b1;

            @(negedge clk);
            start_valid = 1'b1;
            raw_valid   = 1'b1;
            raw_w       = stream_code[P-1];
            for (bit_idx = P-2; bit_idx >= 0; bit_idx = bit_idx - 1) begin
                @(negedge clk);
                start_valid = 1'b0;
                raw_w = stream_code[bit_idx];
            end
            @(negedge clk);
            raw_valid = 1'b0;
            raw_w = 1'b0;

            timeout = 0;
            while (!mul_done && timeout < (2*P+24)) begin
                @(posedge clk);
                #1;
                timeout = timeout + 1;
            end

            tests = tests + 1;
            if ($test$plusargs("TRACE"))
                $display(
                    "TRACE P=%0d act=%02h es=%0d code=%0h decoded=%0h decoder_zero=%0b product_zero=%0b sign/exp/mag=%0b/%0d/%0d",
                    P, act_code, es, code, decoded_data,
                    u_decode.d_zero, u_pe.product_zero,
                    product_sign, product_exp, product_mag
                );
            if (!mul_done) begin
                $display("FAIL P=%0d act=%02h es=%0d code=%0h: mul_done timeout",
                         P, act_code, es, code);
                failures = failures + 1;
            end
            else if (!expected_zero &&
                     ((product_sign !== expected_sign) ||
                      (product_exp !== expected_exp) ||
                      (product_mag !== expected_mag))) begin
                $display(
                    "FAIL P=%0d act=%02h es=%0d code=%0h: got sign/exp/mag=%0d/%0d/%0d expected=%0d/%0d/%0d",
                    P, act_code, es, code, product_sign, product_exp, product_mag,
                    expected_sign, expected_exp, expected_mag
                );
                failures = failures + 1;
            end

            timeout = 0;
            while (!acc_valid && timeout < 8) begin
                @(posedge clk);
                #1;
                timeout = timeout + 1;
            end
            if (!acc_valid) begin
                $display("FAIL P=%0d act=%02h es=%0d code=%0h: acc_valid timeout",
                         P, act_code, es, code);
                failures = failures + 1;
            end
            else if (expected_zero &&
                     ((acc_sign !== 1'b0) || (acc_exp !== 0) || (acc_man !== 0))) begin
                $display("FAIL P=%0d act=%02h es=%0d code=%0h: zero modified accumulator",
                         P, act_code, es, code);
                failures = failures + 1;
            end
            else if (!expected_zero) begin
                // Product mag already carries MUL_GUARD sub-integer bits; the
                // accumulator inserts it shifted by (GRS - MUL_GUARD). With
                // MUL_GUARD == GRS the shift is zero.
                expected_acc_value = expected_sign
                    ? -(expected_mag <<< (GRS - MUL_GUARD))
                    :  (expected_mag <<< (GRS - MUL_GUARD));
                if ((acc_exp !== expected_exp) ||
                    ($signed({acc_sign, acc_man}) !== expected_acc_value)) begin
                    $display(
                        "FAIL P=%0d act=%02h es=%0d code=%0h: acc exp/value=%0d/%0d expected=%0d/%0d",
                        P, act_code, es, code,
                        acc_exp, $signed({acc_sign, acc_man}),
                        expected_exp, expected_acc_value
                    );
                    failures = failures + 1;
                end
            end
            else if ($test$plusargs("TRACE"))
                $display(
                    "TRACE accumulator sign/exp/man=%0b/%0d/%0d",
                    acc_sign, acc_exp, acc_man
                );
        end
    endtask

    integer es_idx;
    integer code_idx;
    integer act_idx;
    integer only_act;
    integer only_es;
    integer only_code;
    integer filter_act;
    integer filter_es;
    integer filter_code;
    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start_valid = 1'b0;
        raw_valid = 1'b0;
        raw_w = 1'b0;
        posit_es = 2'd0;
        activation = 8'h00;
        tests = 0;
        failures = 0;
        filter_act = $value$plusargs("ACT=%h", only_act);
        filter_es = $value$plusargs("ES=%d", only_es);
        filter_code = $value$plusargs("CODE=%h", only_code);

        if ($test$plusargs("VCD")) begin
            $dumpfile("build/single_pe_es_tb.vcd");
            $dumpvars(0, single_pe_es_tb);
        end

        for (act_idx = 0; act_idx < 256; act_idx = act_idx + 1) begin
            for (es_idx = 0; es_idx <= 2; es_idx = es_idx + 1) begin
                for (code_idx = 0; code_idx < (1 << P); code_idx = code_idx + 1) begin
                    if (code_idx != (1 << (P-1)) &&
                        (!filter_act || act_idx == only_act) &&
                        (!filter_es || es_idx == only_es) &&
                        (!filter_code || code_idx == only_code))
                        run_case(code_idx[P-1:0], es_idx[1:0], act_idx[7:0]);
                end
            end
        end

        if (failures == 0)
            $display(
                "PASS: %0d exhaustive single-PE FP8 x Posit%0d es=0/1/2 cases",
                tests, P
            );
        else
            $fatal(
                1,
                "FAIL: %0d failures across %0d Posit%0d cases",
                failures,
                tests,
                P
            );

        $finish;
    end
endmodule

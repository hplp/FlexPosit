# FlexPosit hardware — needs Icarus Verilog (iverilog >= 11) and Python 3 + numpy.
#
#   make test        run everything below
#   make test-pe     exhaustive single-PE check, every FP8 x Posit(P, es) pair
#   make test-array  random GEMM tiles through the N x N array vs the Python model
#   make wave        dump a VCD of one small array run (build/array_tb.vcd)

PYTHON ?= python3
BUILD  := build

PE_RTL := rtl/posit_decode_and_queue.v rtl/fp8_pe_mac_stream.v

.PHONY: test test-pe test-array wave clean

test: test-pe test-array

test-pe:
	@mkdir -p $(BUILD)
	@for p in 4 5 6 7 8; do \
	    iverilog -g2012 -o $(BUILD)/single_pe_p$$p -s single_pe_es_tb \
	        -P single_pe_es_tb.P=$$p tb/single_pe_es_tb.v $(PE_RTL) || exit 1; \
	    vvp -n $(BUILD)/single_pe_p$$p || exit 1; \
	done

test-array:
	$(PYTHON) scripts/run_array_test.py

wave:
	$(PYTHON) scripts/run_array_test.py --n 4 --k 4 --p 4 --es 1
	vvp -n $(BUILD)/array_tb_n4_p4 +VCD
	@echo "Open with: gtkwave $(BUILD)/array_tb.vcd"

clean:
	rm -rf $(BUILD)

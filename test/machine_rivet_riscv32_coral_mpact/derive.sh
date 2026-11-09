#!/bin/sh
# The VME core suite's source retargeted at MPACT (mpact-m4: Zvt, Zve32f and
# bfloat16, as the VME core), for correctness only. MPACT has no timing model
# (its mcycle counts instructions), so cycle reports are discarded and their
# lines leave the goldens; the tests the RTL suite tags "disabled" for the
# simulator's cost run here, since MPACT takes milliseconds. A result line
# (bitwise, DIFFERS, refused) must be identical on both, except the one known
# divergence below.
#
# Known divergence: the matrix engine's paired bfloat16 products of the linear
# fixtures differ on MPACT (1 of 288 and 50 of 4096 outputs, all at the float32
# overflow edge, where the RTL and the descriptor's reference give the largest
# finite value and MPACT gives infinity; the same with the v0.0.9 simulator). The rule
# below expects it, so it fails when MPACT agrees and should then be removed.
#
# The suites of machine_rivet_riscv32_coral derive the same way: the RVV core
# becomes mpact-m3 (Zve32f) and the VME core mpact-m4, the cycle budget a limit
# on instructions, and what they report through Cycle_report (inline in a
# result line) leaves the goldens with it.
#
# Usage: derive.sh <RTL suite source>
exec sed -E \
	-e 's/"vme-core-mini-axi"/"mpact-m4"/' \
	-e 's/"rvv-core-mini-(highmem-)?axi"/"mpact-m3"/' \
	-e 's/--cycles=/--max_instructions=/' \
	-e 's/, [0-9]+ cycles( \([0-9.]+ per element\))?(, [0-9.]+ ops\/cycle)?//' \
	-e 's/Fmt\.pr "  cycles/Fmt.kstr ignore "  cycles/' \
	-e 's/Machine_rivet_riscv32_coral_vme_test\.//' \
	-e 's/ \[@tags "disabled"\]//' \
	-e '/linear bf16 [0-9x]+ k64, paired/s/: bitwise$/: DIFFERS/' \
	-e '/^ +(cycles [0-9]|whole: [0-9]|scores block: [0-9]|exp pass: [0-9]|context block: [0-9]|normalize: [0-9])/{/\|\}\]$/!d}' \
	-e 's/^ +(cycles [0-9]|whole: [0-9]|scores block: [0-9]|exp pass: [0-9]|context block: [0-9]|normalize: [0-9])[^|]*(\|\}\]) *$/    \2/' \
	"$1"

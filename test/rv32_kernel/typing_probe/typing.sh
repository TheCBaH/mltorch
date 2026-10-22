#!/bin/sh
# Type-level regression harness for the RV32 kernel AST: what the type checker
# must reject. Every negative case sits next to a control that must still
# compile, because a harness that rejects everything -- a broken invocation is
# enough -- would "pass" while proving nothing. Acceptance is read from
# `val check :` in the output, since a toplevel reports a type error and carries
# on, exiting 0 either way.
#
# Usage: typing.sh <toplevel> <case-name> <expression>
# The expression is spliced into the scaffold below.
set -eu

top=$1
name=$2
expr=$3

out=$(
  cat <<EOF2 | "$top" -noprompt -no-version 2>&1
open Machine_rivet_riscv32
module Insn = Rv32_kernel_insn
module Imm = Rv32_kernel_imm
module R = Rv32_kernel_reg
open Rv32_kernel_reg
let check () = $expr;;
EOF2
)

case "$out" in
*"val check :"*) printf '%s: COMPILES\n' "$name" ;;
*) printf '%s: rejected\n' "$name" ;;
esac

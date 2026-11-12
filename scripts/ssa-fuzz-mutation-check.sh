#!/bin/bash
# A passing suite is not evidence that it can fail. This applies one deliberate
# defect at a time to an emitter's operations (C, JavaScript, Wasm), requires the
# seeded differential of that emitter (test/ssa_fuzz_c, test/ssa_fuzz_js,
# test/ssa_fuzz_wasm) to go red, and restores the file. Each defect is the kind a
# refactor could plausibly introduce: a wrong guard constant, a double rounding,
# a swapped operator, an off-by-one range bound.
#
# The order of the two integer-division checks is not a defect: a zero divisor
# and a divisor of -1 exclude each other, so swapping them changes nothing a
# program can observe.
#
# usage: scripts/ssa-fuzz-mutation-check.sh [c|js|wasm ...]   (from the repository
# root; every emitter when none is named)
set -u

failures=0
file=
backup=
suite=

restore() {
  if [ -n "$backup" ] && [ -f "$backup" ]; then
    cp "$backup" "$file"
    rm -f "$backup"
  fi
}
trap restore EXIT

run_suite() {
  case "$suite" in
    c) NO_COLOR=1 opam exec -- dune build --root . @test/ssa_fuzz_c/runtest ;;
    js) NO_COLOR=1 opam exec -- dune build --root . @test/ssa_fuzz_js/runtest-js ;;
    wasm) MLTORCH_WASM=1 NO_COLOR=1 opam exec -- dune build --root . @test/ssa_fuzz_wasm/runtest ;;
  esac
}

# target SUITE FILE: the emitter file the following mutants are applied to
target() {
  restore
  suite=$1
  file=$2
  backup=$(mktemp)
  cp "$file" "$backup"
}

mutate() {
  local label=$1 expression=$2
  cp "$backup" "$file"
  sed -i "$expression" "$file"
  if cmp -s "$file" "$backup"; then
    echo "NOT APPLIED  [$suite] $label"
    failures=$((failures + 1))
    return
  fi
  local out
  if out=$(run_suite 2>&1); then
    echo "SURVIVED     [$suite] $label"
    failures=$((failures + 1))
  elif echo "$out" | grep -q 'disagrees with the interpreter'; then
    echo "killed       [$suite] $label"
  else
    # a mutant that does not build, or a suite that failed for another reason,
    # proves nothing about the differential
    echo "NOT A KILL   [$suite] $label (the suite failed without a disagreement)"
    failures=$((failures + 1))
  fi
}

WANTED=("$@")
want() {
  [ "${#WANTED[@]}" -eq 0 ] && return 0
  for w in "${WANTED[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

if want c; then
  target c lib/ssa_c/ssa_c_op.ml
  mutate "division overflow guard on a divisor of 1, not -1" \
    's|if (%s == %s \&\& %s == -1) %s|if (%s == %s \&\& %s == 1) %s|'
  mutate "integer to binary32 through binary64 (double rounding)" \
    's|"(float)" \^ n a|"(float)(double)" ^ n a|'
  mutate "integer and index less-than compared unsigned" \
    '/Ssa_op.I64_compare (c, a, b) | Ssa_op.Index_compare/,/(n b))$/ s|Printf.sprintf "(%s %s %s)" (n a)|Printf.sprintf "((uint64_t)%s %s (uint64_t)%s)" (n a)|'
  mutate "Sub emitted as add" \
    's#| Expr.Value.Sub -> "-"#| Expr.Value.Sub -> "+"#'
  target c lib/ssa_c/ssa_c.ml
  mutate "loop-carried values never transferred to the next iteration" \
    's|transfer cx (depth + 2) carried body.Ssa_region.yields;|ignore transfer;|'
  mutate "scan state limit off by one (reservation up to and including it fails)" \
    's|"if (scan_live + %Ld > %s) %s"|"if (scan_live + %Ld >= %s) %s"|'
  mutate "scan update budget off by one (the last charge fails)" \
    's|"if (scan_remaining <= 0) %s"|"if (scan_remaining < 0) %s"|'
fi

if want js; then
  target js lib/ssa_js/ssa_js.ml
  mutate "division overflow guard on a divisor of 1, not -1" \
    's|(bin A.Eq_strict (v b) (big_lit (-1L)))|(bin A.Eq_strict (v b) (big_lit 1L))|'
  mutate "float to int64 accepts 2^63 (the upper bound inclusive)" \
    's|(bin A.Lt x (num 9223372036854775808.))|(bin A.Le x (num 9223372036854775808.))|'
  mutate "int64 subtraction emitted as add" \
    's#| Ssa_op.I64_op.Sub -> A.Sub)#| Ssa_op.I64_op.Sub -> A.Add)#'
  mutate "binary64 to binary32 conversion without the rounding" \
    's|Ssa_op.Convert.F64_to_f32 -> math "fround" \[ v a \]|Ssa_op.Convert.F64_to_f32 -> v a|'
  mutate "loop-carried values never transferred to the next iteration" \
    's|body = (set_iv :: body_stmts) @ transfer;|body = (set_iv :: body_stmts) @ (ignore transfer; []);|'
  mutate "scan state limit off by one (reservation up to and including it fails)" \
    's|(bin A.Gt (bin A.Add (A.Var scan_live) (num live)) (lit max_state))|(bin A.Ge (bin A.Add (A.Var scan_live) (num live)) (lit max_state))|'
  mutate "scan update budget off by one (the last charge fails)" \
    's|(bin A.Le (A.Var scan_remaining) (num 0.))|(bin A.Lt (A.Var scan_remaining) (num 0.))|'
fi

if want wasm; then
  target wasm lib/ssa_wasm/ssa_wasm_instr.ml
  mutate "division overflow guard on a divisor of 1, not -1" \
    's|I.I64_const (-1L);|I.I64_const 1L;|'
  mutate "float to int64 accepts 2^63 (the upper bound inclusive)" \
    '/f64 9223372036854775808.;/{n;s|n Wasm_op.F64_lt;|n Wasm_op.F64_le;|}'
  mutate "int64 subtraction emitted as add" \
    's#| Ssa_op.I64_op.Sub -> Wasm_op.I64_sub#| Ssa_op.I64_op.Sub -> Wasm_op.I64_add#'
  mutate "integer to binary32 through binary64 (double rounding)" \
    's#Ssa_op.Convert.I64_to_f32 -> \[ n Wasm_op.F32_convert_i64_s \]#Ssa_op.Convert.I64_to_f32 -> [ n Wasm_op.F64_convert_i64_s; n Wasm_op.F32_demote_f64 ]#'
  mutate "scan state limit off by one (reservation up to and including it fails)" \
    '/Ssa_op.Meter_reserve width ->/,/Ssa_op.Meter_reset ->/ s|n Wasm_op.I64_gt_s;|n Wasm_op.I64_ge_s;|'
  mutate "scan update budget off by one (the last charge fails)" \
    '/Ssa_op.Meter_charge ->/,/Ssa_op.Meter_release/ s|n Wasm_op.I64_le_s;|n Wasm_op.I64_lt_s;|'
  target wasm lib/ssa_wasm/ssa_wasm.ml
  mutate "loop-carried values never transferred to the next iteration" \
    's|let moves = transfer st carried body.Ssa_region.yields in|let moves = (ignore transfer; []) in|'
fi

restore
backup=

if [ "$failures" -ne 0 ]; then
  echo "$failures mutant(s) not killed"
  exit 1
fi
echo "all mutants killed"

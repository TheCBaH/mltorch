#!/bin/bash
# A passing suite is not evidence that it can fail. This applies one deliberate
# defect at a time to the Wasm lowering, requires the differential suite
# (test/loop_wasm, under node) to go red, and restores the file. Each defect is
# the kind a refactor could plausibly introduce: a swapped operator, a dropped
# rounding, a signed compare where an unsigned one proves a bounds check, a
# wrong constant, a lost NaN rule.
#
# usage: scripts/wasm-mutation-check.sh   (from the repository root)
set -u
file=lib/loop_ir/loop_wasm_value.ml
backup=$(mktemp)
cp "$file" "$backup"
trap 'cp "$backup" "$file"; rm -f "$backup"' EXIT

failures=0
mutate() {
  local label=$1 pattern=$2 replacement=$3
  cp "$backup" "$file"
  sed -i "s|$pattern|$replacement|" "$file"
  if cmp -s "$file" "$backup"; then
    echo "NOT APPLIED  $label"
    failures=$((failures + 1))
    return
  fi
  if MLTORCH_WASM=1 MLTORCH_WASI_SYSROOT="${WASI_SYSROOT:-}" NO_COLOR=1 \
      opam exec -- dune build @test/loop_wasm/runtest >/dev/null 2>&1; then
    echo "SURVIVED     $label"
    failures=$((failures + 1))
  else
    echo "killed       $label"
  fi
}

mutate "Sub lowered as add" \
  'Expr.Value.Sub -> Wasm_op.F64_sub' 'Expr.Value.Sub -> Wasm_op.F64_add'
mutate "Round_f32 dropped" \
  'num st a @ \[ n Wasm_op.F32_demote_f64; n Wasm_op.F64_promote_f32 \]' 'num st a'
mutate "bounds check signed" \
  'int_const st extent; n Wasm_op.I32_ge_u' 'int_const st extent; n Wasm_op.I32_ge_s'
mutate "Float_max as min" \
  'else Wasm_op.F64_max' 'else Wasm_op.F64_min'
mutate "quantization zero point off by one" \
  'f64 (float_of_int zero);' 'f64 (float_of_int (zero + 1));'
mutate "Pool_better wins ties (greater-or-equal)" \
  'else (Wasm_op.F64_gt, Wasm_op.F64_ne)' 'else (Wasm_op.F64_ge, Wasm_op.F64_ne)'
mutate "I64 add lowered as sub" \
  'Expr.Value.I64_add -> a @ b @ \[ n Wasm_op.I64_add \]' 'Expr.Value.I64_add -> a @ b @ [ n Wasm_op.I64_sub ]'

if [ "$failures" -ne 0 ]; then
  echo "$failures mutation(s) survived or did not apply" >&2
  exit 1
fi
echo "every mutation was killed"

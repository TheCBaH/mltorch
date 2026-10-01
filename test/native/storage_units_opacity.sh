#!/bin/sh
# Type-level regression harness for [Core.Storage_units]: byte and element
# quantities, sizes and offsets, and sizes and alignments must not unify, and
# none may be built from a bare [int64] except through its checked
# constructor. Same shape as tagged_int_opacity.sh, and for the same reason:
# every negative case sits next to a control that must still compile, so a
# broken invocation cannot pass by rejecting everything.
#
# Usage: storage_units_opacity.sh <toplevel> <case-name> <expression>
# The expression is spliced into a scaffold binding one value of each type:
# [bs] a byte size, [bo] a byte offset, [ba] a byte alignment, [ec] an element
# count, [eo] an element offset, [ew] an element width, [n] an [int64].
set -eu

top=$1
name=$2
expr=$3

out=$(
  cat <<EOF | "$top" -noprompt -no-version 2>&1
open Core.Storage_units;;
let check (bs : Byte_size.t) (bo : Byte_offset.t) (ba : Byte_alignment.t)
    (ec : Element_count.t) (eo : Element_offset.t) (ew : Element_bytes.t)
    (n : int64) =
  ignore (bs, bo, ba, ec, eo, ew, n);
  $expr;;
EOF
)

case "$out" in
*"val check :"*) printf '%s: COMPILES\n' "$name" ;;
*) printf '%s: rejected\n' "$name" ;;
esac

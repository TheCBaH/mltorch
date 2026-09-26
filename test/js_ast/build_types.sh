#!/bin/sh
# Type-level regression harness for the phantom kinds of [Js_build].
#
# The builders are typed so that a JavaScript Number/BigInt mix, a float used as
# an array subscript, and a bitwise operator on an index are OCaml type errors
# rather than run-time TypeErrors or silent wraps. Every negative case sits next
# to a control that must still compile, because a harness that rejects
# everything -- a broken invocation is enough -- would "pass" while proving
# nothing. Acceptance is read from `val check :` in the output rather than an
# exit code, since a toplevel reports a type error and carries on, exiting 0
# either way. Same shape as test/native/error_opacity.sh.
#
# Usage: build_types.sh <toplevel> <case-name> <expression>
# The expression is spliced into the scaffold below, where [x] is a [num t], [i]
# an [idx t], [b] a [big t], [h] a [bits16 t] and [a] a [num arr t].
set -eu

top=$1
name=$2
expr=$3

out=$(
  cat <<EOT | "$top" -noprompt -no-version 2>&1
open Js_build;;
let check (x : num t) (i : idx t) (b : big t) (h : bits16 t) (a : num arr t) =
  ignore (x, i, b, h, a);
  $expr;;
EOT
)

case "$out" in
*"val check :"*) printf '%s: COMPILES\n' "$name" ;;
*) printf '%s: rejected\n' "$name" ;;
esac

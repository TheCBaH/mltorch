#!/bin/sh
# Type-level regression harness for [Core.Tagged_int.Make]: two applications
# must not unify, and neither may be built from or mixed with a bare [int]
# except through [of_int]. Same shape as error_opacity.sh, and for the same
# reason: every negative case sits next to a control that must still compile,
# so a broken invocation cannot pass by rejecting everything.
#
# Usage: tagged_int_opacity.sh <toplevel> <case-name> <expression>
# The expression is spliced into a scaffold where [a] is an [A.t], [b] a [B.t]
# and [n] an [int]; [A] and [B] are two applications of the functor.
set -eu

top=$1
name=$2
expr=$3

out=$(
  cat <<EOF2 | "$top" -noprompt -no-version 2>&1
module A = Core.Tagged_int.Make (struct let prefix = "a" end) ();;
module B = Core.Tagged_int.Make (struct let prefix = "b" end) ();;
let check (a : A.t) (b : B.t) (n : int) =
  ignore a;
  ignore b;
  ignore n;
  $expr;;
EOF2
)

case "$out" in
*"val check :"*) printf '%s: COMPILES\n' "$name" ;;
*) printf '%s: rejected\n' "$name" ;;
esac

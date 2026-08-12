#!/bin/sh
# Type-level regression harness for the opacity of [Err.Error.t].
#
# The wrapper carries the detection backtrace. While it was a transparent
# record, `{ e with Err.Error.kind = ... }` typechecked and read like a
# widening, but republished the error under a backtrace that was captured
# somewhere else entirely -- graph_view.ml:352 was a live instance. Making the
# type abstract is what turns "use Err.map_error" from a convention into a
# rule, and this harness is the evidence that the rule is enforced by the type
# checker rather than by review.
#
# Same shape and same reasoning as version_safety.sh: every negative case sits
# next to a control that must still compile, because a harness that rejects
# everything -- a broken invocation is enough -- would "pass" while proving
# nothing. Acceptance is read from `val check :` in the output rather than an
# exit code, since a toplevel reports a type error and carries on, exiting 0
# either way.
#
# Usage: error_opacity.sh <toplevel> <case-name> <expression>
# The expression is spliced into the scaffold below, where [e] is an
# [int Err.Error.t] and [r] an [(int, int) Err.t].
set -eu

top=$1
name=$2
expr=$3

out=$(
  cat <<EOF | "$top" -noprompt -no-version 2>&1
let check (e : int Err.Error.t) (r : (int, int) Err.t) =
  ignore e;
  ignore r;
  $expr;;
EOF
)

case "$out" in
*"val check :"*) printf '%s: COMPILES\n' "$name" ;;
*) printf '%s: rejected\n' "$name" ;;
esac

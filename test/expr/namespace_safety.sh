#!/bin/sh
# Type-level regression harness for the [Expr] library's public surface, in the
# same shape as test/native/version_safety.sh (see that file for why a toplevel
# rather than ocamlfind, and why acceptance is detected from output text).
#
# Two properties are under test, and neither is provable by an ordinary test
# module -- a bad reference there just fails the whole test library to build:
#
#   1. The namespace does not leak. [lib/expr] is wrapped, so a consumer sees
#      [Expr.Axis] and never a top-level [Axis]. That is the whole reason the
#      language moved out of [lib/native], which is (wrapped false) and puts
#      every unit in one flat global namespace.
#
#   2. The AST is private. External code may PATTERN-MATCH the variants -- the
#      grounding pass and the transform verifier need to -- but may not
#      CONSTRUCT them, so a malformed scope or an invalid divisor cannot be
#      built casually, only through the smart constructors.
#
# Every negative case sits next to a control that must still compile: a broken
# harness rejects everything and would otherwise "pass" while proving nothing.
#
# Usage: namespace_safety.sh <toplevel> <case-name> <expression>
set -eu

top=$1
name=$2
expr=$3

out=$(
  cat <<EOF | "$top" -noprompt -no-version 2>&1
let check () =
  $expr;;
EOF
)

case "$out" in
*"val check :"*) printf '%s: COMPILES\n' "$name" ;;
*) printf '%s: rejected\n' "$name" ;;
esac

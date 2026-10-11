#!/bin/sh
# Type-level regression harness for the schema's nominal types.
#
# Every identifier, digest, name and text role is its own abstract type, so a
# unit id cannot stand for a case id and a graph digest cannot stand for a
# payload digest, although all of them serialize as strings. A type alias or a
# shared string representation would silently let them mix; this harness is the
# evidence that the type checker, not review, rejects each substitution.
#
# Same shape as test/native/error_opacity.sh: every negative case sits next to a
# control that must still compile, because a harness that rejects everything --
# a broken invocation is enough -- would "pass" while proving nothing.
# Acceptance is read from `val check :` in the output rather than an exit
# code, since a toplevel reports a type error and carries on, exiting 0 either
# way.
#
# Usage: type_boundaries.sh <toplevel> <case-name> <expression>
set -eu

top=$1
name=$2
expr=$3

out=$(
  cat <<EOS | "$top" -noprompt -no-version 2>&1
open Inference_compare_schema;;
let check
    (u : Id.Unit.t) (c : Id.Case.t)
    (g : Digest.Graph.t) (p : Digest.Payload.t)
    (d : Prose.Description.t) (x : Prose.Diagnostic_text.t)
    (i : Input_port.t) (o : Output_port.t)
    (nr : Node_ref.t) (tr : Tensor_ref.t) (s : string) =
  ignore u; ignore c; ignore g; ignore p; ignore d; ignore x;
  ignore i; ignore o; ignore nr; ignore tr; ignore s;
  $expr;;
EOS
)

case "$out" in
*"val check :"*) printf '%s: COMPILES\n' "$name" ;;
*) printf '%s: rejected\n' "$name" ;;
esac

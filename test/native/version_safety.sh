#!/bin/sh
# Type-level regression harness: feed one snippet to a custom toplevel that has
# `native` linked in, and report only whether the type checker ACCEPTED it.
#
# This is the type-level form of the rule in CLAUDE.md that a check which has
# never failed is not evidence. Every negative case here sits next to a control
# that must still compile, because a harness that rejects everything — a broken
# invocation is enough — would "pass" while proving nothing.
#
# The toplevel comes from dune's `(toplevel ...)` stanza, so the library and its
# dependencies are linked in and tracked as a normal dune dependency. The
# alternative, invoking ocamlfind with -I pointing into `.native.objs/byte`,
# reaches into dune's internals and additionally needs every .cmi globbed into
# the cram sandbox — `native` is `wrapped false`, so that is the whole library.
#
# Acceptance is detected from `val check :` in the output rather than an exit
# code: a toplevel reports a type error and carries on, exiting 0 either way.
# The diagnostics themselves are not asserted on — they name types like
# `Cluster_relation.Make(Correspondence.Id)(Correspondence.Label).Tagged.id` and
# existentials like `$Pack_'v1`, whose spelling is a compiler detail, while the
# property under test is acceptance.
#
# Usage: version_safety.sh <toplevel> <case-name> <expression>
# The expression is spliced into the scaffold below, where `m` is a relation
# from snapshot `sa` to snapshot `sb`, `s` is an edge of `sa` and `d` one of
# `sb`. The graph is a parameter, so nothing has to be built to type-check.
set -eu

top=$1
name=$2
expr=$3

out=$(
  cat <<EOF | "$top" -noprompt -no-version 2>&1
open Correspondence.Tagged;;
let check (g : Graph_ir.graph) =
  match (Snapshot.create g, Snapshot.create g) with
  | Ok (Snapshot.Pack sa), Ok (Snapshot.Pack sb) -> (
      match
        of_clusters ~src:(Snapshot.edges sa) ~dst:(Snapshot.edges sb) []
      with
      | Error _ -> ()
      | Ok m -> (
          match
            ( Snapshot.edge sa (Tensor_id.of_int 0),
              Snapshot.edge sb (Tensor_id.of_int 0) )
          with
          | Some s, Some d ->
              ignore s;
              ignore d;
              ignore m;
              $expr
          | _ -> ()))
  | _ -> ();;
EOF
)

case "$out" in
*"val check :"*) printf '%s: COMPILES\n' "$name" ;;
*) printf '%s: rejected\n' "$name" ;;
esac

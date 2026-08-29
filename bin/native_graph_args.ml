(* Cmdliner argument/converter terms shared across the native_graph
   subcommand files, or specific to one of them but grouped here with the
   rest of the CLI's argument surface. Split out of native_graph.ml; see native_graph_common.ml for non-argument shared helpers. *)

open Cmdliner

(* An option (not a bare positional) since [print] is meant to grow other
   graph sources (e.g. a serialized native graph) as sibling options
   alongside this one. *)
let pt2_arg =
  let doc = "Path to the exported .pt2 model archive to convert." in
  Arg.(required & opt (some file) None & info [ "pt2" ] ~doc ~docv:"MODEL.pt2")

let input_arg =
  let doc = "Path to an external torch-save input tensor map." in
  Arg.(required & opt (some file) None & info [ "input" ] ~docv:"INPUT.pt" ~doc)

(* [transform] prints the rewritten graph when given no input, so that seeing
   what a pipeline produced does not require running inference over it. *)
let optional_input_arg =
  let doc =
    "Path to an external torch-save input tensor map. Omit to print the \
     transformed graph instead of evaluating it."
  in
  Arg.(value & opt (some file) None & info [ "input" ] ~docv:"INPUT.pt" ~doc)

let expect_arg =
  let doc = "Native JSON tensor written by aten_graph_ref for comparison." in
  Arg.(
    value & opt (some file) None & info [ "expect" ] ~docv:"REFERENCE.json" ~doc)

let verbose_arg =
  let doc =
    "Print each evaluated native node and its CPU compute time to stderr."
  in
  Arg.(value & flag & info [ "verbose" ] ~doc)

let fold_arg =
  let doc =
    "Load every captured weight up front so constant folding can hoist \
     parameter arithmetic to load time. Reads the whole archive."
  in
  Arg.(value & flag & info [ "fold" ] ~doc)

let verify_arg =
  let doc =
    "Also evaluate the untransformed graph and report the distance between the \
     two outputs. Requires --input."
  in
  Arg.(value & flag & info [ "verify" ] ~doc)

(* A different check from --verify, and complementary. That one runs the whole
   model twice on real weights and compares the two OUTPUTS; this one checks
   every corresponding edge of every pass's mapping symbolically, without
   payloads for the graph inputs, so what it says holds for every input rather
   than for the one tensor supplied.

   Each edge is a LOCAL obligation: "proved" means the transformation computes
   the same function of its corresponding dependencies, not that the whole graph
   still computes the same values. The summary is the conjunction over every
   cluster, which is what a [Policy] enforces — a proved edge downstream of a
   refuted one is expected, not a contradiction.

   The conjunction is still not output equality: a vacuous cluster counts as
   satisfied, and nothing checks that the two graphs' OUTPUTS are covered by
   non-vacuous clusters. See .ai/native_transform_verify.md §1.

   It is budget-capped, so a real model's activation-shaped clusters come back
   "too large" and the useful coverage is the constant-shaped ones — folded
   weights and biases, exactly where fold_const and fold_batch_norm act.
   See .ai/native_transform_verify.md and
   .ai/native_transform_local_verify_plan.md §§1-3. *)
let effort_conv =
  let parse s =
    match Map_verify.Effort.of_string s with
    | Ok e -> Ok e
    | Error (`Unknown_effort other) ->
        Error (`Msg (Printf.sprintf "unknown effort %S" other))
  in
  Arg.conv (parse, fun fmt e -> Map_verify.Effort.pp fmt e)

let verify_symbolic_arg =
  let doc =
    Printf.sprintf
      "Symbolically verify each pass's mapping as it is applied, at EFFORT \
       (%s). Needs no --input, and fails only on an actual counterexample."
      (String.concat ", "
         (List.map Map_verify.Effort.to_string Map_verify.Effort.all))
  in
  Arg.(
    value
    & opt (some effort_conv) None
    & info [ "verify-symbolic" ] ~docv:"EFFORT" ~doc)

(* Per group, then the roll-up. Groups are what a reader recognises in a real
   model — "layer1.0", "features.3" — so a report over 170 clusters is only
   legible attributed to them. The roll-up counts by outcome AND reason, since
   "40 unproved" says nothing without "because too large". *)
let limits_conv =
  (* Only the two WIRE-selectable profiles, which is what makes
     [Me_limits.Wire_limits] a guarantee rather than a UI convention: [large]
     and [trusted] exist for callers holding data they produced and are not
     offered here. *)
  Arg.enum
    [
      ("untrusted", Me_limits.Limits.untrusted);
      ("small", Me_limits.Limits.small);
    ]

let limits_arg =
  let doc = "Resource profile: $(b,untrusted) (the default) or $(b,small)." in
  Arg.(
    value & opt limits_conv Me_limits.Limits.untrusted & info [ "limits" ] ~doc)

let output_arg =
  let doc = "Write the session to this path instead of stdout." in
  Arg.(value & opt (some string) None & info [ "output" ] ~doc ~docv:"FILE")

let format_arg =
  let doc =
    "$(b,session) (the default) emits the full document; $(b,collections) \
     emits the bare graph-collection array, which is LOSSY -- comparisons, \
     node data, capabilities and the flow are discarded, and a warning names \
     what went."
  in
  Arg.(
    value
    & opt
        (Arg.enum [ ("session", `Session); ("collections", `Collections) ])
        `Session
    & info [ "format" ] ~doc)

let model_arg =
  let doc = "Path to the model: a .pt2 archive or an exported model.json." in
  Arg.(
    required & opt (some file) None & info [ "model"; "pt2" ] ~doc ~docv:"MODEL")

(* The session builder lives in [Me_export], not here: the browser compiles
   that library, and a second implementation in [bin/] is exactly the
   divergence between "what the CLI exports" and "what the page renders" that
   putting it in a library prevents. What stays here is the shell -- the
   filesystem, the profile flag, and the two output formats. *)
let parent_arg =
  let doc = "The graph holding the value, e.g. $(b,g/kernel/000)." in
  Arg.(required & opt (some string) None & info [ "graph" ] ~doc ~docv:"GRAPH")

let value_arg =
  let doc = "The kernel value's tensor id." in
  Arg.(required & opt (some int) None & info [ "value" ] ~doc ~docv:"N")

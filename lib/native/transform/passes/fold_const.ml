(* Materialise a node whose operands are all constants.
   See .ai/native_transform_design.md §12b.

   This is what lets every other pass stay simple: constant arithmetic is
   expressed as ordinary graph ops — a relayout emits a [Permute] of a weight,
   batch-norm folding emits the parameter arithmetic as nodes — and this one pass
   turns the resulting all-constant sub-DAG into data. Under [Pass.fixpoint] it
   collapses a whole sub-DAG, one layer per sweep, because a node only becomes
   foldable once the sweep before it bound its operands.

   Evaluation reuses [Region.extract] and [Eval_direct], so there is no second
   evaluation path that could disagree with the op it replaces. *)

open Graph_ir

(* A materialized fold is the empirical source of truth for Const-SSA coverage:
   it has already passed shape validation and [Eval_direct], so its operation
   and output signature describe exactly one static operation the old path was
   willing to cache. Operand ids are intentionally rendered as [_]; a corpus
   manifest must be stable under SSA numbering while retaining every static
   configuration parameter. *)
module Trace = struct
  type event = { op : Graph_ir.op; config : string; output : string }

  (* [Format.asprintf] uses its ordinary line width, which would turn one
     manifest record into several physical lines for a large tensor. Render each
     field with an intentionally huge margin instead: the on-disk trace is a
     line-oriented, diffable protocol, not a human-facing graph layout. *)
  let one_line pp value =
    let buffer = Buffer.create 96 in
    let fmt = Format.formatter_of_buffer buffer in
    Format.pp_set_margin fmt 1_000_000;
    pp fmt value;
    Format.pp_print_flush fmt ();
    Buffer.contents buffer

  let output_text output =
    let (Payload.Fmt packed) = output.Tensor_sig.fmt in
    one_line
      (fun fmt shape ->
        Fmt.pf fmt "%s %a" (Payload.fmt_name packed) Vec6.pp_shape shape)
      output.shape

  let make ~op ~output =
    {
      op;
      config = one_line (Graph_ir.pp_op_with ~pp_ref:(Fmt.any "_")) op;
      output = output_text output;
    }

  let compare a b =
    match String.compare (Graph_ir.op_name a.op) (Graph_ir.op_name b.op) with
    | 0 -> (
        match String.compare a.config b.config with
        | 0 -> String.compare a.output b.output
        | n -> n)
    | n -> n

  let canonical events = List.sort_uniq compare events
  let op e = e.op

  let pp fmt e =
    Fmt.pf fmt "@[<h>%s | %s | %s@]" (Graph_ir.op_name e.op) e.config e.output
end

(* Exactly one output and at least one operand. Zero outputs is [Discard], whose
   removal is dead-code elimination's business; several outputs would need every
   one of them bound in a single recipe, which [Recipe.fold_to_constant] does not
   express — left as future work rather than half-done.

   A constant with no bound payload is skipped rather than treated as an error:
   payloads live outside the IR (§2), so "declared constant, not yet loaded" is a
   legitimate state, and the node simply stays in the graph. *)
let foldable ~constant_store ~view (n : node) =
  match (n.Node.outputs, Graph_ir.operands n.Node.op) with
  | [ out ], (_ :: _ as operands)
    when List.for_all
           (fun id ->
             Graph_view.is_constant view id
             && Constant_store.is_effective_constant constant_store id)
           operands ->
      Option.map
        (fun output -> (out, output, operands))
        (Graph_view.sig_of view out)
  | _ -> None

(* [None] is unreachable given what [foldable] checked and what the view already
   validated — the region is a known singleton, shapes and arities were verified
   by [Graph_view.of_graph], the extracted graph has no [Input]-kind inputs since
   every operand is constant, and every constant has a payload. Skipping is the
   safe response to a case that cannot arise: the graph keeps computing the node
   itself. *)
let evaluate ~constant_store ~constants ~view (n : node) out =
  match Region.of_nodes view (Node_id.Set.singleton n.Node.id) with
  | Error _ -> None
  | Ok region -> (
      let g = Region.extract view region in
      (* A recipe-created literal intentionally has no cache payload: it is
         symbolic state, not a preload.  The old numeric fallback still needs
         bytes when a materialized operand shares the fold, so materialize just
         this region's plan inputs into an ephemeral cache. Captures already
         lacking bytes make the fallback decline as before. *)
      let constants =
        match
          Const_ssa_materialize.materialize
            ~needed:(Tensor_id.Set.of_list g.Graph.inputs)
            (fun capture -> Err.fail (`Missing_capture capture))
            constant_store
        with
        | Ok (store, _) ->
            Tensor_id.Map.union
              (fun _ old _ -> Some old)
              constants
              (Constant_store.materialized store)
        | Error _ -> constants
      in
      let payloads =
        List.filter_map
          (fun id ->
            Option.map (fun p -> (id, p)) (Tensor_id.Map.find_opt id constants))
          g.Graph.inputs
      in
      match Eval_direct.run g ~constants:payloads ~inputs:[] with
      | Error _ -> None
      (* [Eval_direct] returns every edge, so the value is there whether or not
         the folded node's output escaped the region. *)
      | Ok env -> Tensor_id.Map.find_opt out env)

let on_node : type v.
    (Trace.event -> unit) -> Pass.env -> node -> (v, unit) Recipe.t option =
 fun record { constant_store; constants; view } n ->
  Option.bind (foldable ~constant_store ~view n) (fun (out, output, operands) ->
      let open Recipe in
      if
        List.for_all
          (fun operand ->
            Option.is_some (Constant_store.binding constant_store operand))
          operands
      then
        Some
          (let* out = existing out in
           let* sources = Recipe.all existing operands in
           fold_to_deferred ~node:n.Node.id ~output:out ~op:n.Node.op ~sources)
      else
        Option.map
          (fun value ->
            record (Trace.make ~op:n.Node.op ~output);
            let open Recipe in
            let* out = existing out in
            let* sources = Recipe.all existing operands in
            fold_to_constant ~node:n.Node.id ~output:out ~value ~sources)
          (evaluate ~constant_store ~constants ~view n out))

let pass_with_trace record =
  let on_node' : type v. Pass.env -> node -> (v, unit) Recipe.t option =
   fun env node -> on_node record env node
  in
  Pass.per_node ~name:"fold_const" { Pass.on_node = on_node' }

let pass = pass_with_trace (fun _ -> ())

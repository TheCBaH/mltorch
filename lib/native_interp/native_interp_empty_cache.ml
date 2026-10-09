(* Empty-cache normalization: the one use of a zero-length tensor the static
   decode graphs make, resolved before lowering so [Native] keeps its positive
   extents.

   An exporter lifts the initial (empty) key/value cache to a [Tensor_constant]
   of shape [0], clones it, and concatenates it with the step's new keys. ATen's
   [cat] skips a 1-D size-0 operand whatever the other operands' rank, so the
   concatenation is the new keys alone. Exactly that, and nothing else, is
   rewritten:

   - a [Tensor_constant] whose metadata is exactly [[0]] is an empty source;
   - [clone.default] of an empty is empty (and is dropped);
   - [cat.default] drops its empty operands, keeping the others in order; one
     remaining operand becomes a [clone.default] (ATen's [cat] of one tensor is
     a copy);
   - any other reader, a graph output, a mutating signature, an all-empty cat
     or a dtype that differs from the remaining operands (ATen promotes over
     the skipped operand too) is refused.

   The result is a new program; the original, its digest and the capture
   inventory are untouched. The report names each source's disposition so the
   rewrite is auditable. A source nothing reads is dropped and reported
   unused. *)

open Pytorch_types
open Schema_runtime
module Fault = Native_interp_error.Empty_cache

module Report = struct
  (* A position in a cat's original operand list: not an extent, an axis or a
     count, and nothing else's [int] can be passed for it. *)
  module Operand =
    Core.Tagged_int.Make
      (struct
        let prefix = "operand"
      end)
      ()

  type cat = { cat : string; removed : Operand.t list }
  type source = { ssa : string; clones : string list; cats : cat list }
  type t = { sources : source list }
end

let clone_target = "torch.ops.aten.clone.default"
let cat_target = "torch.ops.aten.cat.default"

(* Every tensor name an argument reads. *)
let arg_reads (a : Argument.t) =
  match a with
  | Argument.Tensor t -> [ t.TensorArgument.name ]
  | Argument.Optional_tensor (OptionalTensorArgument.Tensor t) ->
      [ t.TensorArgument.name ]
  | Argument.Tensors ts -> List.map (fun t -> t.TensorArgument.name) ts
  | Argument.Optional_tensors ts ->
      List.filter_map
        (function
          | OptionalTensorArgument.Tensor t -> Some t.TensorArgument.name
          | OptionalTensorArgument.None _ -> None)
        ts
  | _ -> []

let node_reads (n : Node.t) =
  List.concat_map (fun (i : NamedArgument.t) -> arg_reads i.arg) n.Node.inputs

let is_zero_vector (m : TensorMeta.t) =
  match m.TensorMeta.sizes with [ SymInt.Int 0 ] -> true | _ -> false

type state = {
  origin : string String_map.t;  (** empty SSA name -> its source *)
  clones : string list String_map.t;  (** source -> clones, reversed *)
  cats : Report.cat list String_map.t;  (** source -> sites, reversed *)
}

let push key x m =
  String_map.add key
    (x :: Option.value ~default:[] (String_map.find_opt key m))
    m

let normalize (program : ExportedProgram.t) =
  Err.Escape.with_escape @@ fun esc ->
  Err.Escape.or_throw esc
  @@
  let fail e = Err.Escape.throw esc (`Empty_cache e) in
  let gm = program.ExportedProgram.graph_module in
  let graph = gm.GraphModule.graph in
  let sign = gm.GraphModule.signature in
  let meta name =
    match String_map.find_opt name graph.Graph.tensor_values with
    | Some m -> m
    | None -> fail (Fault.Missing_metadata name)
  in
  let sources =
    List.filter_map
      (function
        | InputSpec.Tensor_constant p
          when is_zero_vector (meta p.arg.TensorArgument.name) ->
            Some p.arg.TensorArgument.name
        | _ -> None)
      sign.GraphSignature.input_specs
  in
  if sources = [] then Err.return (program, { Report.sources = [] })
  else begin
    if
      List.exists
        (function OutputSpec.User_output _ -> false | _ -> true)
        sign.GraphSignature.output_specs
    then fail Fault.Mutating_signature;
    let st =
      ref
        {
          origin =
            List.fold_left
              (fun m s -> String_map.add s s m)
              String_map.empty sources;
          clones = String_map.empty;
          cats = String_map.empty;
        }
    in
    let source_of name = String_map.find_opt name !st.origin in
    let rewrite (n : Node.t) : Node.t option =
      let reads = node_reads n in
      let empties = List.filter (fun r -> source_of r <> None) reads in
      if empties = [] then Some n
      else if n.Node.target = clone_target then
        match n.Node.inputs with
        | [ { NamedArgument.name = "self"; arg = Argument.Tensor t; _ } ]
        | [
            { NamedArgument.name = "self"; arg = Argument.Tensor t; _ };
            { NamedArgument.name = "memory_format"; arg = Argument.None _; _ };
          ] -> (
            match (source_of t.TensorArgument.name, n.Node.outputs) with
            | Some root, [ Argument.Tensor out ] ->
                st :=
                  {
                    !st with
                    origin =
                      String_map.add out.TensorArgument.name root !st.origin;
                    clones = push root out.TensorArgument.name !st.clones;
                  };
                None
            | _ ->
                fail
                  (Fault.Unsupported_reader
                     { source = List.hd empties; target = n.Node.target }))
        | _ ->
            fail
              (Fault.Unsupported_reader
                 { source = List.hd empties; target = n.Node.target })
      else if n.Node.target = cat_target then (
        let out_name =
          match n.Node.outputs with
          | [ Argument.Tensor out ] -> out.TensorArgument.name
          | _ -> fail (Fault.Missing_metadata (List.hd empties))
        in
        let tensors_input, tensors =
          match
            List.find_opt
              (fun (i : NamedArgument.t) -> i.name = "tensors")
              n.Node.inputs
          with
          | Some ({ arg = Argument.Tensors ts; _ } as i) -> (i, ts)
          | _ ->
              fail
                (Fault.Unsupported_reader
                   { source = List.hd empties; target = n.Node.target })
        in
        let indexed = List.mapi (fun i t -> (i, t)) tensors in
        let is_empty (_, t) = source_of t.TensorArgument.name <> None in
        let removed = List.filter is_empty indexed in
        let kept = List.filter (fun x -> not (is_empty x)) indexed in
        (* Another input (dim, ...) never reads a tensor, so every empty read
           is an operand of [tensors]. *)
        if kept = [] then fail (Fault.All_empty_cat out_name);
        let _, first_kept = List.hd kept in
        let kept_dtype = (meta first_kept.TensorArgument.name).dtype in
        List.iter
          (fun (_, e) ->
            if (meta e.TensorArgument.name).dtype <> kept_dtype then
              fail
                (Fault.Dtype_mismatch
                   {
                     empty = e.TensorArgument.name;
                     other = first_kept.TensorArgument.name;
                   }))
          removed;
        List.iter
          (fun (pos, e) ->
            let root = Option.get (source_of e.TensorArgument.name) in
            st :=
              {
                !st with
                cats =
                  push root
                    {
                      Report.cat = out_name;
                      removed = [ Report.Operand.of_int pos ];
                    }
                    !st.cats;
              })
          removed;
        let kept_ts = List.map snd kept in
        match kept_ts with
        | [ only ] ->
            Some
              {
                n with
                Node.target = clone_target;
                inputs =
                  [
                    {
                      NamedArgument.name = "self";
                      arg = Argument.Tensor only;
                      kind = tensors_input.kind;
                    };
                  ];
              }
        | _ ->
            Some
              {
                n with
                Node.inputs =
                  List.map
                    (fun (i : NamedArgument.t) ->
                      if i.name = "tensors" then
                        { i with arg = Argument.Tensors kept_ts }
                      else i)
                    n.Node.inputs;
              })
      else
        fail
          (Fault.Unsupported_reader
             { source = List.hd empties; target = n.Node.target })
    in
    let nodes = List.filter_map rewrite graph.Graph.nodes in
    List.iter
      (fun a ->
        List.iter
          (fun r ->
            if source_of r <> None then fail (Fault.Escapes_to_output r))
          (arg_reads a))
      graph.Graph.outputs;
    let dropped name = List.mem name sources in
    let keep_arg a =
      match a with
      | Argument.Tensor t -> not (dropped t.TensorArgument.name)
      | _ -> true
    in
    let input_specs =
      List.filter
        (function
          | InputSpec.Tensor_constant p ->
              not (dropped p.arg.TensorArgument.name)
          | _ -> true)
        sign.GraphSignature.input_specs
    in
    let program' =
      {
        program with
        ExportedProgram.graph_module =
          {
            gm with
            GraphModule.graph =
              {
                graph with
                Graph.nodes;
                inputs = List.filter keep_arg graph.Graph.inputs;
              };
            signature = { sign with GraphSignature.input_specs };
          };
      }
    in
    let rev m s =
      List.rev (Option.value ~default:[] (String_map.find_opt s m))
    in
    Err.return
      ( program',
        {
          Report.sources =
            List.map
              (fun ssa ->
                {
                  Report.ssa;
                  clones = rev !st.clones ssa;
                  cats = rev !st.cats ssa;
                })
              sources;
        } )
  end

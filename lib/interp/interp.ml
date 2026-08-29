(* Graph interpreter for exported `.pt2` models. Walks the ExportedProgram graph
   node by node, decoding each node's named arguments and dispatching to the
   bound ATen ops (Aten_c.Aten_operations). Tensors flow as raw atc_tensor
   handles; SSA value names are kept in an immutable environment that is threaded
   through the node fold.

   Argument decoding, output binding, and the environment live in Interp_decode;
   per-node dispatch (Interp_dispatch.dispatch) is generated from the ATen schema
   by bin/aten_ops_gen. This module is just the driver around them.

   Only the parameters/buffers actually referenced by some node are loaded (see
   [referenced_tensor_names]), so unused buffers (e.g. the int64
   num_batches_tracked, which no node reads) never reach the float32-only
   bridge. *)

open Ctypes
open Pytorch_types
open Schema_runtime
open Interp_decode
open Err.Syntax
module O = Aten_c.Aten_operations
module TG = Aten_types_generated

let tget = Aten_operation_description.tensors_get

type error =
  [ `Aten_runtime_failure of string * int
  | Interp_decode.error
  | Pt2_archive.error
  | `Topk_read_failed
  | `Unexpected_graph_output of string
  | `Unhandled_op of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Aten_runtime_failure (op, st) ->
      Fmt.pf ppf "ATen op %s failed with status %d" op st
  | #Interp_decode.error as e -> Interp_decode.pp_error ppf e
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | `Topk_read_failed ->
      Fmt.string ppf "failed to read topk outputs back from ATen"
  | `Unexpected_graph_output summary ->
      Fmt.pf ppf "expected a single tensor graph output, got %s" summary
  | `Unhandled_op target -> Fmt.pf ppf "unhandled op %S" target

(* --- driver --- *)

(* Every value name used as a tensor argument anywhere in the graph. Parameter
   names appearing here are exactly the ones to load; names that turn out to be
   node results or user inputs are filtered out against the params map. *)
let referenced_tensor_names (g : Graph.t) =
  List.concat_map
    (fun (node : Node.t) ->
      List.concat_map
        (fun (na : NamedArgument.t) ->
          match na.arg with
          | Argument.Tensor ta -> [ ta.TensorArgument.name ]
          | Argument.Optional_tensor (OptionalTensorArgument.Tensor ta) ->
              [ ta.TensorArgument.name ]
          | Argument.Tensors tas ->
              List.map (fun (ta : TensorArgument.t) -> ta.name) tas
          | _ -> [])
        node.Node.inputs)
    g.Graph.nodes

let graph_output_summary outputs =
  match outputs with
  | [ output ] -> Interp_decode.argument_kind_name output
  | outputs -> Printf.sprintf "%d outputs" (List.length outputs)

(* The graph and its I/O signature out of an opened archive — the starting
   point shared by [run] and any other node-by-node walk (e.g.
   Native_graph). *)
let graph_and_signature archive =
  let prog = Pt2_archive.program archive in
  ( prog.ExportedProgram.graph_module.GraphModule.graph,
    prog.ExportedProgram.graph_module.GraphModule.signature )

(* graph input name -> captured payload name, for every tensor source the
   inference runtime can bind before execution. *)
let param_names (sign : GraphSignature.t) =
  List.fold_left
    (fun params (spec : InputSpec.t) ->
      match spec with
      | InputSpec.Parameter p ->
          String_map.add p.InputToParameterSpec.arg.TensorArgument.name
            p.InputToParameterSpec.parameter_name params
      | InputSpec.Buffer b ->
          String_map.add b.InputToBufferSpec.arg.TensorArgument.name
            b.InputToBufferSpec.buffer_name params
      | InputSpec.Tensor_constant c ->
          String_map.add c.InputToTensorConstantSpec.arg.TensorArgument.name
            c.InputToTensorConstantSpec.tensor_constant_name params
      | _ -> params)
    String_map.empty sign.GraphSignature.input_specs

(* Load exactly the captured tensors referenced by some node of [g] into
   [env], so unused buffers (e.g. the int64 num_batches_tracked) never reach
   the float32 bridge. Names [env] already binds (e.g. user inputs) are left
   untouched. *)
let load_referenced_params archive (g : Graph.t) (sign : GraphSignature.t) env =
  let params = param_names sign in
  Err.List.fold_left
    (fun env name ->
      if String_map.mem name env then Err.return env
      else
        match String_map.find_opt name params with
        | Some cfg_name ->
            let* weight =
              Pt2_archive.load_captured_tensor archive cfg_name
              |> Err.map_error (fun e -> (e :> error))
            in
            Err.return (String_map.add name (Pt2_aten.to_tensor weight) env)
        | None -> Err.return env)
    env
    (referenced_tensor_names g)

let run archive (image : Pt2_tensor.t) =
  let g, sign = graph_and_signature archive in
  (* User inputs are bound to the supplied image. *)
  let env =
    List.fold_left
      (fun env (spec : InputSpec.t) ->
        match spec with
        | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor ta; _ } ->
            String_map.add ta.TensorArgument.name (Pt2_aten.to_tensor image) env
        | _ -> env)
      String_map.empty sign.GraphSignature.input_specs
  in
  let* env = load_referenced_params archive g sign env in
  let* env =
    Err.List.fold_left
      (fun env node ->
        Interp_dispatch.dispatch env node
        |> Err.map_error (fun e -> (e :> error)))
      env g.Graph.nodes
  in
  match g.Graph.outputs with
  | [ Argument.Tensor ta ] -> resolve env ta.TensorArgument.name
  | outputs ->
      Err.fail (`Unexpected_graph_output (graph_output_summary outputs))

(* Index of the maximum element of the flattened logits, computed by ATen
   (at::argmax with dim=None) and read back as an int. *)
let none_int = from_voidp int64_t null

(* [item_int] reads through the shim and never manages, so the result has to be
   managed here or the handle lives until the process exits. *)
let argmax logits =
  Err.return
    (Aten_tensor.item_int
       (O.argmax logits none_int false
       |> Aten_tensor.check |> Aten_tensor.manage))

(* The [k] highest-scoring (class index, probability) pairs of [logits]
   ([1; classes]), descending: softmax over the last dim, then at::topk. *)
let top_predictions logits k =
  (* Own every handle before reading it: [Aten_tensor.data]'s finaliser only
     anchors an already-registered handle for the Bigarray's lifetime, it never
     frees, so an unmanaged op result here leaks. Materialize after managing —
     topk's outputs are dense today, so the fast path returns them unchanged and
     "manage exactly once" still holds; were that to change, the original stays
     managed and the helper independently manages its clone. *)
  let probs =
    O._softmax logits 1L false |> Aten_tensor.check |> Aten_tensor.manage
  in
  let out = make TG.tensors2_struct in
  let st = O.topk probs (Int64.of_int k) (-1L) true true (addr out) in
  if st <> 0 then Err.fail (`Aten_runtime_failure ("topk", st))
  else
    let own v =
      tget out v |> Aten_tensor.check |> Aten_tensor.manage
      |> Aten_tensor.materialize_for_raw_read
    in
    match
      ( Aten_tensor.data Aten_dtype.float32 (own TG.tensors2_v0),
        Aten_tensor.data Aten_dtype.int64 (own TG.tensors2_v1) )
    with
    | Some vs, Some idx ->
        Err.return
          (List.init (Bigarray.Array1.dim idx) (fun i ->
               (Int64.to_int idx.{i}, vs.{i})))
    | _ -> Err.fail `Topk_read_failed

(* Drives bin/native_graph's `print` subcommand: walk a real exported `.pt2`
   model's ATen graph node by node and, for each node, print the native
   engine's translation of it (the relayout + core-op [Graph_ir.graph] that
   [Op_bridge.dispatch] already builds for ATen-vs-native verification — see
   .ai/native_aten_bridge_layout.md).

   Only the *structure* is wanted here, not a real inference result, so every
   graph-level user input is a freshly allocated, uninitialised ATen tensor of
   the shape/dtype recorded in the exported graph's own [tensor_values]
   metadata (no sample image required). Parameters/buffers are still loaded
   for real (via [Interp.load_referenced_params]) since later nodes' relayouts
   (e.g. conv weight OIHW->native) read their actual shapes.

   Op coverage is exactly [Op_bridge]'s (see .ai/native_aten_bridge_layout.md's
   two tables): a node whose target has no native mapping, or whose decode/
   shape-validation fails, stops the walk immediately with a typed error —
   this is a coverage report, not a best-effort partial dump. *)

open Pytorch_types
open Schema_runtime
open Core.Syntax

type error =
  [ Interp.error
  | Pt2_dtype.error
  | `Symbolic_shape of string
  | `No_tensor_meta of string
  | `No_native_impl of string
  | `Op_bridge_error of Op_bridge.error ]

let pp_error ppf : [< error ] -> unit = function
  | #Interp.error as e -> Interp.pp_error ppf e
  | #Pt2_dtype.error as e -> Pt2_dtype.pp_error ppf e
  | `Symbolic_shape name ->
      Format.fprintf ppf "%s: dynamic (symbolic) shape not supported" name
  | `No_tensor_meta name ->
      Format.fprintf ppf "no tensor_values metadata for %S" name
  | `No_native_impl target ->
      Format.fprintf ppf "no native implementation for %S" target
  | `Op_bridge_error e -> Op_bridge.pp_error ppf e

(* Routed through [Pt2_dtype] (the module that already owns the
   [ScalarType.t] enum and its "unsupported" error/printer) rather than
   re-deciding coverage here; only the [Pt2_dtype.t -> Aten_scalar_type.t]
   step is this module's own. *)
let aten_dtype_of_scalar_type (st : ScalarType.t) :
    (Aten_scalar_type.t, error) Core.result =
  let* dtype =
    Pt2_dtype.of_scalar_type st |> Core.map_error (fun e -> (e :> error))
  in
  Core.return
    (match dtype with
    | Pt2_dtype.Float32 -> Aten_scalar_type.Float
    | Float64 -> Double
    | Int64 -> Long
    | Int32 -> Int
    | Int16 -> Short
    | Int8 -> Char
    | UInt8 -> Byte
    | Bool -> Bool)

let concrete_sizes name (sizes : SymInt.t list) =
  Core.List.map
    (function
      | SymInt.Int i -> Core.return i
      | SymInt.Expr _ -> Core.fail (`Symbolic_shape name))
    sizes

(* A fresh, correctly shaped/typed but uninitialised ATen tensor for graph
   input [name], taken from the exported graph's own [tensor_values]: only
   shapes need to flow through the native conversion, not real data. *)
let synth_tensor (g : Graph.t) name =
  match String_map.find_opt name g.Graph.tensor_values with
  | None -> Core.fail (`No_tensor_meta name)
  | Some (meta : TensorMeta.t) ->
      let* sizes = concrete_sizes name meta.TensorMeta.sizes in
      let* dtype = aten_dtype_of_scalar_type meta.TensorMeta.dtype in
      Core.return (Aten_tensor.create ~dtype sizes)

let initial_env archive =
  let g, sign = Interp.graph_and_signature archive in
  let* env =
    Core.List.fold_left
      (fun env (spec : InputSpec.t) ->
        match spec with
        | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor ta; _ } ->
            let* t = synth_tensor g ta.TensorArgument.name in
            Core.return (String_map.add ta.TensorArgument.name t env)
        | _ -> Core.return env)
      String_map.empty sign.GraphSignature.input_specs
  in
  let* env =
    Interp.load_referenced_params archive g sign env
    |> Core.map_error (fun e -> (e :> error))
  in
  Core.return (g, env)

let node_label (node : Node.t) =
  match node.Node.name with
  | Some name -> Printf.sprintf "%s (%s)" node.Node.target name
  | None -> node.Node.target

let print_node ~verbose ppf env (node : Node.t) =
  match Op_bridge.dispatch ~aten_env:env node with
  | None -> Core.fail (`No_native_impl node.Node.target)
  | Some (Error e) -> Core.map_error (fun k -> `Op_bridge_error k) (Error e)
  | Some (Ok (g, _bindings)) ->
      if verbose then
        Format.fprintf ppf "@[<v>=== %s ===@,%a@]@." (node_label node)
          Graph_ir.pp g
      else Format.fprintf ppf "@[<v>%a@]@." Graph_ir.pp g;
      Core.return ()

(* Walk every node of [archive]'s exported graph, printing each one's native
   translation to [ppf] in program order. Threads the real ATen env forward
   via [Interp_dispatch.dispatch] (not just [Op_bridge]) so later nodes see
   the actual shapes their operands end up with. [verbose] additionally
   prints each node's source ATen target/name before its translation;
   without it, only the translated graphs themselves are printed. *)
let run ~verbose ppf archive =
  let* g, env = initial_env archive in
  let* (_ : Interp_decode.env) =
    Core.List.fold_left
      (fun env node ->
        let* () = print_node ~verbose ppf env node in
        Interp_dispatch.dispatch env node
        |> Core.map_error (fun e -> (e :> error)))
      env g.Graph.nodes
  in
  Core.return ()

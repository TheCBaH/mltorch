(* Run a decoded verification spec through the ATen-vs-native harness.

   [to_node] synthesizes each tensor input (one PCG instance, seeded by the
   spec's [seed], threaded through every draw so the case is reproducible) and
   lowers the arguments to a single-node Pytorch_types.Node — exactly the shape
   the interpreter and the verifier already consume. [run] then executes that
   node on both paths (Interp_dispatch for ATen, Op_bridge for native) and
   compares the outputs with Verify, returning whether they agreed.

   The number of output tensors to allocate for the comparison comes from the
   generated Aten_op_config (the same op's return arity). *)

open Pytorch_types
open Schema_runtime
module Tspec = Aten_spec.Tensor_spec
module Sv = Aten_spec.Scalar_value
module Av = Aten_spec.Arg_value
module Pcg = Aten_spec.Pcg

let numel shape = List.fold_left ( * ) 1 shape

let scalar_to_float = function
  | Sv.Int i -> Aten_spec.Float32.to_f32 (float_of_int i)
  | Sv.Float f -> f
  | Sv.Bool b -> if b then 1.0 else 0.0

(* Sample one f32 from a distribution, advancing the RNG. *)
let sample_f32 (d : Aten_spec.Distribution.t) pcg =
  match d with
  | Uniform { low; high } -> Pcg.uniform ~low ~high pcg
  | Normal { mean; variance } -> Pcg.normal ~mean ~std:(sqrt variance) pcg

(* Fill a float32 Bigarray from a payload source, threading [pcg]. *)
let fill_f32 shape (source : Tspec.source) pcg =
  let n = numel shape in
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n in
  let pcg =
    match source with
    | Tspec.Values vs ->
        let arr = Array.of_list vs in
        if Array.length arr <> n then
          failwith
            (Printf.sprintf "spec: values has %d entries, shape needs %d"
               (Array.length arr) n);
        let rec go i =
          if i < n then (
            ba.{i} <- scalar_to_float arr.(i);
            go (i + 1))
        in
        go 0;
        pcg
    | Tspec.Sequence { start; step } ->
        let rec go i =
          if i < n then (
            ba.{i} <-
              Aten_spec.Float32.to_f32 (start +. (float_of_int i *. step));
            go (i + 1))
        in
        go 0;
        pcg
    | Tspec.Random dist ->
        let rec go i pcg =
          if i >= n then pcg
          else
            let v, pcg = sample_f32 dist pcg in
            ba.{i} <- v;
            go (i + 1) pcg
        in
        go 0 pcg
  in
  (pcg, ba)

(* Fill an int64 Bigarray (exact match path; no float distributions). *)
let fill_i64 shape (source : Tspec.source) pcg =
  let n = numel shape in
  let ba = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout n in
  (match source with
  | Tspec.Values vs ->
      let arr = Array.of_list vs in
      if Array.length arr <> n then
        failwith
          (Printf.sprintf "spec: values has %d entries, shape needs %d"
             (Array.length arr) n);
      let rec go i =
        if i < n then (
          ba.{i} <- Int64.of_float (scalar_to_float arr.(i));
          go (i + 1))
      in
      go 0
  | Tspec.Sequence { start; step } ->
      let rec go i =
        if i < n then (
          ba.{i} <- Int64.of_float (start +. (float_of_int i *. step));
          go (i + 1))
      in
      go 0
  | Tspec.Random _ ->
      failwith "spec: random payloads are only supported for f32 tensors");
  (pcg, ba)

let synthesize pcg (ts : Tspec.t) =
  match ts.dtype with
  | Aten_spec.Dtype.F32 ->
      let pcg, ba = fill_f32 ts.shape ts.source pcg in
      (pcg, Aten_tensor.of_bigarray Aten_dtype.float32 ba ts.shape)
  | Aten_spec.Dtype.I64 ->
      let pcg, ba = fill_i64 ts.shape ts.source pcg in
      (pcg, Aten_tensor.of_bigarray Aten_dtype.int64 ba ts.shape)
  | _ -> failwith "spec: tensor synthesis supports f32 and i64 only"

let scalar_to_arg = function
  | Sv.Int i -> Argument.Int i
  | Sv.Float f -> Argument.Float f
  | Sv.Bool b -> Argument.Bool b

(* Lower one argument: synthesize tensors (binding them in [env] under [name])
   and translate everything to a Pytorch_types.Argument. *)
let lower pcg env name (av : Av.t) =
  let tensor pcg env nm spec =
    let pcg, t = synthesize pcg spec in
    (pcg, String_map.add nm t env, TensorArgument.make nm)
  in
  match av with
  | Av.Tensor spec ->
      let pcg, env, ta = tensor pcg env name spec in
      (pcg, env, Argument.Tensor ta)
  | Av.Tensor_opt None -> (pcg, env, Argument.None false)
  | Av.Tensor_opt (Some spec) ->
      let pcg, env, ta = tensor pcg env name spec in
      (pcg, env, Argument.Tensor ta)
  | Av.Tensor_list specs ->
      let pcg, env, tas, _ =
        List.fold_left
          (fun (pcg, env, acc, k) spec ->
            let nm = Printf.sprintf "%s_%d" name k in
            let pcg, env, ta = tensor pcg env nm spec in
            (pcg, env, ta :: acc, k + 1))
          (pcg, env, [], 0) specs
      in
      (pcg, env, Argument.Tensors (List.rev tas))
  | Av.Int i -> (pcg, env, Argument.Int i)
  | Av.Int_opt None -> (pcg, env, Argument.None false)
  | Av.Int_opt (Some i) -> (pcg, env, Argument.Int i)
  | Av.Int_list xs -> (pcg, env, Argument.Ints xs)
  | Av.Int_list_opt None -> (pcg, env, Argument.None false)
  | Av.Int_list_opt (Some xs) -> (pcg, env, Argument.Ints xs)
  | Av.Float f -> (pcg, env, Argument.Float f)
  | Av.Float_opt None -> (pcg, env, Argument.None false)
  | Av.Float_opt (Some f) -> (pcg, env, Argument.Float f)
  | Av.Bool b -> (pcg, env, Argument.Bool b)
  | Av.Bool_opt None -> (pcg, env, Argument.None false)
  | Av.Bool_opt (Some b) -> (pcg, env, Argument.Bool b)
  | Av.Scalar sv -> (pcg, env, scalar_to_arg sv)
  | Av.Scalar_opt None -> (pcg, env, Argument.None false)
  | Av.Scalar_opt (Some sv) -> (pcg, env, scalar_to_arg sv)
  | Av.Str s -> (pcg, env, Argument.String s)

let outputs_for target =
  let arity =
    match Aten_op_config.find target with
    | Some c -> (
        match c.returns with
        | Aten_op_config.Single -> 1
        | Aten_op_config.Tuple2 -> 2
        | Aten_op_config.Tuple3 -> 3)
    | None -> 1
  in
  List.init arity (fun i ->
      Argument.Tensor (TensorArgument.make (Printf.sprintf "out%d" i)))

let to_node (spec : Aten_spec.Op_spec.t) =
  let pcg0 = Pcg.default in
  let _pcg, env, inputs_rev =
    List.fold_left
      (fun (pcg, env, acc) (name, av) ->
        let pcg, env, arg = lower pcg env name av in
        (pcg, env, NamedArgument.make name arg None :: acc))
      (pcg0, String_map.empty, [])
      spec.args
  in
  let node =
    Node.make spec.target (List.rev inputs_rev) (outputs_for spec.target)
      String_map.empty None (Some "spec")
  in
  (env, node)

(* Execute the spec on both paths and report. Returns [true] if the native and
   ATen outputs agreed (or the op has no native impl, which is reported and
   treated as a pass), [false] on mismatch or error. *)
let run ?(ppf = Format.std_formatter) (spec : Aten_spec.Op_spec.t) : bool =
  let env, node = to_node spec in
  let env' = Interp_dispatch.dispatch env node in
  match Op_bridge.dispatch ~aten_env:env node with
  | None ->
      Format.fprintf ppf "[spec] %s: skipped (no native impl)@." node.target;
      true
  | Some (Error msg) ->
      Format.fprintf ppf "[spec] %s: bridge error: %s@." node.target msg;
      false
  | Some (Ok (graph, bindings)) ->
      let result_env = Eval_direct.run graph ~inputs:bindings in
      let native_outputs =
        List.map
          (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
          graph.Graph_ir.Graph.outputs
      in
      let errors =
        Verify.verify_node ~atol:1e-5 ~aten_env:env' node native_outputs
      in
      if errors = [] then (
        Format.fprintf ppf "[spec] %s: matched@." node.target;
        true)
      else (
        Verify.report ppf node.target errors;
        false)

let pp_aten ppf t =
  match Aten_tensor.as_float32 t with
  | Some ba -> Aten_tensor.pp_float32 ppf ba
  | None -> Format.pp_print_string ppf "<non-f32>"

(* A native output is a dense 6D tensor; flatten it to a 1D ATen tensor (same
   element order, see native_aten_bridge_design) so it prints with the same
   formatter as the ATen output. *)
let pp_native ppf packed =
  match Tensor_bridge.to_aten_flat packed with
  | Ok t -> pp_aten ppf t
  | Error e -> Format.fprintf ppf "<error: %s>" e

(* Evaluate the spec on both paths and print each output tensor, ATen above
   native, so the two can be compared by eye. *)
let eval_print ?(ppf = Format.std_formatter) (spec : Aten_spec.Op_spec.t) : unit
    =
  let env, node = to_node spec in
  let env' = Interp_dispatch.dispatch env node in
  let out_names =
    List.filter_map
      (function Argument.Tensor ta -> Some ta.TensorArgument.name | _ -> None)
      node.outputs
  in
  let native =
    match Op_bridge.dispatch ~aten_env:env node with
    | None -> `None
    | Some (Error e) -> `Err e
    | Some (Ok (graph, bindings)) ->
        let result_env = Eval_direct.run graph ~inputs:bindings in
        `Ok
          (List.map
             (fun oid -> Graph_ir.Tensor_id.Map.find oid result_env)
             graph.Graph_ir.Graph.outputs)
  in
  Format.fprintf ppf "[eval] %s@." node.target;
  List.iteri
    (fun i name ->
      (match String_map.find_opt name env' with
      | Some t -> Format.fprintf ppf "  aten   %s = %a@." name pp_aten t
      | None -> Format.fprintf ppf "  aten   %s = <missing>@." name);
      match native with
      | `None ->
          if i = 0 then Format.fprintf ppf "  native = <no native impl>@."
      | `Err e -> if i = 0 then Format.fprintf ppf "  native = <error: %s>@." e
      | `Ok outs -> (
          match List.nth_opt outs i with
          | Some packed ->
              Format.fprintf ppf "  native %s = %a@." name pp_native packed
          | None -> Format.fprintf ppf "  native %s = <missing>@." name))
    out_names

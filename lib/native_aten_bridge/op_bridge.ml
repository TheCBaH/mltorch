(* Native-side dispatch: given a graph node and the ATen environment (inputs),
   compute the equivalent result using the pure-OCaml Native ops.

   Returns [None] if no native implementation exists for this op (the verify
   harness skips the node).  Returns [Some (Error msg)] if the op is mapped but
   tensor conversion or execution fails.  Returns [Some (Ok outputs)] on success.

   Coverage grows incrementally: start with simple pointwise ops, expand per
   .ai/native_aten_bridge_design.md.  The ATen env carries pre-dispatch inputs;
   the bridge converts them to Native tensors on the fly (no persistent native
   environment is maintained between nodes). *)

open Pytorch_types

(* Reuse interp_decode helpers for non-tensor arguments: the decoding logic
   (find_arg, int_arg, ints_arg, bool_arg) is identical to the ATen path. *)
module D = Interp_decode

type aten_env = Interp_decode.env

(* Decode a tensor argument from the ATen env, convert to Native.  Raises on
   missing arg; returns Error on conversion failure. *)
let native_tensor_arg aten_env node name =
  let t = D.tensor_arg aten_env node name in
  Tensor_bridge.of_aten t

(* Run a unary op that takes one tensor and returns one tensor. *)
let unary aten_env node ~f name_self =
  match native_tensor_arg aten_env node name_self with
  | Error e -> Some (Error (Printf.sprintf "%s: %s" name_self e))
  | Ok x -> (
      match f x with
      | exception Failure msg ->
          Some (Error (Printf.sprintf "native op failed: %s" msg))
      | result -> Some (Ok [ result ]))

(* Run a binary op that takes two tensors and returns one tensor. *)
let binary aten_env node ~f name_a name_b =
  match
    ( native_tensor_arg aten_env node name_a,
      native_tensor_arg aten_env node name_b )
  with
  | Error e, _ | _, Error e -> Some (Error e)
  | Ok a, Ok b -> (
      match f a b with
      | exception Failure msg ->
          Some (Error (Printf.sprintf "native op failed: %s" msg))
      | result -> Some (Ok [ result ]))

(* --- Direct scheduler helpers --- *)

let run_relu (x : Tensor.packed) : Tensor.packed =
  let module C = Pointwise.Relu.Compute (Direct) in
  let (Tensor.Tensor r) = x in
  let shape = Pointwise.Relu.output_shape r.shape in
  Schedule.evaluate shape (C.pixel x)

let run_add (a : Tensor.packed) (b : Tensor.packed) : Tensor.packed =
  let module C = Pointwise.Add.Compute (Direct) in
  let (Tensor.Tensor ra) = a in
  let (Tensor.Tensor rb) = b in
  let shape = Pointwise.Add.output_shape ra.shape rb.shape in
  Schedule.evaluate shape (C.pixel a b)

(* --- Op dispatch --- *)

(* Dispatch a graph node to the native path.  The [aten_env] contains the
   pre-dispatch ATen inputs (SSA name -> tensor handle).  Returns:
   - None              : no native implementation; skip verification
   - Some (Error msg)  : mapped but failed (conversion error or unsupported args)
   - Some (Ok outputs) : native outputs as packed tensors, same length and order
                         as the ATen op's outputs *)
let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Tensor.packed list, string) result option =
  match node.target with
  | "torch.ops.aten.relu.default" | "torch.ops.aten.relu_.default" ->
      unary aten_env node ~f:run_relu "self"
  | "torch.ops.aten.add.Tensor" | "torch.ops.aten.add_.Tensor" ->
      binary aten_env node ~f:run_add "self" "other"
  | _ -> None

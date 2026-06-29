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

(* Direct-scheduler runners in global alphabetical order. *)
let run_add (a : Tensor.packed) (b : Tensor.packed) : Tensor.packed =
  let module C = Pointwise.Add.Compute (Direct) in
  let (Tensor.Tensor ra) = a in
  let (Tensor.Tensor rb) = b in
  let shape = Pointwise.Add.output_shape ra.shape rb.shape in
  Schedule.evaluate shape (C.pixel ~a_shape:ra.shape ~b_shape:rb.shape a b)

let run_bmm (a : Tensor.packed) (b : Tensor.packed) : Tensor.packed =
  let module C = Matmul.Bmm.Compute (Direct) in
  let (Tensor.Tensor ra) = a in
  let (Tensor.Tensor rb) = b in
  let shape =
    Matmul.Bmm.output_shape ~input_shape:ra.shape ~mat2_shape:rb.shape
  in
  Schedule.evaluate shape (C.pixel ~input_shape:ra.shape ~input:a ~mat2:b)

let run_mean (x : Tensor.packed) ~dims ~keepdim : Tensor.packed =
  let module C = Reduce.Mean.Compute (Direct) in
  let (Tensor.Tensor r) = x in
  let p = { Reduce.Mean.dims; keepdim } in
  let shape = Reduce.Mean.output_shape ~x_shape:r.shape p in
  Schedule.evaluate shape (C.pixel p ~x_shape:r.shape ~x)

let run_mul (a : Tensor.packed) (b : Tensor.packed) : Tensor.packed =
  let module C = Pointwise.Mul.Compute (Direct) in
  let (Tensor.Tensor ra) = a in
  let (Tensor.Tensor rb) = b in
  let shape = Pointwise.Mul.output_shape ra.shape rb.shape in
  Schedule.evaluate shape (C.pixel ~a_shape:ra.shape ~b_shape:rb.shape a b)

let run_permute (x : Tensor.packed) (perm : Permute.Permute.perm) :
    Tensor.packed =
  let module C = Permute.Permute.Compute (Direct) in
  let (Tensor.Tensor r) = x in
  let shape = Permute.Permute.output_shape ~x_shape:r.shape perm in
  Schedule.evaluate shape (C.pixel perm ~x)

let run_relu (x : Tensor.packed) : Tensor.packed =
  let module C = Pointwise.Relu.Compute (Direct) in
  let (Tensor.Tensor r) = x in
  let shape = Pointwise.Relu.output_shape r.shape in
  Schedule.evaluate shape (C.pixel x)

let run_rms_norm (x : Tensor.packed) ~weight ~dims ~eps : Tensor.packed =
  let module C = Norm.RmsNorm.Compute (Direct) in
  let (Tensor.Tensor r) = x in
  let p = { Norm.RmsNorm.dims; eps } in
  let shape = Norm.RmsNorm.output_shape ~x_shape:r.shape in
  Schedule.evaluate shape (C.pixel p ~x_shape:r.shape ~x ~weight)

(* --- Argument decoding for the reductions/normalisation ---

   [mean.dim] and [rms_norm] reference frame axes through [Aten_shape.axis_of_dim],
   so the dims must be derived from the ATen input's RANK (not the right-aligned
   6D shape, whose outer padding axes would shift the mapping).  This keeps the
   reduced axes consistent with where [of_aten] positionally places the data, so
   no relayout is needed — see .ai/native_aten_bridge_layout.md. *)

let aten_rank t = Array.length (Aten_tensor.shape t)

(* Build a full 6D native permutation from an ATen [dims] list and the tensor
   rank.  For the [rank] used axes, [dims.(i)] is the ATen input dim for output
   position [i]; for the outer padding axes the permutation is the identity. *)
let native_perm_of_aten ~rank dims =
  let used = Aten_shape.used_axes ~rank in
  let outer = List.filter (fun a -> not (List.mem a used)) Axis.all in
  let outer_perm = List.map (fun a -> (a, a)) outer in
  let inner_perm =
    List.mapi
      (fun i d ->
        (Aten_shape.axis_of_dim ~rank i, Aten_shape.axis_of_dim ~rank d))
      dims
  in
  outer_perm @ inner_perm

(* An ATen dim list arg -> frame axes for the tensor of the given rank.  A
   missing/None dim (e.g. [mean.dim] with no [dim]) means "all axes". *)
let dims_arg node ~rank name =
  match D.find_arg node name with
  | Some (Argument.Ints xs) -> List.map (Aten_shape.axis_of_dim ~rank) xs
  | _ -> Aten_shape.used_axes ~rank

(* The innermost [k] axes of a rank-[rank] tensor — the normalised axes of an
   `rms_norm(normalized_shape)` (ATen normalises the trailing dims). *)
let trailing_axes ~rank ~k =
  let all = Aten_shape.used_axes ~rank in
  List.filteri (fun i _ -> i >= rank - k) all

(* `rms_norm`'s optional [eps]: when absent ATen uses finfo(dtype).eps; for
   float32 that is 2^-23. *)
let float32_eps = 1.1920929e-07

let eps_arg node name =
  match D.find_arg node name with
  | Some (Argument.Float f) -> f
  | _ -> float32_eps

(* Is an optional-Tensor arg actually present (vs absent / explicit None)? *)
let optional_tensor_present node name =
  match D.find_arg node name with
  | Some (Argument.Tensor _)
  | Some (Argument.Optional_tensor (OptionalTensorArgument.Tensor _)) ->
      true
  | _ -> false

(* `rms_norm`'s no-affine path: a weight of all ones, shaped to broadcast over
   the normalised axes (extent = x's extent there, 1 elsewhere). *)
let ones_weight (x_shape : Vec6.shape) dims =
  let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let wshape =
    List.fold_left (fun s a -> Vec6.set s a (Vec6.get x_shape a)) ones dims
  in
  Tensor.materialize wshape (fun _ -> 1.0)

(* --- Op dispatch --- *)

(* Dispatch a graph node to the native path.  The [aten_env] contains the
   pre-dispatch ATen inputs (SSA name -> tensor handle).  Returns:
   - None              : no native implementation; skip verification
   - Some (Error msg)  : mapped but failed (conversion error or unsupported args)
   - Some (Ok outputs) : native outputs as packed tensors, same length and order
                         as the ATen op's outputs *)
let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Tensor.packed list, string) result option =
  (* Arms in global alphabetical order by the dispatched op name. *)
  match node.target with
  | "torch.ops.aten.add.Tensor" | "torch.ops.aten.add_.Tensor" ->
      binary aten_env node ~f:run_add "self" "other"
  | "torch.ops.aten.bmm.default" ->
      binary aten_env node ~f:run_bmm "self" "mat2"
  | "torch.ops.aten.mean.dim" ->
      let t = D.tensor_arg aten_env node "self" in
      let rank = aten_rank t in
      let dims = dims_arg node ~rank "dim" in
      let keepdim = D.bool_arg node "keepdim" in
      unary aten_env node ~f:(run_mean ~dims ~keepdim) "self"
  | "torch.ops.aten.mul.Tensor" | "torch.ops.aten.mul_.Tensor" ->
      binary aten_env node ~f:run_mul "self" "other"
  | "torch.ops.aten.permute.default" ->
      let t = D.tensor_arg aten_env node "self" in
      let rank = aten_rank t in
      let dims = D.ints_arg node "dims" in
      let perm = native_perm_of_aten ~rank dims in
      unary aten_env node ~f:(fun x -> run_permute x perm) "self"
  | "torch.ops.aten.relu.default" | "torch.ops.aten.relu_.default" ->
      unary aten_env node ~f:run_relu "self"
  | "torch.ops.aten.rms_norm.default" -> (
      let t = D.tensor_arg aten_env node "input" in
      let rank = aten_rank t in
      let k = List.length (D.ints_arg node "normalized_shape") in
      let dims = trailing_axes ~rank ~k in
      let eps = eps_arg node "eps" in
      match Tensor_bridge.of_aten t with
      | Error e -> Some (Error (Printf.sprintf "input: %s" e))
      | Ok x -> (
          let weight =
            if optional_tensor_present node "weight" then
              native_tensor_arg aten_env node "weight"
            else
              let (Tensor.Tensor r) = x in
              Ok (ones_weight r.shape dims)
          in
          match weight with
          | Error e -> Some (Error (Printf.sprintf "weight: %s" e))
          | Ok weight -> (
              match run_rms_norm x ~weight ~dims ~eps with
              | exception Failure msg ->
                  Some (Error (Printf.sprintf "native op failed: %s" msg))
              | result -> Some (Ok [ result ]))))
  | _ -> None

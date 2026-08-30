(* Reduction family (amax/mean/softmax/vector_norm) op-dispatch arms for
   [Op_bridge], split from op_bridge.ml. See op_bridge.ml for the [dispatch]
   facade that folds every family module (including this one) into one
   lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.amax.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dims = dims_arg node ~op:"amax.default" ~rank "dim" in
         let* keepdim = bool_arg node "keepdim" in
         let* x = native_of_aten "self" t in
         let params = { Reduce.Amax.dims; keepdim } in
         build_g ~name:"amax" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = amax params x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.mean.dim" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dims = dims_arg node ~op:"mean.dim" ~rank "dim" in
         let* keepdim = bool_arg node "keepdim" in
         let* x = native_of_aten "self" t in
         let params = { Reduce.Mean.dims; keepdim } in
         build_g ~name:"mean" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = mean params x_id in
               [ y ]
           | _ -> assert false))
  (* ord=2 (the schema default) only: general ord needs its own `pow`, out of
     scope. `dtype` is decoded and rejected rather than silently dropped
     when present -- the same treatment `cudnn_enable` gets on
     `layer_norm.default`/`group_norm.default`. *)
  | "torch.ops.aten.linalg_vector_norm.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* ord = scalar_arg ~default:(Aten_scalar.Int 2L) node "ord" in
         let* () =
           if Float.equal ord 2.0 then return ()
           else
             fail
               (`Validation_failure
                  (Printf.sprintf
                     "linalg_vector_norm.default: only ord=2 is supported, got \
                      %g"
                     ord))
         in
         let* (_ : _ option) =
           decode_result (D.scalar_type_opt_arg_result node "dtype")
         in
         let* dims =
           dims_arg node ~op:"linalg_vector_norm.default" ~rank "dim"
         in
         let* keepdim = bool_arg node "keepdim" in
         let* x = native_of_aten "self" t in
         let params = { Reduce.Vector_norm.dims; keepdim } in
         build_g ~name:"vector_norm" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = vector_norm params x_id in
               [ y ]
           | _ -> assert false))
  (* [dim] is required by the schema (no default) -- unlike [dims_arg]'s
     dim-LIST convention for [amax]/[mean]/[vector_norm], which treats an
     absent/empty/None list as "all dims", softmax always names exactly one
     axis. `dtype` is decoded and rejected rather than silently dropped when
     present, the same treatment `linalg_vector_norm.default`'s gets above.
     See .ai/matmul_softmax_design.md §3. *)
  | "torch.ops.aten.softmax.int" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dim = int_arg node "dim" in
         let* axis = dim_axis ~op:"softmax.int" ~rank dim in
         let* (_ : _ option) =
           decode_result (D.scalar_type_opt_arg_result node "dtype")
         in
         let* x = native_of_aten "self" t in
         let params = { Reduce.Softmax.axis } in
         build_g ~name:"softmax" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = softmax params x_id in
               [ y ]
           | _ -> assert false))
  | _ -> None

(* Elementwise/activation op-dispatch arms for [Op_bridge], split from op_bridge.ml. See op_bridge.ml for the [dispatch] facade that
   folds every family module (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.add.Tensor" | "torch.ops.aten.add_.Tensor" ->
      Some
        (let* () = reject_alpha node in
         let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"add" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = add a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"add_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = add_scalar scalar a_id in
                   [ y ]
               | _ -> assert false))
  (* addcmul(self, tensor1, tensor2, value=1) = self + value * tensor1 *
     tensor2 -- decomposed to existing [Mul]/[Add]/[Mul_scalar] nodes rather
     than a new [Graph_ir] op: the ATen semantics literally IS that
     composition, with no shape/axis logic of its own to lose (unlike
     select/stack, which had their own shape rule a decomposition would have
     hidden). [Mul_scalar] is skipped entirely for
     the verified default [value=1], the only value this corpus serialises,
     rather than always emitting a `*1.` node. *)
  | "torch.ops.aten.addcmul.default" ->
      Some
        (let* self = native_tensor_arg aten_env node "self" in
         let* t1 = native_tensor_arg aten_env node "tensor1" in
         let* t2 = native_tensor_arg aten_env node "tensor2" in
         let* value = scalar_arg ~default:(Aten_scalar.Float 1.) node "value" in
         build_g ~name:"addcmul" [ self; t1; t2 ] (function
           | [ self_id; t1_id; t2_id ] ->
               let open Graph_builder in
               let* prod = mul t1_id t2_id in
               let* scaled =
                 if Float.equal value 1. then return prod
                 else mul_scalar value prod
               in
               let+ y = add self_id scaled in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.clamp.default" | "torch.ops.aten.clamp_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* min = scalar_opt_arg node "min" in
         let* max = scalar_opt_arg node "max" in
         (* The both-absent pair is rejected by [Graph_builder] via
            [Clamp.output_shape], the same place ATen's meta function rejects
            it, so no check is needed here. *)
         build_g ~name:"clamp" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = clamp { Pointwise.Clamp.min; max } x_id in
               [ y ]
           | _ -> assert false))
  (* clamp_min(self, min) = clamp(self, min=min, max=None): one bound of the
     same node [clamp.default] already builds, not a separate op -- reuses
     [Pointwise.Clamp] unchanged. *)
  | "torch.ops.aten.clamp_min.default" | "torch.ops.aten.clamp_min_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* s = decode_result (D.scalar_arg_result node "min") in
         let* min = float_of_aten_scalar "min" s in
         build_g ~name:"clamp_min" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y =
                 clamp { Pointwise.Clamp.min = Some min; max = None } x_id
               in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.div.Tensor" | "torch.ops.aten.div_.Tensor" ->
      Some
        (let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"div" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = div a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"div_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = div_scalar scalar a_id in
                   [ y ]
               | _ -> assert false))
  | "torch.ops.aten.gelu.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* approximate = string_arg ~default:"none" node "approximate" in
         let* approximate =
           match approximate with
           | "none" -> return Pointwise.Gelu.Exact
           | "tanh" -> return Pointwise.Gelu.Tanh
           | _ ->
               fail
                 (`Validation_failure
                    (Printf.sprintf
                       "gelu approximate=%s is not supported (only \"none\" or \
                        \"tanh\")"
                       approximate))
         in
         build_g ~name:"gelu" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = gelu approximate x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.hardsigmoid.default" | "torch.ops.aten.hardsigmoid_.default"
    ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"hardsigmoid" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = hardsigmoid x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.hardswish.default" | "torch.ops.aten.hardswish_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"hardswish" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = hardswish x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.hardtanh.default" | "torch.ops.aten.hardtanh_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         (* Schema `Scalar min_val=-1, Scalar max_val=1`: both bounds may arrive
            as Int or Float, and either may be absent. *)
         let* min_val =
           scalar_arg ~default:(Aten_scalar.Float (-1.)) node "min_val"
         in
         let* max_val =
           scalar_arg ~default:(Aten_scalar.Float 1.) node "max_val"
         in
         build_g ~name:"hardtanh" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = hardtanh { Pointwise.Hardtanh.min_val; max_val } x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.leaky_relu.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* negative_slope =
           scalar_arg ~default:(Aten_scalar.Float 0.01) node "negative_slope"
         in
         build_g ~name:"leaky_relu" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y =
                 leaky_relu { Pointwise.Leaky_relu.negative_slope } x_id
               in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.mul.Tensor" | "torch.ops.aten.mul_.Tensor" ->
      Some
        (let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"mul" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = mul a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"mul_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = mul_scalar scalar a_id in
                   [ y ]
               | _ -> assert false))
  | "torch.ops.aten.mul.Scalar" ->
      Some
        (let* a = native_tensor_arg aten_env node "self" in
         let* s = decode_result (D.scalar_arg_result node "other") in
         let* scalar = float_of_aten_scalar "other" s in
         build_g ~name:"mul_scalar" [ a ] (function
           | [ a_id ] ->
               let open Graph_builder in
               let+ y = mul_scalar scalar a_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.pow.Tensor_Scalar" ->
      Some
        (let* a = native_tensor_arg aten_env node "self" in
         let* s = decode_result (D.scalar_arg_result node "exponent") in
         let* exponent = float_of_aten_scalar "exponent" s in
         build_g ~name:"pow" [ a ] (function
           | [ a_id ] ->
               let open Graph_builder in
               let+ y = pow exponent a_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.relu.default" | "torch.ops.aten.relu_.default" -> (
      match native_tensor_arg aten_env node "self" with
      | Error e -> Some (Error e)
      | Ok x ->
          build_g ~name:"relu" [ x ] (function
            | [ x_id ] ->
                let open Graph_builder in
                let+ y = relu x_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  (* rsqrt(x) = x ** -0.5: [Pow]'s own [Compute] already special-cases exponent
     -0.5 as [reciprocal (sqrt x)], the same reciprocal-of-sqrt kernel ATen's
     [PowKernel.cpp] uses for both `pow(x,-0.5)` and `rsqrt` -- so this legalizes
     onto the existing [Pow] node rather than adding a new one, the same "map
     onto an existing op" pattern [sub.Tensor]'s scalar form and the original
     [pow(x,0.5) -> Sqrt] legalization used. *)
  | "torch.ops.aten.rsqrt.default" | "torch.ops.aten.rsqrt_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"rsqrt" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = pow (-0.5) x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.sigmoid.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"sigmoid" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = sigmoid x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.silu.default" | "torch.ops.aten.silu_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"silu" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = silu x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.sqrt.default" | "torch.ops.aten.sqrt_.default" -> (
      match native_tensor_arg aten_env node "self" with
      | Error e -> Some (Error e)
      | Ok x ->
          build_g ~name:"sqrt" [ x ] (function
            | [ x_id ] ->
                let open Graph_builder in
                let+ y = sqrt x_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  (* [x - s] legalizes to [x + (-s)]: IEEE negation is exact and the builder
     narrows to f32 on both spellings either way, so the two are bit-identical
     (op3-impl.md F7). No [sub_scalar] builder exists and none should: this
     negation is the whole legalization. *)
  | "torch.ops.aten.sub.Tensor" | "torch.ops.aten.sub_.Tensor" ->
      Some
        (let* () = reject_alpha node in
         let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"sub" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = sub a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"sub_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = add_scalar (-.scalar) a_id in
                   [ y ]
               | _ -> assert false))
  | _ -> None

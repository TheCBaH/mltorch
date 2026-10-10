(* Comparison, boolean, selection and constant-factory operator dispatch for
   [Native_interp_lower]: the vocabulary attention masks are built from.

   Comparisons produce a bool, which Native keeps as a [Bool] edge holding
   0./1. A comparison against a Scalar reads the scalar as written ([Int] or
   [Float]); the operand may be float, bool or int64, read through the float
   domain (see [Pointwise_logic] for the 2^53 caveat).

   [new_ones.default]'s [self] only supplies defaults for options left unset,
   and the dtype is required explicitly -- the edge has no per-tensor dtype tag
   to default from -- as bool or float32. *)

open Pytorch_types
open Native_interp_decode
open Native_interp_decode_shape

let targets =
  [
    "torch.ops.aten.__and__.Tensor";
    "torch.ops.aten._weight_norm.default";
    "torch.ops.aten.eq.Scalar";
    "torch.ops.aten.eq.Tensor";
    "torch.ops.aten.exp.default";
    "torch.ops.aten.full_like.default";
    "torch.ops.aten.ge.Scalar";
    "torch.ops.aten.gt.Scalar";
    "torch.ops.aten.le.Tensor";
    "torch.ops.aten.log.default";
    "torch.ops.aten.log1p.default";
    "torch.ops.aten.lt.Scalar";
    "torch.ops.aten.min.other";
    "torch.ops.aten.ne.Scalar";
    "torch.ops.aten.ne.Tensor";
    "torch.ops.aten.new_ones.default";
    "torch.ops.aten.t.default";
    "torch.ops.aten.tanh.default";
    "torch.ops.aten.where.ScalarOther";
    "torch.ops.aten.where.self";
    "torch.ops.aten.zeros_like.default";
  ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let graph = ctx.Native_interp_lower_context.graph in
       let get = Native_interp_lower_context.get ctx env node in
       let scalar () = required_scalar_arg esc node "other" in
       match node.target with
       | "torch.ops.aten.__and__.Tensor" ->
           let* y = bitwise_and (get "self") (get "other") in
           return [ y ]
       (* `_weight_norm(Tensor v, Tensor g, int dim=0)`: the weight a
          `weight_norm` parametrization recomputes. [dim] names the one
          dimension the norm keeps. *)
       | "torch.ops.aten._weight_norm.default" ->
           let v_name = tensor_name esc node "v" in
           let rank =
             meta_rank
               (tensor_meta esc graph ~ssa:v_name ~role:`Weight_norm_input)
           in
           let d = normalize_dim esc ~rank (dim_arg esc node "dim") in
           let axis = List.nth (used_axes_for esc ~tensor:v_name rank) d in
           let* y =
             weight_norm
               { Weight_norm.Weight_norm.axis }
               ~v:(get "v") ~g:(get "g")
           in
           return [ y ]
       | "torch.ops.aten.eq.Scalar" ->
           let* y = eq_scalar (scalar ()) (get "self") in
           return [ y ]
       | "torch.ops.aten.eq.Tensor" ->
           let* y = eq_tensor (get "self") (get "other") in
           return [ y ]
       | "torch.ops.aten.exp.default" ->
           let* y = exp (get "self") in
           return [ y ]
       (* `full_like(Tensor self, Scalar fill_value, *, ...)` and
          `zeros_like(Tensor self, *, ...)`: a constant in [self]'s shape and
          format. An explicit dtype, layout or non-cpu device would change what
          is made, so any of them is refused; [zeros_like] is the zero fill. *)
       | "torch.ops.aten.full_like.default"
       | "torch.ops.aten.zeros_like.default" ->
           let optional name =
             List.find_opt
               (fun (a : NamedArgument.t) -> a.name = name)
               node.Node.inputs
             |> Option.map (fun a -> a.NamedArgument.arg)
           in
           List.iter
             (fun name ->
               match optional name with
               | None | Some (Argument.None _) -> ()
               | Some _ ->
                   malformed esc
                     (`Unsupported_option { op = node.target; option = `Dtype }))
             [ "dtype"; "layout" ];
           (match optional "device" with
           | None | Some (Argument.None _) -> ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) -> ()
           | Some _ ->
               malformed esc
                 (`Wrong_arg_kind
                    { op = node.target; arg = "device"; expected = `Tensor }));
           (match optional "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) -> ()
           | Some _ ->
               malformed esc
                 (`Wrong_arg_kind
                    { op = node.target; arg = "pin_memory"; expected = `Bool }));
           let value =
             if String.equal node.target "torch.ops.aten.zeros_like.default"
             then 0.
             else required_scalar_arg esc node "fill_value"
           in
           let* y = full_like value (get "self") in
           return [ y ]
       | "torch.ops.aten.ge.Scalar" ->
           let* y = ge_scalar (scalar ()) (get "self") in
           return [ y ]
       | "torch.ops.aten.gt.Scalar" ->
           let* y = gt_scalar (scalar ()) (get "self") in
           return [ y ]
       | "torch.ops.aten.le.Tensor" ->
           let* y = le_tensor (get "self") (get "other") in
           return [ y ]
       | "torch.ops.aten.log.default" ->
           let* y = log (get "self") in
           return [ y ]
       | "torch.ops.aten.log1p.default" ->
           let* y = log1p (get "self") in
           return [ y ]
       | "torch.ops.aten.min.other" ->
           let* y = min_other (get "self") (get "other") in
           return [ y ]
       | "torch.ops.aten.lt.Scalar" ->
           let* y = lt_scalar (scalar ()) (get "self") in
           return [ y ]
       | "torch.ops.aten.ne.Scalar" ->
           let* y = ne_scalar (scalar ()) (get "self") in
           return [ y ]
       | "torch.ops.aten.ne.Tensor" ->
           let* y = ne_tensor (get "self") (get "other") in
           return [ y ]
       | "torch.ops.aten.new_ones.default" ->
           let optional name =
             List.find_opt
               (fun (a : NamedArgument.t) -> a.name = name)
               node.Node.inputs
             |> Option.map (fun a -> a.NamedArgument.arg)
           in
           let fmt =
             match optional "dtype" with
             | Some (Argument.Scalar_type Pytorch_types.ScalarType.BOOL) ->
                 Payload.Fmt Payload.Bool
             | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
                 Payload.Fmt Payload.F32
             | _ ->
                 malformed esc
                   (`Unsupported_option { op = node.target; option = `Dtype })
           in
           (match optional "layout" with
           | None | Some (Argument.None _) -> ()
           | Some _ ->
               malformed esc
                 (`Wrong_arg_kind
                    { op = node.target; arg = "layout"; expected = `Tensor }));
           (match optional "device" with
           | None | Some (Argument.None _) -> ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) -> ()
           | Some _ ->
               malformed esc
                 (`Wrong_arg_kind
                    { op = node.target; arg = "device"; expected = `Tensor }));
           (match optional "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) -> ()
           | Some _ ->
               malformed esc
                 (`Wrong_arg_kind
                    { op = node.target; arg = "pin_memory"; expected = `Bool }));
           let (_ : Graph_ir.Tensor_id.t) = get "self" in
           let shape =
             shape_of_sizes esc "new_ones.size"
               (List.map (fun i -> SymInt.Int i) (ints_arg esc node "size"))
           in
           let* y = new_ones { Factory.New_ones.shape; fmt } in
           return [ y ]
       (* `t(Tensor self)`: a matrix transpose, and the identity below rank 2.
          It is `transpose.int(0, 1)` on a rank-2 tensor, so it lowers to the
          same one permute node. *)
       | "torch.ops.aten.t.default" ->
           let x_name = tensor_name esc node "self" in
           let rank =
             meta_rank
               (tensor_meta esc graph ~ssa:x_name ~role:`Transpose_input)
           in
           let dims =
             List.init
               (rank :> int)
               (fun i ->
                 Aten_int.Dim.of_int (if (rank :> int) = 2 then 1 - i else i))
           in
           let* y =
             permute (native_perm esc ~tensor:x_name ~rank dims) (get "self")
           in
           return [ y ]
       | "torch.ops.aten.tanh.default" ->
           let* y = tanh (get "self") in
           return [ y ]
       | "torch.ops.aten.where.self" ->
           let* y =
             where_self ~condition:(get "condition") (get "self") (get "other")
           in
           return [ y ]
       | "torch.ops.aten.where.ScalarOther" ->
           let* y =
             where_scalar_other ~condition:(get "condition") (scalar ())
               (get "self")
           in
           return [ y ]
       | _ -> assert false)

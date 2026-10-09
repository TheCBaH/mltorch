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

let targets =
  [
    "torch.ops.aten.__and__.Tensor";
    "torch.ops.aten.eq.Scalar";
    "torch.ops.aten.eq.Tensor";
    "torch.ops.aten.ge.Scalar";
    "torch.ops.aten.gt.Scalar";
    "torch.ops.aten.le.Tensor";
    "torch.ops.aten.lt.Scalar";
    "torch.ops.aten.ne.Scalar";
    "torch.ops.aten.ne.Tensor";
    "torch.ops.aten.new_ones.default";
    "torch.ops.aten.tanh.default";
    "torch.ops.aten.where.ScalarOther";
  ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       let get = Native_interp_lower_context.get ctx env node in
       let scalar () = required_scalar_arg esc node "other" in
       match node.target with
       | "torch.ops.aten.__and__.Tensor" ->
           let* y = bitwise_and (get "self") (get "other") in
           return [ y ]
       | "torch.ops.aten.eq.Scalar" ->
           let* y = eq_scalar (scalar ()) (get "self") in
           return [ y ]
       | "torch.ops.aten.eq.Tensor" ->
           let* y = eq_tensor (get "self") (get "other") in
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
       | "torch.ops.aten.tanh.default" ->
           let* y = tanh (get "self") in
           return [ y ]
       | "torch.ops.aten.where.ScalarOther" ->
           let* y =
             where_scalar_other ~condition:(get "condition") (scalar ())
               (get "self")
           in
           return [ y ]
       | _ -> assert false)

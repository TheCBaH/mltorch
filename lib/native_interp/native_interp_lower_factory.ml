(* Factory-family operator dispatch for [Native_interp_lower] (ranges, zeros,
   identity matrices and meshgrids): ops that build a tensor from scalar
   arguments rather than transform an input. Like every family it returns
   [None] for targets it does not own; the lowerer owns the one
   unsupported-operator fallback. Split out of the shape family under the
   tracked file-size ceiling (scripts/check-file-size.sh). *)

open Pytorch_types
open Native_interp_decode

let targets =
  [
    "torch.ops.aten.arange.default";
    "torch.ops.aten.arange.start";
    "torch.ops.aten.arange.start_step";
    "torch.ops.aten.eye.m";
    "torch.ops.aten.meshgrid.indexing";
    "torch.ops.aten.zeros.default";
  ]

let dispatch ~ctx ~env (node : Node.t) =
  if not (List.mem node.target targets) then None
  else
    Some
      (let open Graph_builder in
       let esc = ctx.Native_interp_lower_context.esc in
       match node.target with
       | ( "torch.ops.aten.arange.default" | "torch.ops.aten.arange.start"
         | "torch.ops.aten.arange.start_step" ) as target ->
           let optional name =
             List.find_opt
               (fun (a : NamedArgument.t) -> a.name = name)
               node.Node.inputs
             |> Option.map (fun a -> a.NamedArgument.arg)
           in
           let scalar_input_is_float =
             List.exists
               (fun name ->
                 match optional name with
                 | Some (Argument.Float _) -> true
                 | _ -> false)
               [ "start"; "end"; "step" ]
           in
           let fmt =
             match optional "dtype" with
             | None | Some (Argument.None _) ->
                 if scalar_input_is_float then Payload.Fmt Payload.F32
                 else Payload.Fmt Payload.I64
             | Some (Argument.Scalar_type Pytorch_types.ScalarType.LONG) ->
                 Payload.Fmt Payload.I64
             | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
                 Payload.Fmt Payload.F32
             | Some _ ->
                 malformed esc
                   (`Unsupported_option { op = node.target; option = `Dtype })
           in
           List.iter
             (fun name ->
               match optional name with
               | None | Some (Argument.None _) -> ()
               | Some _ ->
                   malformed esc
                     (`Wrong_arg_kind
                        { op = node.target; arg = name; expected = `Tensor }))
             [ "layout" ];
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
           let start, stop =
             match target with
             | "torch.ops.aten.arange.default" ->
                 (0., required_scalar_arg esc node "end")
             | "torch.ops.aten.arange.start"
             | "torch.ops.aten.arange.start_step" ->
                 ( required_scalar_arg esc node "start",
                   required_scalar_arg esc node "end" )
             | _ -> assert false
           in
           (* [start_step] is the only overload whose schema makes [step]
              required; the other two default it to one. *)
           let step =
             match target with
             | "torch.ops.aten.arange.start_step" ->
                 required_scalar_arg esc node "step"
             | _ -> scalar_arg esc ~default:1. node "step"
           in
           (* An exact int64 view exists only when every bound the graph
              spelled is an [Argument.Int] that is not one of the decoder's
              saturation values: [Schema_runtime.python_int_jsont] clamps
              anything past the host [int] to [min_int]/[max_int] (2^31 under
              js_of_ocaml, see CLAUDE.md's JS-reachable-library rule), so
              those two values may stand for a larger integer and are not
              recoverable. Every other [Argument.Int] is the literal itself,
              so [Int64.of_int] is lossless. A float bound, or a saturated
              one, keeps the legacy float path ([exact = None]). *)
           let exact_int ~default name =
             match optional name with
             | None | Some (Argument.None _) -> Some default
             | Some (Argument.Int i) when i > min_int && i < max_int ->
                 Some (Int64.of_int i)
             | Some _ -> None
           in
           let exact =
             match fmt with
             | Payload.Fmt Payload.I64 -> (
                 let start_exact, stop_exact =
                   match target with
                   | "torch.ops.aten.arange.default" ->
                       (Some 0L, exact_int ~default:0L "end")
                   | _ ->
                       ( exact_int ~default:0L "start",
                         exact_int ~default:0L "end" )
                 in
                 match
                   (start_exact, stop_exact, exact_int ~default:1L "step")
                 with
                 | Some start, Some stop, Some step ->
                     Some { Factory.Arange.Exact.start; stop; step }
                 | _ -> None)
             | _ -> None
           in
           let* y = arange { Factory.Arange.start; stop; step; fmt; exact } in
           return [ y ]
       | "torch.ops.aten.zeros.default" ->
           let optional name =
             List.find_opt
               (fun (a : NamedArgument.t) -> a.name = name)
               node.Node.inputs
             |> Option.map (fun a -> a.NamedArgument.arg)
           in
           let fmt =
             match optional "dtype" with
             | None | Some (Argument.None _) -> Payload.Fmt Payload.F32
             | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
                 Payload.Fmt Payload.F32
             | Some (Argument.Scalar_type Pytorch_types.ScalarType.DOUBLE) ->
                 Payload.Fmt Payload.F64
             | Some _ ->
                 malformed esc
                   (`Unsupported_option { op = node.target; option = `Dtype })
           in
           List.iter
             (fun name ->
               match optional name with
               | None | Some (Argument.None _) -> ()
               | Some _ ->
                   malformed esc
                     (`Wrong_arg_kind
                        { op = node.target; arg = name; expected = `Tensor }))
             [ "layout" ];
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
           let shape =
             shape_of_sizes esc "zeros.size"
               (List.map (fun i -> SymInt.Int i) (ints_arg esc node "size"))
           in
           let* y = zeros { Factory.Zeros.shape; fmt } in
           return [ y ]
       (* [eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout?
         layout=None, Device? device=None, bool? pin_memory=None) -> Tensor],
         the same trailing-argument rejection/dtype dispatch [zeros.default]'s
         own arm above uses -- [n]/[m] are two scalar [SymInt]s rather than a
         [SymInt[]] list, so they are read with [int_arg] and packed into
         [shape_of_sizes]'s own list form. *)
       | "torch.ops.aten.eye.m" ->
           let optional name =
             List.find_opt
               (fun (a : NamedArgument.t) -> a.name = name)
               node.Node.inputs
             |> Option.map (fun a -> a.NamedArgument.arg)
           in
           let fmt =
             match optional "dtype" with
             | None | Some (Argument.None _) -> Payload.Fmt Payload.F32
             | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
                 Payload.Fmt Payload.F32
             | Some (Argument.Scalar_type Pytorch_types.ScalarType.DOUBLE) ->
                 Payload.Fmt Payload.F64
             | Some _ ->
                 malformed esc
                   (`Unsupported_option { op = node.target; option = `Dtype })
           in
           List.iter
             (fun name ->
               match optional name with
               | None | Some (Argument.None _) -> ()
               | Some _ ->
                   malformed esc
                     (`Wrong_arg_kind
                        { op = node.target; arg = name; expected = `Tensor }))
             [ "layout" ];
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
           let n = int_arg esc node "n" in
           let m = int_arg esc node "m" in
           let shape =
             shape_of_sizes esc "eye.m" [ SymInt.Int n; SymInt.Int m ]
           in
           let* y = eye { Factory.Eye.shape; fmt } in
           return [ y ]
       (* [meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]],
         restricted to the corpus's own shape -- see [Meshgrid.Meshgrid]'s
         own comment: every input rank-1, [indexing="ij"] only. Rank itself
         is [Graph_shape]'s own check, not restated here. *)
       | "torch.ops.aten.meshgrid.indexing" ->
           let indexing = string_arg esc ~default:"ij" node "indexing" in
           if not (String.equal indexing "ij") then
             malformed esc
               (`Unsupported_option
                  { op = node.target; option = `Indexing indexing });
           let names = tensor_names_arg esc node "tensors" in
           meshgrid (List.map (env_find esc env) names)
       | _ -> assert false)

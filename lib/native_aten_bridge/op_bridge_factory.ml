(* Factory family (zeros/eye/arange: no-operand ops whose whole output is
   model-supplied metadata) op-dispatch arms for [Op_bridge], split out of
   op_bridge_shape.ml once that file crossed the tracked 1000-line ceiling
   (scripts/check-file-size.sh). See op_bridge.ml for the [dispatch] facade
   that folds every family module (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  ignore aten_env;
  match node.target with
  (* [zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None,
     Device? device=None, bool? pin_memory=None) -> Tensor].  This is a real
     factory, not a fill disguised as an input-free pointwise op: shape and
     dtype are retained in the Native payload. *)
  | "torch.ops.aten.zeros.default" ->
      Some
        (let* size = ints_arg node "size" in
         let* shape =
           Aten_shape.of_aten (Array.of_list size)
           |> Err.map_error (fun e -> `Aten_shape e)
         in
         let* fmt =
           match D.find_arg node "dtype" with
           | None | Some (Argument.None _) -> return (Payload.Fmt Payload.F32)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
               return (Payload.Fmt Payload.F32)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.DOUBLE) ->
               return (Payload.Fmt Payload.F64)
           | Some _ ->
               fail
                 (`Validation_failure
                    "zeros: only default/FLOAT and DOUBLE dtype are supported")
         in
         let reject_absent name =
           match D.find_arg node name with
           | None | Some (Argument.None _) -> return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    ("zeros: " ^ name ^ " must be omitted or None"))
         in
         let* () = reject_absent "layout" in
         let* () =
           match D.find_arg node "device" with
           | None | Some (Argument.None _) -> return ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "zeros: device must be omitted/None or CPU without an index")
         in
         let* () =
           match D.find_arg node "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "zeros: pin_memory must be omitted/None or false")
         in
         let* g =
           Graph_builder.build ~name:"zeros"
             ~outputs:(fun y -> [ y ])
             (Graph_builder.zeros { Factory.Zeros.shape; fmt })
           |> Err.map_error (fun e -> `Build e)
         in
         return (g, []))
  (* [eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None,
     Device? device=None, bool? pin_memory=None) -> Tensor], the rank-2
     identity-matrix factory -- the same [layout]/[device]/[pin_memory]
     rejection and FLOAT/DOUBLE dtype dispatch [zeros.default]'s own arm
     uses, since the trailing argument list is identical. *)
  | "torch.ops.aten.eye.m" ->
      Some
        (let* n = int_arg node "n" in
         let* m = int_arg node "m" in
         let* shape =
           Aten_shape.of_aten [| n; m |]
           |> Err.map_error (fun e -> `Aten_shape e)
         in
         let* fmt =
           match D.find_arg node "dtype" with
           | None | Some (Argument.None _) -> return (Payload.Fmt Payload.F32)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
               return (Payload.Fmt Payload.F32)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.DOUBLE) ->
               return (Payload.Fmt Payload.F64)
           | Some _ ->
               fail
                 (`Validation_failure
                    "eye.m: only default/FLOAT and DOUBLE dtype are supported")
         in
         let reject_absent name =
           match D.find_arg node name with
           | None | Some (Argument.None _) -> return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    ("eye.m: " ^ name ^ " must be omitted or None"))
         in
         let* () = reject_absent "layout" in
         let* () =
           match D.find_arg node "device" with
           | None | Some (Argument.None _) -> return ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "eye.m: device must be omitted/None or CPU without an index")
         in
         let* () =
           match D.find_arg node "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "eye.m: pin_memory must be omitted/None or false")
         in
         let* g =
           Graph_builder.build ~name:"eye"
             ~outputs:(fun y -> [ y ])
             (Graph_builder.eye { Factory.Eye.shape; fmt })
           |> Err.map_error (fun e -> `Build e)
         in
         return (g, []))
  | ("torch.ops.aten.arange.default" | "torch.ops.aten.arange.start") as target
    ->
      Some
        (let scalar_input_is_float =
           List.exists
             (fun name ->
               match D.find_arg node name with
               | Some (Argument.Float _) -> true
               | _ -> false)
             [ "start"; "end"; "step" ]
         in
         let* fmt =
           match D.find_arg node "dtype" with
           | None | Some (Argument.None _) ->
               return
                 (if scalar_input_is_float then Payload.Fmt Payload.F32
                  else Payload.Fmt Payload.I64)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.LONG) ->
               return (Payload.Fmt Payload.I64)
           | Some (Argument.Scalar_type Pytorch_types.ScalarType.FLOAT) ->
               return (Payload.Fmt Payload.F32)
           | Some _ ->
               fail
                 (`Validation_failure
                    "arange: only default/LONG and FLOAT dtype are supported")
         in
         let reject_absent name =
           match D.find_arg node name with
           | None | Some (Argument.None _) -> return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    ("arange: " ^ name ^ " must be omitted or None"))
         in
         let* () = reject_absent "layout" in
         let* () =
           match D.find_arg node "device" with
           | None | Some (Argument.None _) -> return ()
           | Some (Argument.Device { Device.type_ = "cpu"; index = None }) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "arange: device must be omitted/None or CPU without an \
                     index")
         in
         let* () =
           match D.find_arg node "pin_memory" with
           | None | Some (Argument.None _) | Some (Argument.Bool false) ->
               return ()
           | Some _ ->
               fail
                 (`Validation_failure
                    "arange: pin_memory must be omitted/None or false")
         in
         let* start, stop =
           match target with
           | "torch.ops.aten.arange.default" ->
               let* stop =
                 scalar_arg ~default:(Aten_scalar.Int 0L) node "end"
               in
               return (0., stop)
           | "torch.ops.aten.arange.start" ->
               let* start =
                 scalar_arg ~default:(Aten_scalar.Int 0L) node "start"
               in
               let* stop =
                 scalar_arg ~default:(Aten_scalar.Int 0L) node "end"
               in
               return (start, stop)
           | _ -> assert false
         in
         let* step = scalar_arg ~default:(Aten_scalar.Int 1L) node "step" in
         let* g =
           Graph_builder.build ~name:"arange"
             ~outputs:(fun y -> [ y ])
             (Graph_builder.arange { Factory.Arange.start; stop; step; fmt })
           |> Err.map_error (fun e -> `Build e)
         in
         return (g, []))
  | _ -> None

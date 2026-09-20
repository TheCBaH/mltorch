(* See output_spec.mli. *)

let storable (sg : Tensor_sig.t) =
  (match sg.Tensor_sig.fmt with
    | Payload.Fmt (Payload.F32 | Payload.Bool) -> true
    | _ -> false)
  && Option.is_none sg.Tensor_sig.quant

let store (sg : Tensor_sig.t) tensor =
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.Bool -> Tensor.bool_of_float_cells tensor
  | _ -> tensor

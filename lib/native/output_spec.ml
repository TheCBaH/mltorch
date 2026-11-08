(* See output_spec.mli. *)

let storable (sg : Tensor_sig.t) =
  (match sg.Tensor_sig.fmt with
    | Payload.Fmt (Payload.BF16 | Payload.Bool | Payload.F32) -> true
    | _ -> false)
  && Option.is_none sg.Tensor_sig.quant

let store (sg : Tensor_sig.t) tensor =
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.BF16 -> Tensor.bf16_of_float_cells tensor
  | Payload.Fmt Payload.Bool -> Tensor.bool_of_float_cells tensor
  | _ -> tensor

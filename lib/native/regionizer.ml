(* Provenance is available only while graph construction still has the typed
   [Graph_ir.op].  This transitional dispatcher resolves named operands and
   validates the output contract before calling an operation-owned Region
   constructor.  It owns neither operation arithmetic nor Region execution. *)

type error =
  | Invalid_partition
  | Invalid_program
  | Missing_operand of Tensor_id.t
  | Output_ordinal of int
  | Output_shape

type synthetic_role = Rms_weight | Layer_weight | Layer_bias

let pp_error fmt = function
  | Invalid_partition -> Region_context.pp_error fmt Invalid_partition
  | Invalid_program -> Region_context.pp_error fmt Invalid_program
  | Missing_operand id -> Fmt.pf fmt "missing operand %a" Tensor_id.pp id
  | Output_ordinal output -> Fmt.pf fmt "unsupported output ordinal %d" output
  | Output_shape -> Fmt.string fmt "output shape does not match the input"

let required ~operand id = operand id |> Err.of_option (Missing_operand id)

let check_output ~output ~output_shape x =
  if output <> 0 then Err.fail (Output_ordinal output)
  else if not (Region_context.same_shape output_shape x.Tensor_sig.shape) then
    Err.fail Output_shape
  else Err.return ()

let is_region_authored = function
  | Graph_ir.Rms_norm _ | Graph_ir.Layer_norm _ | Graph_ir.Softmax _ -> true
  | _ -> false

let program ~limits ~op ~output ~output_shape ~operand ~fill =
  let open Graph_ir in
  match op with
  | Rms_norm { Norm.RmsNorm.params; x = x_id; weight } ->
      let open Err.Syntax in
      let* x = required ~operand x_id in
      let* () = check_output ~output ~output_shape x in
      let weight =
        match weight with
        | Some id -> required ~operand id
        | None ->
            Err.return
              (fill Rms_weight 1.
                 (Norm_shared.normalized_shape ~x_shape:x.Tensor_sig.shape
                    ~dims:params.dims))
      in
      let* weight = weight in
      Err.map_error
        (function
          | Region_context.Invalid_partition -> Invalid_partition
          | Region_context.Invalid_program -> Invalid_program)
        (Norm.RmsNorm.Computation.program ~limits params ~x ~weight)
  | Layer_norm { Norm.LayerNorm.params; x = x_id; weight; bias } ->
      let open Err.Syntax in
      let* x = required ~operand x_id in
      let* () = check_output ~output ~output_shape x in
      let shape =
        Norm_shared.normalized_shape ~x_shape:x.Tensor_sig.shape
          ~dims:params.dims
      in
      let resolve_optional value role default =
        match value with
        | Some id -> required ~operand id
        | None -> Err.return (fill role default shape)
      in
      let* weight = resolve_optional weight Layer_weight 1. in
      let* bias = resolve_optional bias Layer_bias 0. in
      Err.map_error
        (function
          | Region_context.Invalid_partition -> Invalid_partition
          | Region_context.Invalid_program -> Invalid_program)
        (Norm.LayerNorm.Computation.program ~limits params ~x ~weight ~bias)
  | Softmax { Reduce.Softmax.params; x = x_id } ->
      let open Err.Syntax in
      let* x = required ~operand x_id in
      let* () = check_output ~output ~output_shape x in
      Err.map_error
        (function
          | Region_context.Invalid_partition -> Invalid_partition
          | Region_context.Invalid_program -> Invalid_program)
        (Reduce.Softmax.Computation.program ~limits params ~x)
  | _ -> assert false

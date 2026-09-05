(* The graph-boundary dispatcher for operation-authored Region computation.
   It resolves typed operands and verifies the output contract before invoking
   the operation-owned builder; it owns no operation arithmetic or fallback. *)

type error =
  | Invalid_partition
  | Invalid_program of Region_program.error
  | Missing_operand of Tensor_id.t
  | Output_ordinal of int
  | Output_shape

type synthetic_role = Layer_bias | Layer_weight | Rms_weight | Sdpa_mask

let pp_error fmt = function
  | Invalid_partition -> Region_context.pp_error fmt Invalid_partition
  | Invalid_program error ->
      Region_context.pp_error fmt (Region_context.Invalid_program error)
  | Missing_operand id -> Fmt.pf fmt "missing operand %a" Tensor_id.pp id
  | Output_ordinal output -> Fmt.pf fmt "unsupported output ordinal %d" output
  | Output_shape -> Fmt.string fmt "output shape does not match the input"

let required ~operand id = operand id |> Err.of_option (Missing_operand id)

let check_output ~output ~output_shape ~(expected : Vec6.shape) =
  if output <> 0 then Err.fail (Output_ordinal output)
  else if not (Region_context.same_shape output_shape expected) then
    Err.fail Output_shape
  else Err.return ()

let is_region_authored = function
  | Graph_ir.Layer_norm _ | Graph_ir.Rms_norm _ | Graph_ir.Sdpa _
  | Graph_ir.Softmax _ ->
      true
  | _ -> false

let program ~limits ~op ~output ~output_shape ~operand ~fill =
  let open Graph_ir in
  match op with
  | Rms_norm { Norm.RmsNorm.params; x = x_id; weight } ->
      let open Err.Syntax in
      let* x = required ~operand x_id in
      let* () =
        check_output ~output ~output_shape ~expected:x.Tensor_sig.shape
      in
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
          | Region_context.Invalid_program error -> Invalid_program error)
        (Norm.RmsNorm.Computation.program ~limits params ~x ~weight)
  | Layer_norm { Norm.LayerNorm.params; x = x_id; weight; bias } ->
      let open Err.Syntax in
      let* x = required ~operand x_id in
      let* () =
        check_output ~output ~output_shape ~expected:x.Tensor_sig.shape
      in
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
          | Region_context.Invalid_program error -> Invalid_program error)
        (Norm.LayerNorm.Computation.program ~limits params ~x ~weight ~bias)
  | Sdpa
      {
        Attention.Sdpa.params;
        query = query_id;
        key = key_id;
        value = value_id;
        mask;
      } ->
      let open Err.Syntax in
      let* query = required ~operand query_id in
      let* key = required ~operand key_id in
      let* value = required ~operand value_id in
      (* Unlike [Rms_norm]/[Layer_norm]/[Softmax], [output_shape] does not
         simply equal one fixed operand's own shape any more: once
         query/key/value may broadcast against each other on [N]/[T]/[D]/[H]
         (`.ai/attention_design.md`'s head-broadcasting note), the true
         output can be LARGER than [query]'s own shape on those axes. The
         sanity check recomputes that broadcast via
         [Attention.Sdpa.batch_shape] (the same fold [output_shape] itself
         uses) rather than assuming [query]'s shape is the answer. *)
      let* full_batch =
        Err.map_error
          (function `Sdpa _ -> Output_shape)
          (let open Err.Syntax in
           let* _qk_batch, full_batch =
             Attention.Sdpa.batch_shape ~query_shape:query.Tensor_sig.shape
               ~key_shape:key.Tensor_sig.shape
               ~value_shape:value.Tensor_sig.shape
           in
           Err.return full_batch)
      in
      let expected =
        Vec6.set full_batch Axis.C (Vec6.get query.Tensor_sig.shape Axis.C)
      in
      let* () = check_output ~output ~output_shape ~expected in
      (* Absent mask fills a single all-ones-shaped element (numel = 1, not
         the score shape), the additive identity 0.0 -- [fill] is the only
         site that picks a synthetic operand's default value and shape, same
         as [Rms_weight]/[Layer_weight]/[Layer_bias] above. *)
      let* mask =
        match mask with
        | Some id -> required ~operand id
        | None ->
            let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
            Err.return (fill Sdpa_mask 0. ones)
      in
      Err.map_error
        (function
          | Region_context.Invalid_partition -> Invalid_partition
          | Region_context.Invalid_program error -> Invalid_program error)
        (Attention.Sdpa.Computation.program ~limits params ~query ~key ~value
           ~mask)
  | Softmax { Reduce.Softmax.params; x = x_id } ->
      let open Err.Syntax in
      let* x = required ~operand x_id in
      let* () =
        check_output ~output ~output_shape ~expected:x.Tensor_sig.shape
      in
      Err.map_error
        (function
          | Region_context.Invalid_partition -> Invalid_partition
          | Region_context.Invalid_program error -> Invalid_program error)
        (Reduce.Softmax.Computation.program ~limits params ~x)
  | _ -> assert false

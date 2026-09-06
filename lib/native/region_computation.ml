(* The graph-boundary dispatcher for operation-authored Region computation.
   It resolves typed operands and verifies the output contract before invoking
   the operation-owned builder; it owns no operation arithmetic or fallback. *)

type error =
  | Invalid_partition
  | Invalid_program of Region_program.error
  | Invalid_shape of Shape_error.t
  | Missing_operand of Tensor_id.t
  | Output_ordinal of int
  | Output_shape

type synthetic_role = Layer_bias | Layer_weight | Rms_weight | Sdpa_mask

let pp_error fmt = function
  | Invalid_partition -> Region_context.pp_error fmt Invalid_partition
  | Invalid_program error ->
      Region_context.pp_error fmt (Region_context.Invalid_program error)
  | Invalid_shape error -> Shape_error.pp fmt error
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
  | Graph_ir.Layer_norm _ | Graph_ir.Lstm _ | Graph_ir.Rms_norm _
  | Graph_ir.Sdpa _ | Graph_ir.Softmax _ ->
      true
  | _ -> false

let built ~limits ~op ~output ~output_shape ~operand ~fill =
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
  | Lstm
      {
        Lstm.Lstm.params;
        input = input_id;
        weight_ih = weight_ih_id;
        weight_hh = weight_hh_id;
        bias;
        h0 = h0_id;
        c0 = c0_id;
      } ->
      let open Err.Syntax in
      let* input = required ~operand input_id in
      let* weight_ih = required ~operand weight_ih_id in
      let* weight_hh = required ~operand weight_hh_id in
      let* h0 = required ~operand h0_id in
      let* c0 = required ~operand c0_id in
      let* bias =
        match bias with
        | None -> Err.return None
        | Some (bi_id, bh_id) ->
            let* bi = required ~operand bi_id in
            let+ bh = required ~operand bh_id in
            Some (bi, bh)
      in
      let* out_shape, hn_shape, cn_shape =
        Err.map_error
          (fun e -> Invalid_shape e)
          (Lstm.Lstm.output_shape params ~input_shape:input.Tensor_sig.shape
             ~weight_ih_shape:weight_ih.Tensor_sig.shape
             ~weight_hh_shape:weight_hh.Tensor_sig.shape
             ~bias_shapes:
               (Option.map
                  (fun (bi, bh) -> (bi.Tensor_sig.shape, bh.Tensor_sig.shape))
                  bias)
             ~h0_shape:h0.Tensor_sig.shape ~c0_shape:c0.Tensor_sig.shape)
      in
      let* expected =
        match output with
        | 0 -> Err.return out_shape
        | 1 -> Err.return hn_shape
        | 2 -> Err.return cn_shape
        | n -> Err.fail (Output_ordinal n)
      in
      let* () =
        if Region_context.same_shape output_shape expected then Err.return ()
        else Err.fail Output_shape
      in
      Err.map_error
        (function
          | Region_context.Invalid_partition -> Invalid_partition
          | Region_context.Invalid_program error -> Invalid_program error)
        (Lstm.Lstm.Computation.program ~limits params ~output ~weight_ih
           ~weight_hh ~bias ~input ~h0 ~c0)
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

(* The OPERATION boundary: [built] resolves operands and dispatches into the
   op's own formula, but [Region_program.check] alone (what [Builder.finish]
   already runs) is not enough once scans exist -- a program is also
   [preflight]ed here, against this call's own [limits] and [output_shape],
   before it reaches Direct/Symbolic/Kernel construction. Wrapping the WHOLE
   dispatch once, rather than duplicating the call in each of the four
   [Computation.program] arms above, is what keeps this a single funnel. *)
let program ~(limits : Kernel.Limits.t) ~op ~output ~output_shape ~operand ~fill
    =
  let open Err.Syntax in
  let* program = built ~limits ~op ~output ~output_shape ~operand ~fill in
  let+ () =
    Err.map_error
      (fun e -> Invalid_program e)
      (Region_program.preflight
         ~max_local_slots:limits.Kernel.Limits.max_local_slots
         ~max_scan_state:limits.Kernel.Limits.max_scan_state
         ~max_scan_updates:limits.Kernel.Limits.max_scan_updates_per_key
         ~output_shape program)
  in
  program

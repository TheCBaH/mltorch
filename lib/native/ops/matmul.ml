(* Batch matrix multiplication (`aten.bmm`). Both inputs are rank-3, right-aligned
   to axes H/W/C: input[H=B, W=n, C=m] × mat2[H=B, W=m, C=p] → output[H=B, W=n, C=p].
   The contraction axis is [C] in [input] and [W] in [mat2]; batch [H] and row [W]
   pass through from [input]. The contraction extent is read from [input_shape.C]
   rather than carried in params — there is no weight tensor so both operands are
   runtime tensors. See .ai/native_compute_design.md §2. *)

module Bmm = struct
  type t = { input : Tensor_ref.t; mat2 : Tensor_ref.t }

  let name = "Bmm"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k = Json_util.req_field ms k Tensor_ref.jsont name in
        { input = get "input"; mat2 = get "mat2" })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj [ ("input", ref_ t.input); ("mat2", ref_ t.mat2) ])
      Jsont.json

  let operands (t : t) = [ t.input; t.mat2 ]
  let map_operands f (t : t) = { input = f t.input; mat2 = f t.mat2 }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>bmm@ input=%a@ mat2=%a@]" pp_ref t.input pp_ref t.mat2

  (* H/W from [input_shape] (batch/rows); C replaced by mat2_shape.C (columns). *)
  let output_shape ~(input_shape : Vec6.shape) ~(mat2_shape : Vec6.shape) =
    let input_batch = Vec6.get input_shape Axis.H in
    let mat2_batch = Vec6.get mat2_shape Axis.H in
    let input_contract = Vec6.get input_shape Axis.C in
    let mat2_contract = Vec6.get mat2_shape Axis.W in
    if not (Dim.equal input_batch mat2_batch) then
      Err.fail
        (`Bmm
           (Shape_error.Bmm.Batch_mismatch
              Shape_error.Bmm.{ lhs = input_batch; rhs = mat2_batch }))
    else if not (Dim.equal input_contract mat2_contract) then
      Err.fail
        (`Bmm
           (Shape_error.Bmm.Contract_mismatch
              Shape_error.Bmm.{ lhs = input_contract; rhs = mat2_contract }))
    else Err.return (Vec6.copy_axis mat2_shape Axis.C input_shape)

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel ~(input_shape : Vec6.shape) ~input ~mat2
        (out : Semantics.position S.index Vec6.t) =
      let oh = Vec6.get out Axis.H and oc = Vec6.get out Axis.C in
      S.sum ~lo:S.index_zero
        ~hi:(S.index_extent (Vec6.get input_shape Axis.C))
        (fun k ->
          S.mul
            (S.load input (out |> Vec6.set_c k))
            (S.load mat2
               (Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero ~h:oh
                  ~w:k ~c:oc)))
  end
end

(* Batched matrix multiplication over the full attention frame `[D,H,W,C]`
   (`matmul.default`'s batched/multi-head shape family -- `mvitv2_tiny`'s real
   multi-head attention, `.ai/matmul_softmax_design.md` §5): input[D=batch,
   H=heads, W=n, C=m] × mat2[D=batch, H=heads, W=m, C=p] -> output[D=batch,
   H=heads, W=n, C=p], contracting [C] against [W] exactly like [Bmm]. The
   generalization over [Bmm] is in which axes name the batch: [Bmm] hard-codes
   N/T/D at index 0 on [mat2] (correct only because the batch-less importer
   restriction keeps them at extent 1 on both operands); here every one of
   [N]/[T]/[D]/[H] is read off the OUTPUT coordinate on both operands, exactly
   the same generalization `attention_design.md` §2 made for [Sdpa] itself. *)
module Batched_matmul = struct
  type t = { input : Tensor_ref.t; mat2 : Tensor_ref.t }

  let name = "Batched_matmul"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k = Json_util.req_field ms k Tensor_ref.jsont name in
        { input = get "input"; mat2 = get "mat2" })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj [ ("input", ref_ t.input); ("mat2", ref_ t.mat2) ])
      Jsont.json

  let operands (t : t) = [ t.input; t.mat2 ]
  let map_operands f (t : t) = { input = f t.input; mat2 = f t.mat2 }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>batched_matmul@ input=%a@ mat2=%a@]" pp_ref t.input
      pp_ref t.mat2

  let batch_axes = [ Axis.N; Axis.T; Axis.D; Axis.H ]

  (* Per-axis ATen broadcasting on the four batch-like axes only: equal, or one
     side is 1 (the broadcast axis, which takes the other's extent) -- the same
     per-axis rule `Pointwise_binary.broadcast_output_shape` applies to every
     axis, restricted here to [batch_axes] since [W]/[C] are the matmul row/
     contraction axes, not batch dimensions. `lambda_resnet26t`'s real corpus
     occurrence (`H`: 4 vs 1) is exactly this case -- `.ai/matmul_softmax_design.md`
     §6's "no corpus evidence either way" no longer holds. *)
  let broadcast_extent ~axis (a : Vec6.shape) (b : Vec6.shape) =
    let lhs = Vec6.get a axis and rhs = Vec6.get b axis in
    if Dim.equal lhs rhs then Err.return lhs
    else if Dim.equal lhs Dim.one then Err.return rhs
    else if Dim.equal rhs Dim.one then Err.return lhs
    else
      Err.fail
        (`Batched_matmul
           (Shape_error.Batched_matmul.Batch_mismatch
              Shape_error.Batched_matmul.{ axis; lhs; rhs }))

  let output_shape ~(input_shape : Vec6.shape) ~(mat2_shape : Vec6.shape) =
    let open Err.Syntax in
    let* batch_shape =
      Err.List.fold_left
        (fun s axis ->
          let* extent = broadcast_extent ~axis input_shape mat2_shape in
          Err.return (Vec6.set s axis extent))
        input_shape batch_axes
    in
    let input_contract = Vec6.get input_shape Axis.C in
    let mat2_contract = Vec6.get mat2_shape Axis.W in
    if not (Dim.equal input_contract mat2_contract) then
      Err.fail
        (`Batched_matmul
           (Shape_error.Batched_matmul.Contract_mismatch
              { lhs = input_contract; rhs = mat2_contract }))
    else Err.return (Vec6.copy_axis mat2_shape Axis.C batch_shape)

  module Compute (S : Semantics.SEMANTICS) = struct
    (* Each operand is read at [out] reduced against ITS OWN shape on every
       broadcast (extent-1) axis -- [Pointwise_binary.broadcast_coord], the
       same helper [Add]/[Mul]/... use, reused rather than restating the
       clamp-to-zero rule. [C]/[W] are unaffected: [input]'s own [C] is
       already replaced by the contraction variable [k] before the clamp
       runs, and [mat2]'s own [W] likewise, so the helper only ever touches
       [N]/[T]/[D]/[H] here even though it is not itself axis-restricted. *)
    let pixel ~(input_shape : Vec6.shape) ~(mat2_shape : Vec6.shape) ~input
        ~mat2 (out : Semantics.position S.index Vec6.t) =
      let index_zero = S.index_zero in
      S.sum ~lo:index_zero
        ~hi:(S.index_extent (Vec6.get input_shape Axis.C))
        (fun k ->
          S.mul
            (S.load input
               (Pointwise_binary.broadcast_coord ~index_zero input_shape
                  (out |> Vec6.set_c k)))
            (S.load mat2
               (Pointwise_binary.broadcast_coord ~index_zero mat2_shape
                  (Vec6.set out Axis.W k))))
  end
end

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
      Core.fail
        (`Bmm
           (Shape_error.Bmm.Batch_mismatch
              Shape_error.Bmm.{ lhs = input_batch; rhs = mat2_batch }))
    else if not (Dim.equal input_contract mat2_contract) then
      Core.fail
        (`Bmm
           (Shape_error.Bmm.Contract_mismatch
              Shape_error.Bmm.{ lhs = input_contract; rhs = mat2_contract }))
    else Core.return (Vec6.copy_axis mat2_shape Axis.C input_shape)

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

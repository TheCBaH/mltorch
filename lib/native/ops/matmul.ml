(* Batch matrix multiplication (`aten.bmm`). Both inputs are rank-3, right-aligned
   to axes H/W/C: input[H=B, W=n, C=m] × mat2[H=B, W=m, C=p] → output[H=B, W=n, C=p].
   The contraction axis is [C] in [input] and [W] in [mat2]; batch [H] and row [W]
   pass through from [input]. The contraction extent is read from [input_shape.C]
   rather than carried in params — there is no weight tensor so both operands are
   runtime tensors. See .ai/native_compute_design.md §2. *)

module Bmm = struct
  (* H/W from [input_shape] (batch/rows); C replaced by mat2_shape.C (columns). *)
  let output_shape ~(input_shape : Vec6.shape) ~(mat2_shape : Vec6.shape) :
      Vec6.shape =
    Vec6.set input_shape Axis.C (Vec6.get mat2_shape Axis.C)

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel ~(input_shape : Vec6.shape) ~input ~mat2
        (out : Axis.t -> Semantics.index S.ix) : S.t =
      S.sum ~lo:S.izero
        ~hi:(S.iext (Vec6.get input_shape Axis.C))
        (fun k ->
          let input_idx a = match a with Axis.C -> k | _ -> out a in
          let mat2_idx a =
            match a with Axis.W -> k | Axis.H | Axis.C -> out a | _ -> S.izero
          in
          S.mul (S.load input input_idx) (S.load mat2 mat2_idx))
  end
end

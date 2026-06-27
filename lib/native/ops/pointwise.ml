(* Pointwise ops: each output pixel reads the input(s) at the same coord. The
   functor is over SEMANTICS, so the same definition serves Direct and Symbolic.
   [out] gives the output coord as one index expression per axis.
   See .ai/native_compute_design.md §2. *)

module Relu = struct
  (* Identity: relu doesn't change shape. See .ai/native_compute_design.md §2b. *)
  let output_shape (x_shape : Vec6.shape) = x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* relu x = (x < 0 ? 0 : x) — derived from [select]+[lt], not a primitive *)
    let pixel x (out : Axis.t -> Semantics.position S.index) =
      let v = S.load x out in
      S.select (S.lt v (S.const 0.)) (S.const 0.) v
  end
end

module Add = struct
  (* Per axis, the non-broadcast (non-extent-1) operand's extent — i.e. the
     larger of the two. Computed in extent-space ([Dim.max], no [:> int]); start
     from [a_shape] and overwrite every axis. See
     .ai/native_compute_design.md §2b. *)
  let output_shape (a_shape : Vec6.shape) (b_shape : Vec6.shape) =
    List.fold_left
      (fun s axis ->
        Vec6.set s axis
          (Dim.max (Vec6.get a_shape axis) (Vec6.get b_shape axis)))
      a_shape Axis.all

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [b] may have extent-1 axes; [load] broadcasts them against b's own shape. *)
    let pixel a b (out : Axis.t -> Semantics.position S.index) =
      S.add (S.load a out) (S.load b out)
  end
end

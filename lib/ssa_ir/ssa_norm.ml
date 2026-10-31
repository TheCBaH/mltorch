(* The loops of a row normalization, built from its descriptor rather than taken
   from the Native norm's lowered program, so the descriptor defines the
   arithmetic and its graph-free tests and the attention layer's body share one
   builder. *)

module B = Ssa_builder
module N = Norm_block
module P = Attention_block.Precision
open Ssa_expand_util

let body b (d : N.t) ~x ~weight ~bias ~y =
  let store =
    match d.N.precision with
    | P.Bf16_paired -> store_bf16
    | P.F32_fused | P.F32_ordered -> store
  in
  let operand present buffer name =
    match (present, buffer) with
    | false, _ -> None
    | true, Some id -> Some id
    | true, None -> invalid_arg ("Ssa_norm: no " ^ name ^ " buffer")
  in
  let weight = operand d.N.weight weight "weight"
  and bias = operand d.N.bias bias "bias" in
  let f = B.f64 b in
  let zero = f 0. and one = f 1. in
  let n = f (Int64.to_float d.N.width) and eps = f d.N.eps in
  let binary b op x y = B.f64_binary b op x y in
  let width b = B.index b d.N.width in
  loop b d.N.rows (fun b r ->
      let xat b c =
        load b x
          (offset b [ (d.N.x.N.Strides.row, r); (d.N.x.N.Strides.col, c) ])
      in
      let squares b v =
        match d.N.precision with
        | P.F32_fused -> (
            match
              B.for_ b ~lo:(B.index b 0L) ~hi:(width b)
                ~init:(B.Cons (zero, B.Nil))
                (fun b c (B.Cons (acc, B.Nil)) ->
                  let e = v b c in
                  B.Cons (B.f64_fma b e e acc, B.Nil))
            with
            | B.Cons (t, B.Nil) -> t)
        | P.Bf16_paired | P.F32_ordered ->
            B.ordered_sum b ~lo:(B.index b 0L) ~hi:(width b) ~seed:zero
              (fun b c ->
                let e = v b c in
                binary b Expr.Value.Mul e e)
      in
      let centre, t =
        match d.N.kind with
        | N.Kind.Layer ->
            let s =
              B.ordered_sum b ~lo:(B.index b 0L) ~hi:(width b) ~seed:zero xat
            in
            let mean = binary b Expr.Value.Div s n in
            let centre b c = binary b Expr.Value.Sub (xat b c) mean in
            (centre, squares b centre)
        | N.Kind.Rms -> (xat, squares b xat)
      in
      let inv =
        binary b Expr.Value.Div one
          (B.f64_unary b Expr.Value.Sqrt
             (binary b Expr.Value.Add (binary b Expr.Value.Div t n) eps))
      in
      loop b d.N.width (fun b c ->
          let v = binary b Expr.Value.Mul (centre b c) inv in
          let v =
            match weight with
            | Some w -> binary b Expr.Value.Mul v (load b w c)
            | None -> v
          in
          let v =
            match bias with
            | Some bi -> binary b Expr.Value.Add v (load b bi c)
            | None -> v
          in
          store b y
            (offset b [ (d.N.y.N.Strides.row, r); (d.N.y.N.Strides.col, c) ])
            v))

let program d ~x ~weight ~bias ~y ~buffers =
  template ~buffers (fun b -> body b d ~x ~weight ~bias ~y)

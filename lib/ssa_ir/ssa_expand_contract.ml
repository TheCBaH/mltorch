(* A contraction as ordinary operations, for a backend with no [Contract] of its
   own: the descriptor's loop nest over batch, row and column, an ordered sum over
   [k] seeded with positive zero or the output's own cell, and a store. Only the
   [Ordered] numerics over binary32 or bfloat16 operands, which are an ordered
   sum of rounded products, and [Fused_sequential] over binary32, whose sum is
   the same loop with each step fused: the fusion is the plan's, a policy that
   contracts dots ({!Ssa_numerics.Contraction.Dots}) fusing exactly an ordered
   sum's single-product term in source order from the seed. A paired or integer
   contraction stays a [Contract]. The arithmetic is binary64 until
   {!Ssa_precision.to_f32}. *)

module B = Ssa_builder
open Ssa_expand_util

let expandable (d : Contraction.t) =
  match (d.Contraction.numerics, d.Contraction.format) with
  | ( Contraction.Numerics.Ordered,
      (Contraction.Element.Bf16 | Contraction.Element.F32) )
  | Contraction.Numerics.Fused_sequential, Contraction.Element.F32 ->
      true
  | ( Contraction.Numerics.Fused_sequential,
      (Contraction.Element.Bf16 | Contraction.Element.I8 _) )
  | Contraction.Numerics.Ordered, Contraction.Element.I8 _
  | Contraction.Numerics.Paired, _ ->
      false

(* How an operand's cells are placed inside the template: by the descriptor
   alone (a whole buffer), moved by one base, or by one base per row of the
   operand's outer axis, which replaces the descriptor's outer stride. *)
type view =
  | Base of Ssa_type.index Ssa_builder.value
  | Rows of Ssa_type.index Ssa_builder.value array

(* [views] makes, inside the template, how each operand (A, B, C) is placed: a
   view call's bases ({!Ssa_view_call}); none for whole buffers. An axis some
   operand names by rows is unrolled, so each row's base is a known value. *)
let expansion ?views (d : Contraction.t) ~a ~b ~c ~buffers =
  let decode =
    match d.Contraction.format with
    | Contraction.Element.Bf16 -> Ssa_op.Decode.Bf16_to_f64
    | Contraction.Element.F32 | Contraction.Element.I8 _ ->
        Ssa_op.Decode.F32_to_f64
  in
  let ao = d.Contraction.a and bo = d.Contraction.b in
  let co = d.Contraction.c in
  template ~buffers (fun bld ->
      let va, vb, vc =
        match views with
        | None -> (None, None, None)
        | Some make ->
            let va, vb, vc = make bld in
            (Some va, Some vb, Some vc)
      in
      let rows = function Some (Rows _) -> true | _ -> false in
      let unroll_i = rows va || rows vc and unroll_j = rows vb in
      (* the cell of an operand with strides [batch], [outer] (on the outer
         axis position [o]), and [inner] on the inner index [x] *)
      let place bld view ~batch ~outer ~inner (bi, o, ostatic, x) =
        let moved base cell = B.index_add bld base cell in
        match (view, ostatic) with
        | Some (Rows r), Some s ->
            moved r.(s) (offset bld [ (batch, bi); (inner, x) ])
        | Some (Base v), _ ->
            moved v (offset bld [ (batch, bi); (outer, o); (inner, x) ])
        | (Some (Rows _) | None), _ ->
            offset bld [ (batch, bi); (outer, o); (inner, x) ]
      in
      let axis bld n ~unroll body =
        if unroll then
          for s = 0 to Int64.to_int n - 1 do
            body bld (B.index bld (Int64.of_int s)) (Some s)
          done
        else loop bld n (fun bld iv -> body bld iv None)
      in
      loop bld d.Contraction.batch (fun bld bi ->
          axis bld d.Contraction.m ~unroll:unroll_i (fun bld i si ->
              axis bld d.Contraction.n ~unroll:unroll_j (fun bld j sj ->
                  let cell =
                    place bld vc ~batch:co.Contraction.Output.batch
                      ~outer:co.Contraction.Output.row
                      ~inner:co.Contraction.Output.col (bi, i, si, j)
                  in
                  let seed =
                    match d.Contraction.seed with
                    | Contraction.Seed.Zero -> B.f64 bld 0.
                    | Contraction.Seed.Tensor -> load bld c cell
                  in
                  let sum =
                    B.ordered_sum bld ~lo:(B.index bld 0L)
                      ~hi:(B.index bld d.Contraction.k) ~seed (fun bld k ->
                        let x =
                          load ~decode bld a
                            (place bld va ~batch:ao.Contraction.Operand.batch
                               ~outer:ao.Contraction.Operand.outer
                               ~inner:ao.Contraction.Operand.k (bi, i, si, k))
                        and y =
                          load ~decode bld b
                            (place bld vb ~batch:bo.Contraction.Operand.batch
                               ~outer:bo.Contraction.Operand.outer
                               ~inner:bo.Contraction.Operand.k (bi, j, sj, k))
                        in
                        B.f64_binary bld Expr.Value.Mul x y)
                  in
                  store bld c cell sum))))

let program (p : Ssa_program.t) =
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr
        {
          Ssa_instr.op = Ssa_op.Contract { desc; a; b; c; _ };
          token = Some token;
          results = [ after ];
          _;
        }
      when expandable desc ->
        let template =
          expansion desc ~a ~b ~c ~buffers:(declared p [ a; b; c ])
        in
        splice t template ~token ~after
    | _ -> [ s ]
  in
  fst (Ssa_rewrite.program rule p)

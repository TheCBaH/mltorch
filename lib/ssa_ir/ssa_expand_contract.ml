(* A contraction as ordinary operations, for a backend with no [Contract] of its
   own: the descriptor's loop nest over batch, row and column, an ordered sum over
   [k] seeded with positive zero or the output's own cell, and a store. Only the
   [Ordered] numerics over binary32 or bfloat16 operands, which are an ordered
   sum of rounded products; a fused, paired or integer contraction stays a
   [Contract]. The arithmetic is binary64 until {!Ssa_precision.to_f32}. *)

module B = Ssa_builder
open Ssa_expand_util

let expandable (d : Contraction.t) =
  d.Contraction.numerics = Contraction.Numerics.Ordered
  &&
  match d.Contraction.format with
  | Contraction.Element.Bf16 | Contraction.Element.F32 -> true
  | Contraction.Element.I8 _ -> false

let expansion (d : Contraction.t) ~a ~b ~c ~buffers =
  let decode =
    match d.Contraction.format with
    | Contraction.Element.Bf16 -> Ssa_op.Decode.Bf16_to_f64
    | Contraction.Element.F32 | Contraction.Element.I8 _ ->
        Ssa_op.Decode.F32_to_f64
  in
  let ao = d.Contraction.a and bo = d.Contraction.b in
  let co = d.Contraction.c in
  template ~buffers (fun bld ->
      loop bld d.Contraction.batch (fun bld bi ->
          loop bld d.Contraction.m (fun bld i ->
              loop bld d.Contraction.n (fun bld j ->
                  let cell =
                    offset bld
                      [
                        (co.Contraction.Output.batch, bi);
                        (co.Contraction.Output.row, i);
                        (co.Contraction.Output.col, j);
                      ]
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
                            (offset bld
                               [
                                 (ao.Contraction.Operand.batch, bi);
                                 (ao.Contraction.Operand.outer, i);
                                 (ao.Contraction.Operand.k, k);
                               ])
                        and y =
                          load ~decode bld b
                            (offset bld
                               [
                                 (bo.Contraction.Operand.batch, bi);
                                 (bo.Contraction.Operand.outer, j);
                                 (bo.Contraction.Operand.k, k);
                               ])
                        in
                        B.f64_binary bld Expr.Value.Mul x y)
                  in
                  store bld c cell sum))))

let program (p : Ssa_program.t) =
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr
        {
          Ssa_instr.op = Ssa_op.Contract { desc; a; b; c };
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

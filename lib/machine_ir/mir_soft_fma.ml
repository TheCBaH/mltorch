(* A binary32 fused multiply-add as ordinary operations, for a target without
   one: {!F32_math.fma32} step for step. The product of two binary32 values is
   exact in binary64; TwoSum splits its sum with [c] into [s + e]; when [e] is
   nonzero [s] is replaced by the neighbouring binary64 whose last mantissa bit
   is odd (a bitcast, an integer nudge toward or away from zero), so the final
   rounding to binary32 sees a sticky bit. An [e] that is zero or NaN (a
   nonfinite operand) takes [s] itself, as [fma32]'s ordinary-arithmetic
   fallback does. Every step is evaluated and the result selected: no branch.
   Scalar binary32 only; a vector [Ffma], or a binary64 one, is left for the
   target to refuse. The result is verified again. *)

let func (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
  let next_value, next_instr = Mir_select.watermarks f in
  let next_value = ref next_value and next_instr = ref next_instr in
  let emit out (origin : Mir_origin.t) ?result ty op =
    let r =
      match result with
      | Some r -> r
      | None ->
          incr next_value;
          { Mir_value.id = Mir_id.Value.of_int !next_value; ty }
    in
    incr next_instr;
    out :=
      {
        Mir_instr.id = Mir_id.Instr.of_int !next_instr;
        results = [ r ];
        op;
        order = None;
        origin;
      }
      :: !out;
    r
  in
  let expand (i : Mir_op.t Mir_instr.t) a b c result =
    let out = ref [] in
    let origin = i.Mir_instr.origin in
    let f64 = emit out origin Mir_type.F64 in
    let pred = emit out origin Mir_type.Pred in
    let i64 = emit out origin Mir_type.i64 in
    let widen x = f64 (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, x)) in
    let bin o x y = f64 (Mir_op.Fbinary (o, x, y)) in
    let const_f64 x = f64 (Mir_op.Const (Mir_const.f64 x)) in
    let const_i64 x = i64 (Mir_op.Const (Mir_const.i64 x)) in
    let a = widen a and b = widen b and c = widen c in
    let p = bin Mir_op.Fbinary.Mul a b in
    let s = bin Mir_op.Fbinary.Add p c in
    let bb = bin Mir_op.Fbinary.Sub s p in
    let e =
      bin Mir_op.Fbinary.Add
        (bin Mir_op.Fbinary.Sub p (bin Mir_op.Fbinary.Sub s bb))
        (bin Mir_op.Fbinary.Sub c bb)
    in
    let zero = const_f64 0. in
    let exact = pred (Mir_op.Fcmp (Mir_op.Fcmp.Eq, e, zero)) in
    let nonfinite = pred (Mir_op.Fcmp (Mir_op.Fcmp.Unordered, e, e)) in
    let narrow x =
      emit out origin Mir_type.F32
        (Mir_op.Fconvert (Mir_op.Fconvert.F64_to_f32, x))
    in
    let plain = narrow s in
    let bits = i64 (Mir_op.Bitcast (Mir_type.i64, s)) in
    let one = const_i64 1L in
    let odd_already =
      pred
        (Mir_op.Icmp
           ( Mir_op.Icmp.Eq,
             i64 (Mir_op.Iarith (Mir_op.Iarith.And, bits, one)),
             one ))
    in
    (* toward the larger magnitude when [e] and [s] have the same sign *)
    let same_sign =
      pred
        (Mir_op.Pnot
           (pred
              (Mir_op.Pbinary
                 ( Mir_op.Pbinary.Xor,
                   pred (Mir_op.Fcmp (Mir_op.Fcmp.Lt, zero, e)),
                   pred (Mir_op.Fcmp (Mir_op.Fcmp.Lt, zero, s)) ))))
    in
    let step = i64 (Mir_op.Select (same_sign, one, const_i64 (-1L))) in
    let nudged = i64 (Mir_op.Iarith (Mir_op.Iarith.Add, bits, step)) in
    let odd_bits = i64 (Mir_op.Select (odd_already, bits, nudged)) in
    let odd = narrow (f64 (Mir_op.Bitcast (Mir_type.F64, odd_bits))) in
    let chosen =
      emit out origin Mir_type.F32 (Mir_op.Select (exact, plain, odd))
    in
    ignore
      (emit out origin ~result Mir_type.F32
         (Mir_op.Select (nonfinite, plain, chosen)));
    List.rev !out
  in
  let blocks =
    List.map
      (fun (blk : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
        let body =
          List.concat_map
            (fun (i : Mir_op.t Mir_instr.t) ->
              match (i.Mir_instr.op, i.Mir_instr.results) with
              | Mir_op.Ffma (a, b, c), [ result ]
                when Mir_type.equal result.Mir_value.ty Mir_type.F32 ->
                  expand i a b c result
              | _ -> [ i ])
            blk.Mir_block.body
        in
        { blk with Mir_block.body })
      f.Mir_func.blocks
  in
  { f with Mir_func.blocks }

let program (g : Mir_verify.Generic.t) =
  let p = Mir_verify.Generic.program g in
  Err.or_raise ~pp_error:Mir_diagnostic.pp
    (Mir_verify.generic
       { p with Mir_program.funcs = List.map func p.Mir_program.funcs })

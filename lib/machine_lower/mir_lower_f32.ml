(* The binary32 pieces of the lowering: the project's [F32_math] sequences as
   generic instructions, and a quantized read's explicit operations. Split from
   {!Mir_lower}, which holds the instruction-by-instruction walk. *)

open Ssa_ir
open Machine_ir
include Mir_lower_core
module L = Mir_layout_map

let sext32 st v = emit st (Mir_op.Iext (Mir_op.Iext.Sext, Mir_width.W32, v))

(* The project's binary32 sequences ({!F32_math}) as Machine IR: every
   operation of a sequence is one generic binary32 instruction, so every target
   gets it with no helper, no call and no binary64 value. *)
module type F32_fns = sig
  val cos : Mir_value.t -> Mir_value.t
  val erf : Mir_value.t -> Mir_value.t
  val exp : Mir_value.t -> Mir_value.t
  val log : Mir_value.t -> Mir_value.t
  val sin : Mir_value.t -> Mir_value.t
end

let f32_math st : (module F32_fns) =
  let module O = struct
    type t = Mir_value.t

    let const x = const st (Mir_const.f32 (Ssa_const.round_f32 x))
    let bin o a b = emit st (Mir_op.Fbinary (o, a, b))
    let add = bin Mir_op.Fbinary.Add
    let sub = bin Mir_op.Fbinary.Sub
    let mul = bin Mir_op.Fbinary.Mul
    let div = bin Mir_op.Fbinary.Div

    let select_lt a b c d =
      select st (emit st (Mir_op.Fcmp (Mir_op.Fcmp.Lt, a, b))) c d

    let select_eq a b c d =
      select st (emit st (Mir_op.Fcmp (Mir_op.Fcmp.Eq, a, b))) c d

    let share x = x
  end in
  (module F32_math.Make (O) : F32_fns)

(* The same sequences on a binary32 vector of [lanes]: a constant is the
   binary32 constant splat to the lanes, every operation lanewise and a select
   on the lanewise comparison's mask. Each lane computes exactly the scalar
   sequence, so the vector is the scalar function lane by lane. *)
let f32_math_vector st ~lanes : (module F32_fns) =
  let module O = struct
    type t = Mir_value.t

    let const x =
      emit st
        (Mir_op.Vsplat (lanes, const st (Mir_const.f32 (Ssa_const.round_f32 x))))

    let bin o a b = emit st (Mir_op.Fbinary (o, a, b))
    let add = bin Mir_op.Fbinary.Add
    let sub = bin Mir_op.Fbinary.Sub
    let mul = bin Mir_op.Fbinary.Mul
    let div = bin Mir_op.Fbinary.Div

    let select_lt a b c d =
      select st (emit st (Mir_op.Fcmp (Mir_op.Fcmp.Lt, a, b))) c d

    let select_eq a b c d =
      select st (emit st (Mir_op.Fcmp (Mir_op.Fcmp.Eq, a, b))) c d

    let share x = x
  end in
  (module F32_math.Make (O) : F32_fns)

(* [scale * (q - zero_point)] as the explicit operations: [q] to float, minus
   the zero point, times the scale, each rounded once, in binary32 ([f32]) or
   in binary64 over the same binary32 parameters. The parameters are per
   tensor, or per channel from the buffer's parameter table at its C
   coordinate. *)
let dequantize st op (e : L.Entry.t) q ~channel ~f32 =
  let q = sext st (sext32 st q) in
  let konst x = const st (if f32 then Mir_const.f32 x else Mir_const.f64 x) in
  let s, z =
    match
      ( Ssa_format.quant e.L.Entry.buffer.Ssa_buffer.format,
        L.Params.view e,
        channel )
    with
    | Some (Ssa_format.Per_tensor { scale; zero_point }), _, _ ->
        (konst scale, konst zero_point)
    | Some (Ssa_format.Per_channel _), Some table, Some c ->
        let base = emit st (Mir_op.Addr table) in
        let c = if mutated st Mutation.Channel_zero then i32k st 0L else c in
        let off =
          emit st (Mir_op.Iarith (Mir_op.Iarith.Mul, sext st c, i64k st 4L))
        in
        let at k =
          emit st
            (Mir_op.Ptr_add
               ( base,
                 emit st (Mir_op.Iarith (Mir_op.Iarith.Add, off, i64k st k)) ))
        in
        let load a =
          let bits =
            emit st
              (Mir_op.Load
                 { Mir_op.Access.width = Mir_width.W32; addr = a; align = 4L })
          in
          let v = emit st (Mir_op.Bitcast (Mir_type.F32, bits)) in
          if f32 then v
          else emit st (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, v))
        in
        (load (at 0L), load (at (L.Params.zero_point_offset e)))
    | _ -> unsupported st op
  in
  let qf =
    emit st
      (Mir_op.Fconvert
         ( (if f32 then Mir_op.Fconvert.S64_to_f32
            else Mir_op.Fconvert.S64_to_f64),
           q ))
  in
  emit st
    (Mir_op.Fbinary
       ( Mir_op.Fbinary.Mul,
         emit st (Mir_op.Fbinary (Mir_op.Fbinary.Sub, qf, z)),
         s ))

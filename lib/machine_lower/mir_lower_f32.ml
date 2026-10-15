(* The binary32 pieces of the lowering: the project's [F32_math] sequences as
   generic instructions. Split from {!Mir_lower}, which holds the
   instruction-by-instruction walk. *)

open Ssa_ir
open Machine_ir
include Mir_lower_core

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

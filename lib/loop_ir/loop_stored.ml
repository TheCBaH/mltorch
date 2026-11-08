(* A value paired with the buffer's encode, so a store's own conversion is
   visible rather than implied by the buffer type. [Bf16] rounds to binary32,
   then to bfloat16 ([Expr.Value.bf16_value]) on write; [Bool] carries a working
   float and writes [v <> 0.] as a canonical 0/1 byte; [F32] rounds to binary32
   on write. *)
type t =
  | Bf16 of float Loop_expr.t
  | Bool of float Loop_expr.t
  | F32 of float Loop_expr.t
  | I64 of int64 Loop_expr.t

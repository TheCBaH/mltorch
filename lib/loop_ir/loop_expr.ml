(* [Expr.Value.t]'s pure scalar forms over the [float]/[int64] carriers. Every
   binding form is gone: a [Reduce] is a loop with accumulator temporaries, a
   [Local] is an array read, and a load names a typed buffer. *)
type _ t =
  | Array_get : Loop_array.t * Loop_index.t -> float t
  | Binary : Expr.Value.binary_op * float t * float t -> float t
  | Const : float -> float t
  | Float_max : float t * float t -> float t
  | Float_to_i64 : float t -> int64 t
  | I64_binary : Expr.Value.i64_binary_op * int64 t * int64 t -> int64 t
  | I64_const : int64 -> int64 t
  | I64_of_index : Loop_index.t -> int64 t
  | I64_to_float : int64 t -> float t
  | Load : Loop_buffer.t * Loop_index.coord -> float t
  | Load_i64 : Loop_buffer.t * Loop_index.coord -> int64 t
  | Round_f32 : float t -> float t
  | Select : (float t, int64 t) Loop_bool.t * 'a t * 'a t -> 'a t
  | Temp : 'a Loop_carrier.t * Loop_temp.t -> 'a t
  | Unary : Expr.Value.unary_op * float t -> float t
  | Value_of_index : Loop_index.t -> float t

type pred = (float t, int64 t) Loop_bool.t

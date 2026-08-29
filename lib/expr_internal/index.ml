(* Concrete internally; [Expr.Index] exposes this type as private.  Keeping
   this unit without an [.mli] lets trusted transformations rebuild syntax
   without adding public unsafe constructors. *)

type _ t =
  | Add : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
  | Assume_position : Role.Delta.t t -> Role.Position.t t
  | Ceil_div_pos : Role.Delta.t t * int -> Role.Delta.t t
  | Clamp_low : Role.Delta.t t -> Role.Position.t t
  | Const : int -> Role.Delta.t t
  | Floor_div_pos : Role.Delta.t t * int -> Role.Delta.t t
  | Max : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
  | Min : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
  | Of_position : Role.Position.t t -> Role.Delta.t t
  | Output : Axis.t -> Role.Position.t t
  | Reduce : Reduce_var.t -> Role.Position.t t
  | Scale : int * Role.Delta.t t -> Role.Delta.t t
  | Zero : Role.Position.t t

type error = [ `Non_positive_divisor of int ]

let pp_error fmt : [< error ] -> unit = function
  | `Non_positive_divisor d -> Fmt.pf fmt "divisor must be > 0, got %d" d

let output a = Output a
let reduce v = Reduce v
let zero = Zero
let const n = Const n
let of_position i = Of_position i
let add a b = match (a, b) with Const 0, x | x, Const 0 -> x | _ -> Add (a, b)

let scale k a =
  match (k, a) with 1, _ -> a | _, Const 0 -> Const 0 | _ -> Scale (k, a)

let min a b = Min (a, b)
let max a b = Max (a, b)
let clamp_low a = Clamp_low a
let assume_position a = Assume_position a

let floor_div_pos a d =
  if d <= 0 then Err.fail (`Non_positive_divisor d)
  else if d = 1 then Err.return a
  else Err.return (Floor_div_pos (a, d))

let ceil_div_pos a d =
  if d <= 0 then Err.fail (`Non_positive_divisor d)
  else if d = 1 then Err.return a
  else Err.return (Ceil_div_pos (a, d))

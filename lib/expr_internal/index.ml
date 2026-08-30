(* Concrete internally; [Expr.Index] exposes this type as private.  Keeping
   this unit without an [.mli] lets trusted transformations rebuild syntax
   without adding public unsafe constructors. *)

(* [Data]'s coordinate field is [Role.Position.t t Coord.t] -- a [Coord.t]
   whose six per-axis components are themselves [Role.Position.t t]
   expressions in THIS SAME GADT. That makes [Data] self-recursive within
   [Index.t], not mutually recursive with [Value.t]: a [Load]'s own
   coordinate is still [Role.Position.t Index.t Coord.t] one level up, in
   [Value.t], unaffected. [Data]'s third field is the gathered axis's extent;
   it is a plain [int] here (checked positive only where it originates,
   [Dim.extent] on the [native] side that builds this node) rather than
   [Dim.extent Dim.t], since [expr_internal] must not depend on [native] --
   see CLAUDE.md's dependency-direction rule and dim.mli's own comment on why
   [index]/[delta] are the shared vocabulary but [extent] stays local to
   [native]. *)
type _ t =
  | Add : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
  | Assume_position : Role.Delta.t t -> Role.Position.t t
  | Ceil_div_pos : Role.Delta.t t * int -> Role.Delta.t t
  | Clamp_low : Role.Delta.t t -> Role.Position.t t
  | Const : int -> Role.Delta.t t
  | Data : Source.t * Role.Position.t t Coord.t * int -> Role.Position.t t
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

(* No folding: unlike [add]/[scale], [Data] has no exact integer identity to
   collapse to, and its value is not known until resolved (never at
   construction time -- see the [Index.Data] design record). *)
let data source coord extent = Data (source, coord, extent)
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

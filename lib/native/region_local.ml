module Rhs = struct
  type t =
    | Scalar of Expr.Value.t
    | Vector of { extent : int; var : Expr.Reduce_var.t; body : Expr.Value.t }

  let scalar value = Scalar value
  let vector ~extent ~var ~body = Vector { extent; var; body }
  let slot_count = function Scalar _ -> 1 | Vector { extent; _ } -> extent
  let value = function Scalar value -> value | Vector { body; _ } -> body
end

module Shape = struct
  type t = Scalar | Vector of { extent : int }

  let of_rhs = function
    | Rhs.Scalar _ -> Scalar
    | Rhs.Vector { extent; _ } -> Vector { extent }

  let pp fmt = function
    | Scalar -> Fmt.string fmt "scalar"
    | Vector { extent } -> Fmt.pf fmt "vector[%d]" extent
end

type t = { id : Expr.Local_var.t; rhs : Rhs.t }

let scalar ~id ~value = { id; rhs = Rhs.scalar value }

(* [value]'s body may freely mention [var] (via [Expr.Index.reduce var]) as
   its own per-element index -- the binder [Rhs.vector] mints and stores
   alongside [extent], never bound by a [Value.Reduce] node inside [value]
   itself. Reading it back at a computed index ([Expr.Value.local_at]) is a
   beta-reduction of that binder, carried out by
   [Expr.Rewrite.substitute_locals]'s [Vector] case during specialization. *)
let vector ~id ~var ~extent ~value =
  { id; rhs = Rhs.vector ~extent ~var ~body:value }

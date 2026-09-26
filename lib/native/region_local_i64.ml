(* [int64 Expr.Value.t] counterpart of [Region_local.t]: scalar/vector Region
   locals only, mirroring [Region_local.Rhs]'s [Scalar]/[Vector] shape
   exactly, at the carrier [Expr.Value.i64_local]/[i64_local_at] read. No
   [Scan] variant -- a scan's own [init]/[update]/[prev] stay [float
   Expr.Value.t] (see [Expr_repr.scan]), so a typed I64 trace local is later
   work, not this module's.

   Kept as a SEPARATE type from [Region_local.t] rather than generalizing it
   into a GADT/existential, so every existing float-only Region caller (every
   op building a [Region_program.t] today, via [Region_local.scalar]/
   [.vector]/[.scan] and [Region_program.Builder]) is source-unchanged. *)

module Rhs = struct
  type t =
    | Scalar of int64 Expr.Value.t
    | Vector of {
        extent : Slot.extent Slot.t;
        var : Expr.Reduce_var.t;
        body : int64 Expr.Value.t;
      }

  let scalar value = Scalar value
  let vector ~extent ~var ~body = Vector { extent; var; body }

  let slot_count = function
    | Scalar _ -> Slot.one
    | Vector { extent; _ } -> Slot.count_of_extent extent

  let value = function Scalar value -> value | Vector { body; _ } -> body
end

module Shape = struct
  type t = Scalar | Vector of { extent : Slot.extent Slot.t }

  let of_rhs = function
    | Rhs.Scalar _ -> Scalar
    | Rhs.Vector { extent; _ } -> Vector { extent }

  let pp fmt = function
    | Scalar -> Fmt.string fmt "scalar"
    | Vector { extent } -> Fmt.pf fmt "vector[%a]" Slot.pp extent
end

type t = { id : Expr.Local_var.t; rhs : Rhs.t }

let scalar ~id ~value = { id; rhs = Rhs.scalar value }

(* [value]'s body may freely mention [var] (via [Expr.Index.reduce var]) as
   its own per-element index, mirroring [Region_local.vector]'s identical
   contract at [float]. *)
let vector ~id ~var ~extent ~value =
  { id; rhs = Rhs.vector ~extent ~var ~body:value }

module Rhs = struct
  type t =
    | Scalar of float Expr.Value.t
    | Vector of {
        extent : Slot.extent Slot.t;
        var : Expr.Reduce_var.t;
        body : float Expr.Value.t;
      }
    | Scan of Expr.Scan.t

  let scalar value = Scalar value
  let vector ~extent ~var ~body = Vector { extent; var; body }
  let scan s = Scan s

  (* [width]/[steps] are proven bounded well below the 32-bit range by
     [Scan_limits]'s own hard ceilings at construction -- [Builder.scan] is
     the only way to build an [Expr.Scan.t], and it rejects [2*width] and
     [steps*width] past [Scan_limits.hard_max_state]/[hard_max_updates]
     (2^20 each) before this ever runs -- so this product needs no separate
     checked arithmetic here. *)
  let slot_count = function
    | Scalar _ -> Slot.one
    | Vector { extent; _ } -> Slot.count_of_extent extent
    | Scan s ->
        Slot.trace_count
          ~steps:(Slot.extent s.Expr.Scan.steps)
          ~width:(Slot.extent s.Expr.Scan.width)

  (* The one [float Expr.Value.t] a scalar/vector RHS carries; for a scan, a
     FOLDABLE stand-in built the same way a real trace read is -- wrapped as
     [Expr.Value.scan_at] so [Expr.Fold]/[Expr.Check] apply their existing
     per-child masking of [lane]/[step]/[prev] unchanged. [row]/[lane] here
     are [Expr.Index.zero], never evaluated, so this is not a materialized
     projection -- but it is therefore NOT for rendering an unspecialized
     scan to a reader: that must show [init]/[update] directly (see
     [Region_program.pp], [Region_trace.pp], [Me_detail]), never this
     wrapper, or the closed placeholder index would print as if it were a
     real read. *)
  let value = function
    | Scalar value -> value
    | Vector { body; _ } -> body
    | Scan s -> Expr.Value.scan_at s ~row:Expr.Index.zero ~lane:Expr.Index.zero
end

module Shape = struct
  type t =
    | Scalar
    | Vector of { extent : Slot.extent Slot.t }
    | Scan of { width : Slot.extent Slot.t; steps : Slot.extent Slot.t }

  let of_rhs = function
    | Rhs.Scalar _ -> Scalar
    | Rhs.Vector { extent; _ } -> Vector { extent }
    | Rhs.Scan s ->
        Scan
          {
            width = Slot.extent s.Expr.Scan.width;
            steps = Slot.extent s.Expr.Scan.steps;
          }

  let pp fmt = function
    | Scalar -> Fmt.string fmt "scalar"
    | Vector { extent } -> Fmt.pf fmt "vector[%a]" Slot.pp extent
    | Scan { width; steps } ->
        Fmt.pf fmt "scan[width=%a,steps=%a]" Slot.pp width Slot.pp steps
end

type t = { id : Expr.Local_var.t; rhs : Rhs.t }

let scalar ~id ~value = { id; rhs = Rhs.scalar value }
let scan ~id ~scan = { id; rhs = Rhs.scan scan }

(* [value]'s body may freely mention [var] (via [Expr.Index.reduce var]) as
   its own per-element index -- the binder [Rhs.vector] mints and stores
   alongside [extent], never bound by a [Value.Reduce] node inside [value]
   itself. Reading it back at a computed index ([Expr.Value.local_at]) is a
   beta-reduction of that binder, carried out by
   [Expr.Rewrite.substitute_locals]'s [Vector] case during specialization. *)
let vector ~id ~var ~extent ~value =
  { id; rhs = Rhs.vector ~extent ~var ~body:value }

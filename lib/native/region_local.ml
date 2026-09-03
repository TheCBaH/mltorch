module Shape = struct
  (* [Vector]'s own binder lives HERE, not as a fourth field on [t]: a scalar
     local has no binder at all, and a field that is meaningful only for one
     variant is the illegal state CLAUDE.md's ordering convention exists to
     rule out elsewhere in this codebase -- [Reduction.t] makes the same
     choice, keeping its own [var] a sibling of [kind]/[lo]/[hi]/[body]
     rather than optional. *)
  type t = Scalar | Vector of { extent : int; var : Expr.Reduce_var.t }

  let scalar = Scalar
  let vector ~extent ~var = Vector { extent; var }
  let slot_count = function Scalar -> 1 | Vector { extent; _ } -> extent

  let pp fmt = function
    | Scalar -> Fmt.string fmt "scalar"
    | Vector { extent; _ } -> Fmt.pf fmt "vector[%d]" extent
end

type t = { id : Expr.Local_var.t; shape : Shape.t; value : Expr.Value.t }

let scalar ~id ~value = { id; shape = Shape.scalar; value }

(* [value]'s body may freely mention [var] (via [Expr.Index.reduce var]) as
   its own per-element index -- the binder [Shape.vector] mints and stores
   alongside [extent], never bound by a [Value.Reduce] node inside [value]
   itself. Reading it back at a computed index ([Expr.Value.local_at]) is a
   beta-reduction of that binder, carried out by
   [Expr.Rewrite.substitute_locals]'s [Vector] case during specialization. *)
let vector ~id ~var ~extent ~value =
  { id; shape = Shape.vector ~extent ~var; value }

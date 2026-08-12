(* A claimed set of nodes plus the boundary it implies. The boundary is always
   DERIVED, never declared: a matcher claims nodes and the region works out which
   edges enter it, which leave, and which are private to it — so a pattern cannot
   describe its own boundary wrongly. See .ai/native_transform_design.md §11. *)

open Graph_ir

module Make (D : Dialect.S) : sig
  type node = D.op Graph_common.Node.t
  type graph = D.op Graph_common.Graph.t
  type error = [ `Empty_region | `Unknown_node of Node_id.t ]

  val pp_error : Format.formatter -> [< error ] -> unit

  type t = {
    nodes : Node_id.Set.t;
    inputs : Tensor_id.t list; (* operands whose producer is outside *)
    outputs : Tensor_id.t list;
        (* claimed edges used outside, or graph outputs *)
    interior : Tensor_id.t list; (* claimed edges used only inside *)
    convex : bool;
  }

  (* [convex] is REPORTED, not enforced. Non-convexity — a path that leaves the
     region and comes back — forces a cycle only when the replacement collapses the
     region to a single scheduling point, so the fusing recipe constructors require
     it while other rewrites need not care, and [Rewrite.apply]'s cycle check is the
     general backstop. *)
  val of_nodes : Graph_view.Make(D).t -> Node_id.Set.t -> (t, error) Err.t

  (* The matched portion as a standalone graph, so a test can print it with plain
     [Graph_ir.pp]. Boundary inputs become graph inputs, keeping the parent's
     [Input.kind]; the nodes land in one flat group. *)
  val extract : Graph_view.Make(D).t -> t -> graph
  val pp : Format.formatter -> t -> unit
end

include module type of Make (Native_dialect)

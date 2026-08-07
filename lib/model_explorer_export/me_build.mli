(* Project a native graph into a [Model_explorer.Graph.t].

   FUNCTORISED over the dialect side, because Native4D is projected through the
   same body with a different op type — the alternative is two projections that
   have to be kept saying the same thing, which is the drift this repository
   spends its `.ai/` docs preventing.

   Bounded rendering is the point of the [attr] machinery below: an attribute is
   written into a buffer with a hard cap and the PRINTER IS STOPPED AT THE CAP,
   never built at full size and truncated after. A megabyte attribute that is
   then cut to 8KB has already cost the megabyte.

   See .ai/model_explorer_design.md. *)

module type SIDE = sig
  type op

  val op_name : op -> string
  (** The node label, and the same string the op's serialisation uses as its
      case tag, so a projection that labels a node and a codec that names it
      cannot disagree. *)

  val operands : op -> Graph_ir.Tensor_id.t list
  val pp_op : Format.formatter -> op -> unit
end

type error =
  [ Me_ids.error
  | `Over_limit of string * int
  | `Unknown_producer of int  (** a tensor id no node and no input produces *)
  ]

val pp_error : Format.formatter -> [< error ] -> unit

val bounded : max:int -> (Format.formatter -> 'a -> unit) -> 'a -> string * bool
(** [bounded ~max pp v] renders [v] into at most [max] bytes and reports whether
    it was cut. The formatter's own output function refuses the write that would
    cross the cap and unwinds, so the printer stops there — this is what makes
    the bound a bound on WORK and not only on the result.

    Exposed because it is the one piece of this module a caller outside it
    needs, and because a bound nobody can test in isolation is a bound taken on
    trust. *)

module Make (S : SIDE) : sig
  val graph :
    limits:Me_limits.Limits.t ->
    id:string ->
    ?labels:(Graph_ir.Node_id.t -> string) ->
    S.op Graph_ir.Graph.t ->
    (Model_explorer.Graph.t, [> error ]) Core.result
  (** Every native node becomes one [GraphNode] with [Me_ids.op_node]'s id, plus
      one boundary node per graph input, constant and output — pinned ids, so a
      comparison can match them across versions like any other node.

      NAMESPACES COME FROM THE [Group] TREE, which is the authoritative
      structural hierarchy. Each level is a [Me_ids.namespace_component], so a
      label carrying a [/] cannot silently become two levels of hierarchy.

      [labels] overrides a node's displayed label — the hook the CLI's
      [--namespace module] mode and importer metadata use — and defaults to
      [S.op_name]. *)
end

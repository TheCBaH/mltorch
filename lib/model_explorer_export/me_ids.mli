(* Deterministic external identifiers.

   Model Explorer splits [namespace] on ['/'], honours [\/], and has special
   handling for [';']-separated variants, while group labels are arbitrary PT2
   strings. So every DYNAMIC component of an id — a group label, a model name —
   goes through one injective encoder, and the structural separators of an
   assembled id never do.

   The renderer sees the ENCODED form, and encoding can expand a component
   threefold, so the raw input is bounded by [max_label_bytes] and the result is
   bounded by [max_id_bytes] / [max_namespace_component_bytes] AFTER encoding.

   Ids here are stable across versions, which is what lets the comparison view
   rest on [MATCH_NODE_ID] for untouched nodes and explicit mapping entries only
   for changed ones. That is sound ONLY because of the id-identity rule — a
   changed value always means a new id (.ai/native_transform_design.md §4) — and
   the comparison view rests on it entirely.

   See .ai/model_explorer_design.md. *)

type error =
  [ `Control_byte of int  (** the offending byte *)
  | `Label_too_long of int  (** raw bytes offered *)
  | `Id_too_long of int  (** encoded bytes produced *) ]

val pp_error : Format.formatter -> [< error ] -> unit

(** {1 The one encoder} *)

val component : string -> (string, [> error ]) Core.result
(** Percent-encodes [%] and each of [/], [;], [\], [#], [:] — the characters
    the renderer's own grammar reads — in a single pass, which is injective
    because [%] is encoded alongside the rest rather than after them.

    Control bytes ([< 0x20], [0x7F]) are REJECTED, not passed through: they
    would survive percent-encoding into an id the renderer displays verbatim,
    and there is no correct rendering of them.

    A [result], where the plan's sketch had [string -> string]: a signature
    that cannot express its own rejection is not a contract. No length rule
    here — the caller knows which of the two ceilings the assembled id meets. *)

(** {1 The layer vocabulary}

    The DIALECT a flow state, transition or graph belongs to — the [<layer>] of
    every id below that carries one. Owned by the flow spine (see [Me_flow]),
    which is why the counters behind those ids are per-layer and dense: a
    [Pass.Exec_id.t] is dense per dialect, so [t/native/007] and
    [t/native4d/000] can carry equal executions and still not collide.

    NOT the same vocabulary as [Me_session.Capability.graph_stage], which names
    which exported GRAPH a capability is about (initial versus canonical Native
    are two stages of one layer, and fusion is a stage of the Kernel layer).
    They are deliberately separate: collapsing them would make [g/<layer>/<NNN>]
    unable to name the two Native graphs that share a layer. *)

module Layer : sig
  type t = Pt2 | Native | Native4d | Symbolic | Kernel

  val to_string : t -> string
  val of_string : string -> t option

  val all : t list
  (** Built by walking a successor chain, so a constructor cannot be added
      without reaching this list. *)
end

(** {1 Assembled ids}

    The structural forms. Each is total where it takes only structural input,
    and returns a [result] exactly where it takes a dynamic component or has a
    ceiling to meet. *)

val collection :
  limits:Me_limits.Limits.t -> string -> (string, [> error ]) Core.result
(** [mltorch:<component model_name>]. *)

val pt2_graph : Pt2_native_graph.Graph_path.t -> string
(** [pt2/<path>], where the root path renders as [pt2/root]. *)

val pt2_node : Pt2_native_graph.Graph_path.t -> int -> string
(** [<path>#<index>]. *)

val graph : Layer.t -> int -> string
(** [g/<layer>/<NNN>]. Owned by the flow spine rather than by any execution
    identity, so that two runs of the same pipeline produce the same ids. *)

val flow_state : Layer.t -> int -> string
(** [s/<layer>/<NNN>]. *)

val flow_transition : Layer.t -> int -> string
(** [t/<layer>/<NNN>]. *)

val op_node : Graph_ir.Node_id.t -> string
(** [n<k>] — stable across versions, which is what the comparison view rests on.
*)

val boundary : [ `In | `Const | `Out ] -> Graph_ir.Tensor_id.t -> string
(** [in:t<k>] / [const:t<k>] / [out:t<k>], pinned. *)

val namespace_component :
  limits:Me_limits.Limits.t ->
  ?label:string ->
  int ->
  (string, [> error ]) Core.result
(** [<component label>#g<id>], or [g<id>] when unlabelled. Bounded by
    [max_namespace_component_bytes] after encoding, since that is the figure the
    renderer's own splitting works over. *)

val detail :
  limits:Me_limits.Limits.t ->
  parent_graph:string ->
  parent_node:string ->
  int ->
  (string, [> error ]) Core.result
(** [expr/<parent_graph>/<parent_node>/t<k>]. The two parents are ids this
    module already produced, so they are not re-encoded — encoding an encoded id
    is not idempotent and would break the correspondence with the graph it
    names. They are length-checked as part of the whole. *)

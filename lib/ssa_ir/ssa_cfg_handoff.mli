(** What the native handoff takes from a graph, and no more.

    A value of the graph is either machine data, which a selector gives a
    register or a stack slot of the width its type names, or an effect, which
    orders operations and has no run-time representation. Precision is already
    explicit in every type and every operation (a binary32 value is a [f32]
    value, a fused operation is [float.fma]); an access names its layout, a six
    axis coordinate or a flat element offset, and its decode or encode names the
    stored format. Nothing here mentions a register, a spill or a frame: those
    are chosen downstream from this description.

    A block argument is a copy. An edge's arguments rebind the target's
    parameters simultaneously, so the copies of one edge are a parallel copy:
    {!sequentialize} orders them, with one temporary to break a cycle, for an
    allocator that cannot express a parallel move. *)

(** What a value is to a machine. *)
module Class : sig
  type t =
    | Data of Ssa_type.t  (** held in a register or slot of this type *)
    | Erased  (** an effect: ordering only, never materialized *)
end

val classify : Ssa_value.t -> Class.t

(** One move of a copy: [destination <- source]. *)
module Move : sig
  type t = { destination : Ssa_value.t; source : Ssa_value.t }
end

val copies : Ssa_cfg.t -> Ssa_cfg_edge.t -> Move.t list
(** The parallel copy of an edge: every parameter of the target that is machine
    data, with the argument that binds it. Effects are dropped, and so is a move
    of a value to itself. *)

val sequentialize :
  fresh:(Ssa_type.t -> Ssa_value.t) -> Move.t list -> Move.t list
(** An ordering of a parallel copy whose sequential execution leaves every
    destination holding the value its source held {e before} the copy: a move is
    emitted only once nothing still to be copied reads its destination, and a
    cycle is broken by saving one source in a value from [fresh]. Destinations
    are distinct. *)

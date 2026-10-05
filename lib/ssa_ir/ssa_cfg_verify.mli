(** Structural verification of a control-flow graph: every block reachable, one
    definition per value, every use dominated by its definition, edge arguments
    matching the target's parameters, branch conditions predicates, and the
    effect a single chain through the graph. It checks structure, never the
    equivalence of the graph with the structured program it came from, which the
    differential interpretation does.

    Operations that carry a proof ([index.add_in_domain], [load.in_bounds], ...)
    are accepted as claims: the structured program they were lowered from
    verified them, and the lowering re-checks the ones it adds. A graph that is
    transformed after lowering must re-establish its own proofs. *)

(** The effect rule is the structured one made explicit: a block with an effect
    parameter starts its chain there; one without must have a single predecessor
    and continues that predecessor's chain. An edge passes the chain on as the
    argument of the target's effect parameter, and [Return] consumes it. *)
type problem =
  | Block_defined_twice
  | Branch_condition of Ssa_type.t
  | Buffer_format of {
      buffer : Ssa_id.Buffer.t;
      accessed : Ssa_format.Family.t;
      declared : Ssa_format.t;
    }
  | Buffer_not_stored of Ssa_id.Buffer.t
  | Buffer_unknown of Ssa_id.Buffer.t
  | Definition_twice of Ssa_id.Value.t
  | Edge_arguments of { expected : Ssa_type.t list; found : Ssa_type.t list }
      (** The arguments of an edge against the target's parameters. *)
  | Effect_missing
      (** An effectful operation without the effect it consumes, or a pure one
          with one. *)
  | Effect_parameters of Ssa_type.t list
      (** A block with more than one effect parameter, or with none and not
          exactly one predecessor. *)
  | Effect_stale of { used : Ssa_value.t; live : Ssa_value.t }
      (** The operand is not the current effect: a fork, a reuse, or a skipped
          link of the chain. *)
  | Entry_has_predecessor
  | Entry_missing
  | Flat_on_per_channel of Ssa_id.Buffer.t
  | Result_types of { declared : Ssa_type.t list; expected : Ssa_type.t list }
  | Target_unknown of Ssa_id.Block.t
  | Typing of Ssa_typing.error
  | Unreachable
  | Use_not_dominated of Ssa_id.Value.t
      (** Defined, but not on every path to the use. *)
  | Use_retyped of {
      value : Ssa_id.Value.t;
      defined : Ssa_type.t;
      used : Ssa_type.t;
    }
  | Use_undefined of Ssa_id.Value.t

type diagnostic = { block : Ssa_id.Block.t; problem : problem }
type error = [ `Invalid_cfg of diagnostic ]

val pp_problem : Format.formatter -> problem -> unit
val pp_diagnostic : Format.formatter -> diagnostic -> unit
val pp_error : Format.formatter -> [< error ] -> unit
val check : Ssa_cfg.t -> (unit, error) Err.t

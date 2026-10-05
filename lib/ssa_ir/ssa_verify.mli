(** Structural verification of a program revision: definitions, types, scope,
    region signatures, buffer permissions and path-sensitive effect flow. It
    checks structure and the proof prerequisites an operation names, never the
    equivalence of two programs: a transformation is checked by differential
    execution against the reference interpreter.

    The first problem found is reported, with the exact region and statement. *)

module Statement : Core.Tagged_int.S
(** The statement of a region a problem was found in. The ordinal counts
    statements of that region from zero. *)

type site = { region : Ssa_id.Region.t; statement : Statement.t }

type problem =
  | Buffer_declaration of Ssa_id.Buffer.t
      (** Duplicate id, a non-positive extent, more elements than an index
          holds, or per-channel parameters for a different number of channels
          than the C extent. *)
  | Buffer_format of {
      buffer : Ssa_id.Buffer.t;
      accessed : Ssa_format.Family.t;
      declared : Ssa_format.t;
    }
  | Buffer_not_stored of Ssa_id.Buffer.t
      (** A store to an input: only an output or scratch buffer is written. *)
  | Buffer_unknown of Ssa_id.Buffer.t
  | Definition_twice of Ssa_id.Value.t
  | Effect_missing
      (** An effectful operation without the effect it consumes, or a pure one
          with one. *)
  | Effect_stale of { used : Ssa_value.t; live : Ssa_value.t }
      (** The operand is not the current effect: a fork, a reuse, or a skipped
          link of the chain. *)
  | Effect_unique of Ssa_type.t list
      (** A region signature that does not carry exactly one effect. *)
  | Flat_on_per_channel of Ssa_id.Buffer.t
      (** A flat element offset on a per-channel quantized buffer: it cannot
          name the channel the decode needs. *)
  | Region_defined_twice of Ssa_id.Region.t
  | Region_too_deep
  | Result_types of { declared : Ssa_type.t list; expected : Ssa_type.t list }
  | Signature of { expected : Ssa_type.t list; found : Ssa_type.t list }
      (** A region's parameters, yields or a statement's results against the
          signature its operands imply. *)
  | Step_not_positive of int64
  | Typing of Ssa_typing.error
  | Use_retyped of {
      value : Ssa_id.Value.t;
      defined : Ssa_type.t;
      used : Ssa_type.t;
    }
  | Use_undefined of Ssa_id.Value.t
      (** Not in scope at the use: never defined, defined in a region that does
          not enclose it, or defined after it. *)

type diagnostic = { site : site; problem : problem }
type error = [ `Invalid_program of diagnostic ]

val pp_problem : Format.formatter -> problem -> unit
val pp_diagnostic : Format.formatter -> diagnostic -> unit
val pp_error : Format.formatter -> [< error ] -> unit

val max_region_depth : int
(** The region nesting a program may reach. Execution recurses once per level,
    so this is the bound that keeps it inside a JavaScript stack. *)

val check : Ssa_program.t -> (unit, error) Err.t

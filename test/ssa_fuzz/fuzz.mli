(** A seeded generator of well-typed SSA programs (the scalar core, structured
    control, scratch locals and the scan meter), and the differential it drives.
    A case is [(seed, size)]: the same pair regenerates the same program and
    input on every run. The structured interpreter is the oracle; each backend
    is compared with it, never with another backend. *)

type profile = { fma : bool; i64_to_f32 : bool }
(** The operations a backend may have no form for. A typed refusal of one is a
    capability gap, not a disagreement; a backend that cannot express one is
    driven with a profile that leaves it out. *)

val full : profile
(** Every operation. *)

type case = {
  seed : int64;
  size : int;
  program : Ssa_ir.Ssa_program.t;
  input : float array;
}

val default_size : int

val seeds : int
(** The seeds through the interpreters: 2,000, or [MLTORCH_FUZZ_SEEDS]. The
    emitter drivers run a fraction of it. *)

val generate : ?profile:profile -> seed:int64 -> size:int -> unit -> case
(** A program that verifies by construction: a builder or verifier error is a
    defect of the generator and raises [Failure] naming the seed. *)

type outcome = { row : Kernel_eval.error option; cells : float list }
(** What a run reports: the failure row, or the output cells (empty after a
    failure, whose partial stores no backend promises). *)

val equal_outcome : outcome -> outcome -> bool
(** Rows structurally, cells with {!Core.Float_bits.equal_portable}. *)

val pp_outcome : Format.formatter -> outcome -> unit

type run = Ran of outcome | Refused of string
type backend = { name : string; run : case -> run }

val interp : backend
(** The oracle: {!Ssa_ir.Ssa_interp.run}. *)

val cfg : backend
(** {!Ssa_ir.Ssa_cfg_lower.program} then {!Ssa_ir.Ssa_cfg_interp.run}. *)

val emitted :
  case ->
  Ssa_ir.Ssa_buffer.t list ->
  Loop_ir.Loop_program.t * (Tensor_id.t -> Tensor.packed option)
(** What an emitted kernel runs over, for the emitter's [arguments]: the Loop
    program that supplies its argument layout, and the input binding. The
    emitter backends (C, JavaScript, Wasm) live in their own libraries because
    they link different executors; they share this and the two below. *)

val of_row : Loop_ir.Loop_interp.error -> run
(** A failure the emitted kernel reported. *)

val of_outputs : Tensor.packed Tensor_id.Map.t -> run
(** The output cells of a kernel that ran to the end. *)

type disagreement = {
  case : case;
  backend : string;
  expected : outcome;
  got : outcome;
}

val check : case -> backend list -> (outcome, disagreement) result
(** The interpreter's outcome, or the first backend that differs from it. A
    refused backend agrees with anything. *)

val minimize :
  ?profile:profile ->
  seed:int64 ->
  size:int ->
  backend list ->
  disagreement option
(** The disagreement at the smallest size up to [size] for [seed], if any: the
    same seed regenerated smaller, not a shrunk term. *)

val pp_disagreement : Format.formatter -> disagreement -> unit
(** The seed, the size, the program and both outcomes. *)

val replay : ?profile:profile -> seed:int64 -> size:int -> backend list -> unit
(** Prints the case and every backend's outcome, for the toplevel. *)

val row_name : Kernel_eval.error -> string
(** A failure row's kind, for counting which rows a seed range reaches. *)

(** Verified CFG SSA to generic Machine IR, for the admitted slice. Every
    computational width, signedness and conversion becomes explicit; coordinates
    become validated row-major byte offsets from a view's address; each
    source-failing guard ends its generic block and branches to a [fail] with
    the original kind and payload, so a guarded operation's effect runs only
    after all its guards, in the original order; storage decodes and encodes are
    separated from raw access. Anything outside the slice is a typed refusal: no
    partial program is ever returned. *)

open Machine_ir

module Refusal : sig
  type t =
    | Buffer_layout of Ssa_ir.Ssa_id.Buffer.t
        (** a buffer whose byte size is not representable *)
    | Cfg of Ssa_ir.Ssa_cfg_lower.error
    | Invalid_cfg of Ssa_ir.Ssa_cfg_verify.diagnostic
    | Invalid_lowering of Mir_diagnostic.t
        (** the lowering produced a program its verifier rejects: a compiler
            defect *)
    | Invalid_program of Ssa_ir.Ssa_verify.diagnostic
    | Operation of { op : string; slice : Mir_census.Slice.t }
        (** an SSA operation (by its stable name) admitted only by a later slice
        *)
    | Planning of Mir_planning.mismatch
    | Precision of Mir_planning.Precision.t
        (** binary32 arithmetic under a binary64 summary *)
    | Type of { ty : Ssa_ir.Ssa_type.t; slice : Mir_census.Slice.t }

  val pp : Format.formatter -> t -> unit
end

(** Fault injection for the evidence suite: each is one deliberate lowering
    defect the differential harness must detect. No consumer passes one. *)
module Mutation : sig
  type t =
    | Conversion_order
        (** a float-to-i64 range guard before its NaN and infinity guards *)
    | Double_rounding  (** i64 to binary32 through binary64 *)
    | Eager_load  (** a checked load's read moved above its guards *)
    | Guard_order  (** axis guards in reverse order *)
    | Operand_order  (** a float subtraction's or division's operands swapped *)
    | Scale_bytes  (** element offsets scaled by twice the element bytes *)
    | Sequential_transfer
        (** an edge argument that names an earlier rebound parameter reads its
            new value *)
    | Zero_extend  (** index widening by zero- instead of sign-extension *)
end

type result = {
  program : Mir_verify.Generic.t;
  layout : Mir_layout_map.Entry.t list;
  planning : Mir_planning.t;
}

val subject : Ssa_ir.Ssa_program.t -> string
(** The digest a planning summary for this program names: of its canonical
    printed text. *)

val summary :
  ?target:Ssa_ir.Ssa_target.t ->
  ?fma:Mir_planning.Fma.t ->
  Ssa_ir.Ssa_plan.t ->
  Mir_planning.t
(** The resolved summary of a plan and the target it was planned against (none
    for a scalar plan). The FMA mode is read from the target's permissions under
    the plan's policy unless [fma] overrides it — for a test that needs a
    deliberately wrong summary. *)

val program :
  ?mutation:Mutation.t ->
  planning:Mir_planning.t option ->
  Ssa_ir.Ssa_program.t ->
  (result, Refusal.t) Err.t
(** Verifies the structured program, admits the summary against its subject,
    lowers it to a CFG and the CFG to generic Machine IR, and verifies that. *)

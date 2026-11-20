(** What the lowering inspector shows for one {!Lowering_key.t}: the text a
    backend family makes, or the typed reason it makes none, and the facts the
    text does not show. A refusal is a value of the producer's own typed reason,
    never prose; rendering is {!Refusal.pp}'s, at the edge. *)

open Graph_ir

module Language : sig
  (* @alphabetical *)
  type t = C | Js | Loop | Ssa | Wat

  val all : t list
  val name : t -> string
end

module Refusal : sig
  (* @alphabetical *)
  type t =
    | Granted of Node_id.t
        (** a node no Loop body realizes: the bundle holds a placeholder its
            realizing helper replaces, so no family has code to show *)
    | Loop_c of Loop_ir.Loop_c.error
    | Loop_lowering of Loop_ir.Loop_lower.error
        (** re-lowering the placed kernel under a pass subset *)
    | Loop_wasm of Loop_ir.Loop_wasm.error
    | Ssa_backends of string
        (** [Ssa_backends]' own words: its producers still return strings for
            everything after lowering (planning, emission) *)
    | Ssa_lowering of Ssa_backends.Refusal.t
        (** the placed kernel has no SSA program: the converter or the lowering
            refused *)
    | Unknown_node of Node_id.t  (** the bundle has no invocation for it *)
    | Unsupported_family of Lowering_key.family
    | Unsupported_policy of {
        backend : Loop_ir.Loop_numerics.Backend.t;
        policy : Loop_ir.Loop_numerics.t;
      }

  val pp : Format.formatter -> t -> unit
end

module Note : sig
  (* @alphabetical *)
  type t =
    | Fp32_refusal of Loop_ir.Loop_numerics.Refusal.t
        (** why an fp32 policy left the kernel binary64 *)
    | Fused_multiply_adds  (** the plan holds fused multiply-adds *)
    | Helpers of Loop_ir.Loop_c_runtime.Name.t list
        (** the runtime helpers the C text calls *)
    | Local_doubles of int64  (** [double]s of scratch the C kernel needs *)
    | Loop_target of Loop_ir.Loop_target.t
        (** the vector target the Loop emitter planned for *)
    | Numerics of Loop_ir.Loop_numerics.t
        (** the policy applied: the family's default when the key named none *)
    | Numerics_not_applicable of Loop_ir.Loop_numerics.t
        (** the family has no numerical policy; the key's was not used *)
    | Row_blocking  (** the plan blocked row loops around a vector loop *)
    | Ssa_fp32_refusal of Ssa_ir.Ssa_numerics.Refusal.t
        (** why an fp32 policy left the SSA kernel binary64 *)
    | Ssa_target of Ssa_ir.Ssa_target.t  (** the target the SSA plan is for *)
    | Vectorization of Ssa_ir.Ssa_vectorize.report
        (** every loop the SSA planner tried, with its decision *)
    | Working_precision of Loop_ir.Loop_numerics.Precision.t

  val pp : Format.formatter -> t -> unit
end

type body =
  | Refused of Refusal.t
  | Text of { language : Language.t; text : string; truncated : bool }

type t = { key : Lowering_key.t; body : body; notes : Note.t list }

val pp : Format.formatter -> t -> unit
(** The text (or the refusal) followed by the notes, for the CLI. *)

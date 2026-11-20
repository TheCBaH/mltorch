(** What the lowering inspector shows for one {!Lowering_key.t}: the text a
    backend family makes, or the typed reason it makes none, and the facts the
    text does not show. A refusal is a value of the producer's own typed reason,
    never prose; rendering is {!Refusal.pp}'s, at the edge. *)

open Graph_ir

module Language : sig
  (* @alphabetical *)
  type t =
    | C
    | Coral_estimate
        (** the census of the final SSA program and its estimated cycles *)
    | Coral_plan  (** a transfer plan, as {!Coral_transfer.Transfer_plan.pp} *)
    | Gnu_asm of Lowering_key.Target.t
        (** the typed Rivet module of the target's published artifact, printed
            as GNU assembly *)
    | Js
    | Loop
    | Machine_ir
    | Ssa
    | Wat

  val all : t list
  val name : t -> string
end

module Refusal : sig
  (* @alphabetical *)
  type t =
    | Asm_aarch64 of Machine_rivet_aarch64_module.Rivet_a64_refusal.t
        (** the AArch64 artifact has no Rivet module *)
    | Asm_riscv32 of Machine_rivet_riscv32_mir_module.Rv32_mir_form.Refusal.t
        (** the RV32 artifact (grouped or not) has no Rivet module *)
    | Asm_x86_64 of Machine_rivet_x86_64_module.Rivet_x64_refusal.t
        (** the x86-64 artifact has no Rivet module *)
    | Coral_kernel of Coralnpu_kernels.Kernel_refusal.t
        (** the contraction has no generated kernel on that engine *)
    | Coral_transfer of Coral_transfer.Transfer_plan.Refusal.t
        (** the chosen transfer plan does not check within the DTCM budget *)
    | Granted of Node_id.t
        (** a node no Loop body realizes: the bundle holds a placeholder its
            realizing helper replaces, so no family has code to show *)
    | Loop_c of Loop_ir.Loop_c.error
    | Loop_lowering of Loop_ir.Loop_lower.error
        (** re-lowering the placed kernel under a pass subset *)
    | Loop_wasm of Loop_ir.Loop_wasm.error
    | Mir_lowering of Machine_model.Mir_model.Refusal.t
        (** the invocation has no Machine IR kernel *)
    | Mir_stage of Machine_model.Mir_model.Reason.t
        (** the target's stage refused: a selection it does not admit, a frame
            that does not fit, a checker rejection, register groups asked of the
            reference allocator *)
    | Not_a_contraction
        (** the invocation's plan holds no matrix product to generate a kernel
            for *)
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
    | Code_bytes of int64
        (** the size of the text Rivet lays the module out as *)
    | Coral_region_assumed of Coral_estimate.Coral_census.Region.t
        (** the memory region every buffer was priced in: the estimate has no
            placement of its own *)
    | Coral_transfer_budget of Coral_transfer.Transfer_units.Bytes.t
        (** the DTCM the transfer plan was chosen within *)
    | Fp32_refusal of Loop_ir.Loop_numerics.Refusal.t
        (** why an fp32 policy left the kernel binary64 *)
    | Fused_multiply_adds  (** the plan holds fused multiply-adds *)
    | Helper_bindings of (string * string) list
        (** each helper the artifact calls, with the C library symbol it links
        *)
    | Helpers of Loop_ir.Loop_c_runtime.Name.t list
        (** the runtime helpers the C text calls *)
    | Local_doubles of int64  (** [double]s of scratch the C kernel needs *)
    | Loop_target of Loop_ir.Loop_target.t
        (** the vector target the Loop emitter planned for *)
    | Mir_summary of string
        (** the realized or scanned artifact's summary, in Machine IR's own
            words *)
    | Numerics of Loop_ir.Loop_numerics.t
        (** the policy applied: the family's default when the key named none *)
    | Numerics_not_applicable of Loop_ir.Loop_numerics.t
        (** the family has no numerical policy; the key's was not used *)
    | Relocations of Machine_model.Mir_artifact.Relocation.t list
        (** every relocation an instruction of the artifact encodes *)
    | Row_blocking  (** the plan blocked row loops around a vector loop *)
    | Ssa_fp32_refusal of Ssa_ir.Ssa_numerics.Refusal.t
        (** why an fp32 policy left the SSA kernel binary64 *)
    | Ssa_target of Ssa_ir.Ssa_target.t  (** the target the SSA plan is for *)
    | Symbols of Machine_model.Mir_artifact.Symbol.t list
        (** every data and function symbol of the artifact *)
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

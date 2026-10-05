(** Kernel producers for the whole-model bundle builders, from structured SSA.

    Each takes the bundle's invocation, lowers its placed kernel
    ({!Loop_ir.Loop_bundle.invocation}[.placed]) to SSA, runs the chosen
    {!Pipeline}, and emits it with the invocation's own buffer list as the
    argument convention and its own failure-site table to decode records
    against, so the hosts and the artifact ABI are the Loop path's. A kernel the
    SSA path cannot make is [Error] with its reason: no fallback. *)

(** What runs between lowering and emission, so representation, equivalent
    passes and added passes are measured separately. *)
module Pipeline : sig
  type t =
    | Exact
        (** the exact optimizations only: simplify, guards, hoisting and load
            sharing, with no alias assumption (the storage plan may overlay
            buffers) *)
    | Planned of {
        numerics : Ssa_ir.Ssa_numerics.t;
        target : Ssa_ir.Ssa_target.t;
      }
        (** the policy planner: vectorization, scheduled sums and contraction
            where the policy permits them, then the exact passes *)
    | Representation  (** the lowered program, untouched *)

  val name : t -> string
end

val program :
  Pipeline.t ->
  Loop_ir.Loop_bundle.invocation ->
  (Ssa_ir.Ssa_program.t, string) result
(** The program an invocation is emitted from. *)

val c :
  pipeline:Pipeline.t ->
  name:string ->
  Loop_ir.Loop_bundle.invocation ->
  (Loop_ir.Loop_c.t, string) result
(** For {!Loop_ir.Loop_bundle_c.build}'s [kernel]. *)

val wasm :
  pipeline:Pipeline.t ->
  table_alloc:(bytes:int -> int) ->
  Loop_ir.Loop_bundle.invocation ->
  (Loop_ir.Loop_wasm.kernel, string) result
(** For {!Loop_ir.Loop_bundle_wasm.build}'s [kernel]. A [Planned] pipeline for a
    relaxed-SIMD target makes the fused vector operation. *)

val js :
  pipeline:Pipeline.t ->
  Loop_ir.Loop_bundle.invocation ->
  (Js_ast.Program.t, string) result
(** For {!Loop_ir.Loop_bundle_js.build}'s [kernel]. *)

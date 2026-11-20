(** The producers behind the lowering inspector: one bundle invocation and one
    {!Lowering_key.variant} to the text the whole-model backend would run for
    that node, or a typed refusal. Total: every family either emits or refuses.

    Defaults are what the whole-model drivers run (the key's missing segments):
    the Loop emitter for C and WebAssembly under [simd_fp32_relaxed] with a
    128-bit vector target (NEON for C, [wasm128] for WebAssembly); JavaScript
    has no numerical policy, so binary64 and scalar; the structured SSA route
    plans for the same policy and target as the family's Loop route. A
    [reference_f64] policy is scalar everywhere. *)

module Artifact = Lowering_artifact

val kernel_name : string
(** The function name C text is made under: the text begins
    [static int kernel(]. A whole-model unit renames it per distinct kernel. *)

val of_invocation :
  ?dialect:Lowering_key.Dialect.t ->
  granted:(Graph_ir.Node_id.t -> bool) ->
  Loop_ir.Loop_bundle.invocation ->
  Lowering_key.variant ->
  Artifact.t
(** [granted] names the nodes the bundle was built granting (a bundle does not
    record it, and its graph has no operations to ask): a granted node's
    placeholder body is shown by no family, each answers [Granted]. A family
    outside {!Lowering_key.family}'s supported set ([Asm], [Coral]) answers [Unsupported_family]. *)

val of_node :
  ?dialect:Lowering_key.Dialect.t ->
  granted:(Graph_ir.Node_id.t -> bool) ->
  Loop_ir.Loop_bundle.t ->
  Graph_ir.Node_id.t ->
  Lowering_key.variant ->
  Artifact.t
(** {!of_invocation} for the bundle's invocation of the node; [Unknown_node]
    when it has none. *)

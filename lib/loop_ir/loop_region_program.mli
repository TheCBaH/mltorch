(* Wraps a single already-built [Region_program.t] -- the SAME program a
   [region_result]-style caller passed to [Region_execution.lower_region] --
   in a minimal, single-value [Kernel.t] so it can run through [Loop_lower]
   without a second construction from the originating [Op.t]. *)

type error =
  [ `Kernel of Kernel.error
  | `Lower of Loop_lower.error
  | `Unresolved_source of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val lower :
  limits:Kernel.Limits.t ->
  out_shape:Vec6.shape ->
  bindings:Tensor.packed Tensor_id.Map.t ->
  Region_program.t ->
  (Loop_program.t, error) Err.t
(** [bindings] is [Region_executor.t]'s own [~bindings]: every source [program]
    reads, already resolved to a real [Tensor.packed] -- a real operand and a
    synthetic default (e.g. an omitted [Sdpa] mask) are indistinguishable here,
    both just a bound tensor, since [Kernel.Input.t]'s [sg] is derived from the
    tensor itself ([Tensor_sig.shape]/[fmt], not a graph lookup). A source
    [bindings] does not cover is [`Unresolved_source]. The fresh
    [Kernel.Value.t] this mints is a boundary output, [Round_f32]-converted
    (every Region-authored op's stored result is F32) -- Bool-valued Region ops
    don't exist today ([Kernel.Result_conversion.t]'s other case,
    [Nonzero_bool], is a Pixel-stage boundary conversion, never a Region one).
*)

val lower_group :
  limits:Kernel.Limits.t ->
  bindings:Tensor.packed Tensor_id.Map.t ->
  selected:Region_group.Ordinal.t list ->
  Region_group.t ->
  (Loop_program.t * (Region_group.Ordinal.t * Tensor_id.t) list, error) Err.t
(** The group sibling of [lower] (T7.2): several sibling values sharing one
    [Region_group.t] (today only Lstm) rather than one standalone
    [Region_program.t]. No [~out_shape]: each [selected] ordinal's own
    [Region_group.Emitter.t] already carries its own [output_shape] -- a group
    projects several differently-shaped outputs off one shared recurrence
    (Lstm's output/h_n/c_n), so the shape lives per-ordinal in the group itself.
    An ordinal outside [group]'s own range is a caller defect, matching
    [Region_execution.materialize_group]'s own convention.

    Unlike [lower]'s single, discardable [value_id] (a solo caller destructures
    [Loop_js_exec]'s one-entry result map without caring what its key is), a
    group caller needs to know which output buffer answers for which [Ordinal.t]
    -- the second component is that mapping, in [selected]'s order, over the ids
    [lower_group] itself minted (fresh, disjoint from every source id, and from
    each other by construction). *)

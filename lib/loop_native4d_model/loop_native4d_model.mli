(** An archive as a Loop bundle plus the reference that checks it, in either
    dialect. *)

module Dialect : sig
  type t = Native | Native4d

  val of_name : string -> t option
  val name : t -> string
  val all : t list
end

module Native4d_failure : sig
  type t =
    | Bundle of Native4d.Eval_direct4.error Loop_ir.Loop_bundle.generic_error
    | Constants of Const_ssa_materialize.error
    | Convert of Native4d.Error.t
    | Reference of Native4d.Eval_direct4.error
    | Snapshot of Graph_view.error

  val pp : Format.formatter -> t -> unit
end

type error =
  [ Native_interp.error
  | Loop_ir.Loop_bundle.error
  | `Native4d of Native4d_failure.t ]

val pp_error : Format.formatter -> [< error ] -> unit

type t = {
  bundle : Loop_ir.Loop_bundle.t;
  native_graph : Graph_ir.graph option;
      (** the graph the bundle was built from, for the Native dialect only: a
          driver that runs Native per-node executors over it (loop_js_pt2's
          [--baseline]) has nothing to run on a Native4D model *)
  constants : Tensor.packed Tensor_id.Map.t;
      (** every payload the bundle binds, deferred Const-SSA exports included *)
  reference :
    (Tensor_id.t * Tensor.packed) list ->
    (Tensor.packed Tensor_id.Map.t, error) Err.t;
      (** the per-edge evaluator of the graph the bundle was built from, over
          the same [constants]: [Eval_direct.run] or [Eval_direct4.run] *)
}

val load :
  Dialect.t ->
  config:Storage_script.Config.t ->
  Pt2_archive.t ->
  (t, error) Err.t
(** Native: the lowered archive. Native4D: the canonical passes, then the
    four-axis lowering, whose deferred constants are materialized from the
    archive and merged over the materialized ones with the destination store's
    precedence, as {!Native4d.Lower.evaluate} does. *)

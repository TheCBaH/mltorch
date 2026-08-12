(* Fusion, as an overlay and a decision list over an UNCHANGED kernel.

   Never a fabricated rewrite. [Fusion_plan] changes where a value is computed,
   not what it means, and the kernel it plans over is carried inside the plan
   precisely so a plan cannot be applied to a different one. A "fused kernel
   graph" would be a graph nothing produced, and a reader comparing it against
   the kernel would be comparing a rendering against the thing it renders.

   PLACEMENT IS TWO FACTS, NOT ONE. Which dependency EDGES are virtual, and
   which VALUES need stores. An externally live producer is both — virtual for
   its consumer and still stored for whoever else needs it — and a single
   materialized/virtual per value cannot say that. So the overlay carries the
   edges and the node data carries the values, and neither is derivable from the
   other.

   See .ai/model_explorer_design.md. *)

type error = Me_limits.over_limit_error
(** Counting only, under [Me_limits.Scope.Fusion]. *)

val pp_error : Format.formatter -> [< error ] -> unit

type t = {
  overlays : Model_explorer.EdgeOverlaysData.t;
      (** the virtual dependency edges *)
  node_data : Me_session.Node_data_set.t;
      (** per value: stored, virtual, or both *)
  rejections : Me_limits.Diagnostic.t list;
      (** ONE summary: how many edges the plan made virtual and how many
          producers stayed materialized.

          Not one diagnostic per rejection. A rejection is a fact about a VALUE,
          so it belongs on that value's own datum — and [max_diagnostics] is 64
          by design while resnet18 alone produces seventy unfused producers, so
          the per-rejection form is a list the ceiling has to truncate rather
          than a report. The reason for each is in {!node_data}'s label. *)
}

val of_kernel :
  limits:Me_limits.Limits.t -> graph:string -> Kernel.t -> (t, [> error ]) Err.t
(** [graph] is the KERNEL graph's id, which is also the fusion view's: the plan
    is a view over that graph rather than a graph of its own, so a second id
    would name something that does not exist.

    A rejection's [Multiple_uses] count is rendered [">= 2"], never a figure.
    The planner's counter saturates at two because the cross-body total is an
    [int] and the per-value limits admit a mathematical aggregate past 2^31 —
    which wraps negative under js_of_ocaml and would read as unique-use.
    Legality only asks one versus more than one, so printing a count would claim
    precision the value does not have. *)

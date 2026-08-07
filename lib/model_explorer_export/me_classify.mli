(* Two exhaustive classifiers, and the one distinction they exist to draw:
   between a graph a partial conversion does not CLAIM to handle, and a defect.

   A partiality becomes a capability the session reports as unavailable with a
   reason the user can act on. A defect becomes a session error and is never
   downgraded — reporting an internal invariant failure as "this model is
   outside the dialect" tells the user to change their model to work around our
   bug.

   Both are total matches over the upstream error rows, so a row added there
   stops this compiling rather than falling into a default arm that would guess.
   That is the whole reason these are classifiers rather than wildcards at the
   call site.

   See .ai/model_explorer_design.md. *)

type verdict =
  | Unavailable of Me_session.Capability.reason
      (** recoverable: a capability the session reports as unavailable *)
  | Fatal  (** a defect: the session fails rather than reporting a capability *)

val pp_verdict : Format.formatter -> verdict -> unit

val native4d : [< Native4d.Error.t ] -> verdict
(** [`Missing_constant_payload] is [Requires_payloads]; the ten domain rows are
    [Outside_dialect_domain]; [`Bad_constant_payload], [`Map] and [`View] are
    fatal — a payload that was supplied and is wrong, or a map/view invariant
    failure, is a defect rather than a partiality.

    Having payloads removes exactly ONE failure mode. It does not put a graph
    inside the dialect, which is why Native4D is conditional for a [.pt2] and a
    standalone [model.json] alike. *)

val kernel : [< Kernel_adapt.error ] -> verdict
(** [`Passthrough_output] is [Unsupported_graph_shape] and RECOVERABLE — a graph
    input used directly as a graph output has no stage to name, and the kernel
    suite builds one its own comment calls legal.

    [Kernel]'s limit rows are [Over_limit]. Its structural rows,
    [`Program_invalid], [`Unknown_stage_source], [`Missing_live_output] and
    [`Unknown_program_output] are fatal: the stage program is
    repository-generated, so those indicate an internal defect.
    [`Output_not_selected] and [`Unknown_selection] are fatal because they are
    reachable only through [?select], which whole-program export does not use.
*)

val requires_payloads_without_them : Me_session.Capability.reason
(** What a payload-needing feature reports when the input carried none —
    [Requires_payloads]. Named rather than written at each call site, because
    [--fold] on a JSON model is a SUCCESSFUL structural session carrying this,
    plus a note on stderr, and not a usage error: the browser cannot surface one
    and the same code path serves both shells. *)

(** {1 Prerequisite propagation} *)

val propagate :
  lowering_available:bool ->
  Me_session.Capability.t list ->
  Me_session.Capability.t list
(** When lowering is unavailable there is no Native graph, so every key that
    depends on one becomes [Unavailable Prerequisite_unavailable] — EXCEPT where
    it is already [Not_requested], which takes precedence, since a feature
    nobody asked for was not blocked by anything.

    [Feature Flow] is included: with no Native state the spine holds only
    [s/pt2/000] and no transitions, and a one-node flow graph asserts a
    navigability that does not exist.

    [Source], [Feature Loop_ir] and [Feature Codegen] are unaffected. *)

val depends_on_lowering : Me_session.Capability.key -> bool
(** The set {!propagate} rewrites, exposed so a fixture can enumerate it rather
    than restate it — a complete capability vector is what the unsupported-model
    golden pins, since a test checking only the failing key would let every
    downstream row drift. *)

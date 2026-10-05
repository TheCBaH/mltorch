(* The static schedule, invocation bindings and storage plan a whole-model JS
   bundle compiles from (plan W2, design §3 "Preparation and scheduling").
   [build] needs no tensor payload -- like [Loop_node_program.lower] itself --
   so a bundle can be prepared before any weight is loaded.

   Schedule/admission/output-selection/arena-role is not re-derived here:
   [build] reads it straight from [Eval_direct.storage_script]'s own event
   stream (under [Retain.Only empty], the plan's initial retention, and a
   caller-chosen [Storage_script.Config.t]), so a bundle's invocation order
   and role/arena classification are PROVABLY the same decision
   [Eval_direct.run_storage] would follow, not a second implementation that
   could drift from it. The witnessed [Storage_plan.t] comes from the same
   script, via the existing [Storage_plan.create].

   A Region-authored node is lowered from signatures alone
   ([Loop_region_program.lower_sigs]); a grouped one (Lstm) is one invocation
   writing every scheduled output from a single shared recurrence. *)

open Graph_ir

type synthetic = { id : Tensor_id.t; value : float; shape : Vec6.shape }
(** A Region-authored node's optional operand that the graph does not supply (an
    omitted Sdpa mask, say), bound to a constant-filled F32 tensor of [shape].
*)

module Output : sig
  type t = {
    ordinal : Output_ordinal.t;
    oid : Tensor_id.t;  (** the graph edge this output binds *)
    role : Storage_script.Role.t;
    arena : Storage_script.Arena_id.t option;
        (** [None]: outside every arena (a fresh allocation at run time) *)
  }
end

type invocation = {
  node : Node_id.t;
  outputs : Output.t list;
      (** the node's scheduled outputs this program writes, in the order its
          [Output] buffers appear. One for every node but a grouped Region node,
          whose scheduled outputs share one recurrence. *)
  placed : Fusion_plan.t;
      (** the placed kernel [program] was lowered from, for a consumer that
          lowers plans itself: its buffers are [program]'s (a source or an
          output; an unused input may be declared and is never an argument), and
          [edges] binds [program]'s buffers, never this plan's by position *)
  program : Loop_program.t;
  edges : Tensor_id.t list;
      (** the graph edge (or a [synthetics] id) each of [program]'s buffers
          binds, positionally. A Region program's buffer ids are local to it
          (its output id is minted past its sources'), so they can collide with
          unrelated graph ids: never resolve a buffer by its own id. *)
  synthetics : synthetic list;
      (** the defaults [edges] names that are not graph edges *)
}

type error =
  [ Arena_plan.error
  | Eval_direct.error
  | Loop_node_program.error
  | `Plan_mismatch of Alloc_script.Position.t
  | `Region_lower of Loop_region_program.error ]

val pp_error : Format.formatter -> [< error ] -> unit

type t = {
  graph : graph;
  config : Storage_script.Config.t;
  script : Storage_script.t;
  plan : Storage_plan.t;
  inputs : Tensor_id.t list;  (** graph inputs, [Input.Input] kind *)
  constants : Tensor_id.t list;
      (** graph inputs, [Input.Constant] kind, used *)
  outputs : Tensor_id.t list;  (** [graph.Graph.outputs] *)
  invocations : invocation list;  (** schedule order *)
}

val build :
  ?limits:Kernel.Limits.t ->
  ?config:Storage_script.Config.t ->
  ?plan:Storage_plan.t ->
  graph ->
  (t, error) Err.t
(** [config] defaults to [Separate] layout with constants and inputs both
    [Copied] -- the simplest fully-owned corner of the storage contract (no
    caller lifetime obligations to track yet). *)

(** [plan]: a caller's already-witnessed plan, used instead of building one, but
    only if its script equals the one this graph and [config] produce (config,
    alignment policy and every event); otherwise [`Plan_mismatch] at the first
    difference. *)

(* Match combinators over the dataflow DAG: a state+error monad whose state is
   the view plus the nodes claimed so far. See .ai/native_transform_design.md §11.

   **No cursor.** Every primitive names the edge it operates on. For a linear
   token stream a hidden position is natural; for a DAG it is a lie, and passing
   the edge explicitly is both honest and more readable. Matching walks *up*
   operand edges, which is deterministic because operands are typed fields;
   [uses] covers fan-out.

   The projector is a plain [op -> 'a option], so a pattern destructures the
   existing payload records directly ([function Relu r -> Some r | _ -> None]).
   There is no per-op pattern DSL to keep in step as ops are added. *)

open Graph_ir

type mismatch =
  | Ambiguous_use of Tensor_id.t (* [use] found more than one candidate *)
  | Escapes of Tensor_id.t (* used outside, or a graph output *)
  | No_progress of Tensor_id.t (* a [chain] step failed to move upstream *)
  | Not_constant of Tensor_id.t
  | Rejected (* an explicit [fail] or a false [guard] *)
  | Shared of Tensor_id.t (* more than one consumer *)
  | Unproduced of Tensor_id.t (* a graph input where a node was wanted *)
  | Wrong_op of Node_id.t

val pp_mismatch : Format.formatter -> mismatch -> unit

(* [Invalid] is a malformed region rather than a failed match, so it aborts a
   scan instead of backtracking. *)
type failure = Invalid of Region.error | Mismatch of mismatch

val pp_failure : Format.formatter -> failure -> unit

type 'a t

val return : 'a -> 'a t
val ( let* ) : 'a t -> ('a -> 'b t) -> 'b t
val ( let+ ) : 'a t -> ('a -> 'b) -> 'b t
val fail : mismatch -> 'a t
val guard : bool -> unit t

(* First success wins; a failed branch discards its state wholesale, claims
   included, because the state is immutable. *)
val ( <|> ) : 'a t -> 'a t -> 'a t
val choice : 'a t list -> 'a t
val optional : 'a t -> 'a option t

(* The producing node of [edge], if the projector accepts it. Claims that node —
   idempotently, so two paths of a diamond reaching one producer both succeed. *)
val def : Tensor_id.t -> (op -> 'a option) -> ('a * node) t

(* Same, without claiming: for looking at context the rewrite will not touch. *)
val peek : Tensor_id.t -> (op -> 'a option) -> ('a * node) t
val uses : Tensor_id.t -> node list t

(* The single consumer of [edge] that the projector accepts, claimed. Fails when
   several match, since a pattern that silently picked one would be arbitrary. *)
val use : Tensor_id.t -> (op -> 'a option) -> ('a * node) t

(* Exactly one consumer and not a graph output — the precondition for consuming
   an edge inside a fused region. *)
val interior : Tensor_id.t -> unit t
val constant : Tensor_id.t -> unit t
val sig_of : Tensor_id.t -> Tensor_sig.t t
val claim : node -> unit t
val claimed : Node_id.Set.t t

(* The region claimed so far, for a guard that depends on the boundary. *)
val region : Region.t t

(* Repeat a step that consumes an edge and yields the next one upstream, until
   it stops matching. Each step must move strictly earlier in topological order,
   so a non-advancing parser cannot loop. *)
val chain : (Tensor_id.t -> ('a * Tensor_id.t) t) -> Tensor_id.t -> 'a list t

(* Runs a pattern and seals it: the region is derived from whatever the pattern
   claimed, so a match cannot describe its own boundary wrongly or forget to. *)
val run : 'a t -> Graph_view.t -> ('a * Region.t, failure) Stdlib.result

(* Anchors on every node output edge in [Graph.nodes] order, each output of a
   multi-output node separately. Greedy: the first match in that order wins, and
   a later one whose region intersects an accepted region is discarded rather
   than retried, so the results are disjoint. *)
val scan : (Tensor_id.t -> 'a t) -> Graph_view.t -> ('a * Region.t) list

(* The two value-graph stages: the stage program and the kernel.

   NOT [Me_build.Make]. That functor projects an op graph — nodes carrying an
   operator, edges carrying tensor ids — and neither of these is one. A
   [Stage_program.t] and a [Kernel.t] are lists of VALUES, each an
   [Expr.Value.t] whose dependencies are recovered by folding the expression
   rather than read off a field. The shared body would have to be parameterised
   over "how do I get a node's operands" in a way that made both call sites
   harder to read than two projections that each say what they do.

   ONE body serves both here, though, because they really are the same shape:
   an id, a signature, and a set of sources. The difference is where the values
   came from and which boundary each has, and that is what the two entry points
   below carry.

   EDGES COME FROM [Expr.Fold.sources], which is loads AND intrinsic
   descriptors. Taking [loads] alone would drop a real dependency — a max-pool's
   input is named only in its descriptor — and the picture would be wrong rather
   than incomplete.

   See .ai/model_explorer_design.md. *)

type error =
  [ Me_ids.error
  | `Over_limit of string * int
  | `Unknown_producer of int  (** a source no value and no input defines *) ]

val pp_error : Format.formatter -> [< error ] -> unit

val stage_program :
  limits:Me_limits.Limits.t ->
  id:string ->
  Stage_program.t ->
  (Model_explorer.Graph.t, [> error ]) Err.t
(** One node per stage, plus a pinned boundary node per graph input and per
    program output. The attribute is the stage's BODY, rendered through
    [Me_build.bounded] — an expression tree is exactly the value whose printing
    has to be stopped at the cap rather than built and cut. *)

val kernel :
  limits:Me_limits.Limits.t ->
  id:string ->
  Kernel.t ->
  (Model_explorer.Graph.t, [> error ]) Err.t
(** One node per kernel value. Kernel inputs are boundary nodes carrying their
    BINDING — caller, captured constant, or filled — because that is the
    distinction the adaptation exists to make and it appears nowhere in the
    graph shape. Outputs carry their quantisation, likewise. *)

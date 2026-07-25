(* The description of an edit, as data. Unversioned on purpose: the version-bound
   types ([allocator], [recipe]) and the functions that mint them live in
   [Rewrite], so there is no public constructor anywhere that could forge one.
   See .ai/native_transform_design.md §6. *)

open Graph_ir

(* An inserted node carries NO id — [Rewrite.apply] stamps it. So every
   [Node_id.t] in a recipe is a source id by construction, and the fresh-node
   namespace does not exist to be confused with the source one. [from] is the
   provenance the node map records. *)
type insertion = { op : op; outputs : Tensor_id.t list; from : Node_id.t list }

(* Where this replacement's insertions go: [Inherit] means the nearest group
   containing everything the replacement touched. *)
type placement = Inherit | New_group of string option

type replacement = {
  remove : Node_id.Set.t;
  insert : insertion list;
  placement : placement;
  tensors : Tensor_sig.t list; (* signatures for the edges [insert] defines *)
  subst : Tensor_id.t Tensor_id.Map.t;
      (* REWIRING only, applied to definitions and operands alike *)
  value_claims : (Tensor_id.t * Tensor_id.t * Correspondence.relation) list;
      (* correspondence, kept separate from [subst]: a self-claim for a
         redefined-but-unchanged edge involves no rewiring at all, and pinning it
         to a [t -> t] substitution would make normalisation either reject it as
         a cycle or erase it along with its claim *)
  constants : (Tensor_id.t * Tensor.packed) list;
  provenance : (Tensor_id.t list * Tensor_id.t) list;
}

val empty_replacement : replacement

(* The builder monad. Its only state is the id supply, so a pass can allocate
   the edges its replacement defines without ever seeing a raw [Id_supply.t]:
   [Rewrite.plan] unwraps the version-bound allocator around it. *)
type 'a t

val return : 'a -> 'a t
val ( let* ) : 'a t -> ('a -> 'b t) -> 'b t
val ( let+ ) : 'a t -> ('a -> 'b) -> 'b t

(* Allocate an edge and record its signature, so the replacement that defines it
   need not build [Tensor_sig.t] by hand. *)
val fresh :
  ?fmt:Payload.packed_fmt -> ?quant:Quant.t -> Vec6.shape -> Tensor_id.t t

val emit : replacement -> unit t

(* Smart constructors. Each emits its own [value_claims], including self-claims
   where an id is preserved but its definition changes — [Rewrite.apply] never
   infers a claim, so a constructor that stayed silent would be rejected. *)

(* Trivial trimming: drop nodes and tie each removed output to the surviving
   upstream edge it is replaced by. *)
val trim :
  remove:Node_id.t list -> tie:(Tensor_id.t * Tensor_id.t) list -> unit t

(* 1:N / N:1 / M:N. [subst] maps each replacement-side output onto the source
   output it takes over, and [claims] overrides the default [Identical] for any
   pair whose value actually changed. *)
val replace :
  remove:Node_id.t list ->
  insert:insertion list ->
  ?tensors:Tensor_sig.t list ->
  ?subst:(Tensor_id.t * Tensor_id.t) list ->
  ?claims:(Tensor_id.t * Tensor_id.t * Correspondence.relation) list ->
  ?placement:placement ->
  unit ->
  unit t

(* Functional trimming: a node becomes a [Constant] input holding its computed
   value. The output id is preserved — same tensor, computed earlier — so the
   claim is a self-claim, and the constants it consumed become provenance. *)
val fold_to_constant :
  node:Node_id.t ->
  output:Tensor_id.t ->
  value:Tensor.packed ->
  sources:Tensor_id.t list ->
  unit t

val run : 'a t -> Id_supply.t -> 'a * replacement list * Id_supply.t
val pp_replacement : Format.formatter -> replacement -> unit

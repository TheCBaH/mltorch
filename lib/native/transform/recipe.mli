(* The description of an edit, as data — indexed by the graph version it is an
   edit OF. See .ai/native_transform_design.md §6 and
   .ai/native_transform_versioning.md.

   The header used to say "unversioned on purpose": the version-bound types lived
   in [Rewrite], so no public constructor could forge one. That is superseded.
   [Rewrite.plan] still supplies the version — a recipe cannot pick its own — but
   the edit's own metadata is now typed, so a claim cannot be stated backwards
   and a fresh edge cannot be passed where an existing one is required.

   Three kinds of edge, and which one a field takes is the whole point:

   - ['v source]  exists in the graph being rewritten. Only [existing] makes one,
                  and only by finding the id in that graph.
   - ['v fresh]   allocated by this recipe. Does not exist yet anywhere.
   - ['v target]  what an edit DEFINES: either fresh, or a source output taken
                  over.

   Only substitution is symmetric, and it genuinely runs both ways: [trim]
   substitutes existing onto existing, [fold_batch_norm] existing onto fresh. Its
   ['v edit_edge] union is therefore deliberate and confined to that one field —
   a single union across all of them would let a claim be stated as
   [(fresh, source)] and give up the guarantee this exists for. *)

open Graph_ir

module Make (S : Side.S) : sig
  type 'v fresh
  type 'v source
  type 'v target = Fresh of 'v fresh | Preserved of 'v source
  type 'v edit_edge = New of 'v fresh | Old of 'v source

  (* One-way, and the direction that matters: [Graph_ir.op] takes raw
     [tensor_ref]s, so a pass writing an inserted node's payload projects out of
     the typed world here. Nothing lets a raw id acquire a tag; [existing] is the
     only way in, and it checks. See the limitations note in
     .ai/native_transform_versioning.md §7. *)
  val raw_edit_edge : 'v edit_edge -> Tensor_id.t
  val raw_fresh : 'v fresh -> Tensor_id.t
  val raw_source : 'v source -> Tensor_id.t
  val raw_target : 'v target -> Tensor_id.t

  (* An inserted node carries NO id — [Rewrite.apply] stamps it. So every
     [Node_id.t] in a recipe is a source id by construction, and the fresh-node
     namespace does not exist to be confused with the source one. [from] is the
     provenance the node map records. *)
  type 'v insertion = {
    op : S.Dialect.op;
    outputs : 'v target list;
    from : Node_id.t list;
  }

  (* Where this replacement's insertions go: [Inherit] means the nearest group
     containing everything the replacement touched. *)
  type placement = Inherit | New_group of string option

  type 'v replacement = {
    remove : Node_id.Set.t;
    insert : 'v insertion list;
    placement : placement;
    tensors : Tensor_sig.t list; (* signatures for the edges [insert] defines *)
    subst : ('v edit_edge * 'v edit_edge) list;
        (* REWIRING only, applied to definitions and operands alike *)
    value_claims : ('v source * 'v target * Correspondence.relation) list;
        (* correspondence, kept separate from [subst]: a self-claim for a
           redefined-but-unchanged edge involves no rewiring at all, and pinning it
           to a [t -> t] substitution would make normalisation either reject it as
           a cycle or erase it along with its claim.

           Directional, unlike [subst]: [Rewrite.apply] puts the first element into
           the cluster's SOURCE side unconditionally. *)
    constants : ('v target * Tensor.packed) list;
    provenance : ('v source list * 'v target) list;
  }

  val empty_replacement : 'v replacement

  (* The builder monad, over the id supply and the version being edited. A pass
     allocates the edges its replacement defines without ever seeing a raw
     [Id_supply.t]: [Rewrite.plan] unwraps the version-bound allocator around
     it. *)
  type ('v, 'a) t
  type error = [ `Unknown_source_edge of Tensor_id.t ]

  val pp_error : Format.formatter -> [< error ] -> unit
  val return : 'a -> ('v, 'a) t
  val ( let* ) : ('v, 'a) t -> ('a -> ('v, 'b) t) -> ('v, 'b) t
  val ( let+ ) : ('v, 'a) t -> ('a -> 'b) -> ('v, 'b) t

  (* Not a [List] submodule: passes [open Recipe], and shadowing [List] there
     would silently redirect every ordinary list operation in a build function. *)
  val all : ('a -> ('v, 'b) t) -> 'a list -> ('v, 'b list) t

  (* Resolve a raw id against the graph being edited. This is where a matcher's
     output — [Pattern] and [Region] stay raw, being structural queries over the
     view — enters the typed world, and the only place it can. *)
  val existing : Tensor_id.t -> ('v, 'v source) t

  (* Allocate an edge and record its signature, so the replacement that defines it
     need not build [Tensor_sig.t] by hand. *)
  val fresh :
    ?fmt:Payload.packed_fmt -> ?quant:Quant.t -> Vec6.shape -> ('v, 'v fresh) t

  val emit : 'v replacement -> ('v, unit) t

  (* Smart constructors. Each emits its own [value_claims], including self-claims
     where an id is preserved but its definition changes — [Rewrite.apply] never
     infers a claim, so a constructor that stayed silent would be rejected. *)

  (* Trivial trimming: drop nodes and tie each removed output to the surviving
     upstream edge it is replaced by. Both ends are existing source edges, so this
     one stays unprotected against reversal — the types cannot tell them apart. *)
  val trim :
    remove:Node_id.t list -> tie:('v source * 'v source) list -> ('v, unit) t

  (* 1:N / N:1 / M:N. [subst] maps each replacement-side output onto the source
     output it takes over, and [claims] overrides the default [Identical] for any
     pair whose value actually changed. *)
  val replace :
    remove:Node_id.t list ->
    insert:'v insertion list ->
    ?tensors:Tensor_sig.t list ->
    ?subst:('v edit_edge * 'v edit_edge) list ->
    ?claims:('v source * 'v target * Correspondence.relation) list ->
    ?placement:placement ->
    unit ->
    ('v, unit) t

  (* Functional trimming: a node becomes a [Constant] input holding its computed
     value. The output id is preserved — same tensor, computed earlier — so the
     claim is a self-claim, and the constants it consumed become provenance. *)
  val fold_to_constant :
    node:Node_id.t ->
    output:'v source ->
    value:Tensor.packed ->
    sources:'v source list ->
    ('v, unit) t

  (* [Rewrite.plan]'s entry point: the snapshot is what [existing] resolves
     against, so a recipe is bound to the version it was planned from. *)
  val run :
    ('v, 'a) t ->
    'v S.Snapshot.t ->
    Id_supply.t ->
    ('a * 'v replacement list * Id_supply.t, error) Core.result

  val pp_replacement : Format.formatter -> 'v replacement -> unit
end

include module type of Make (Native_side)

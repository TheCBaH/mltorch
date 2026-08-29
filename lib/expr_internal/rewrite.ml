(* Everything here rebuilds with the RAW constructors, never the smart ones.
     Smart constructors fold -- [floor_div_pos] by 1 collapses today, and stage 8
     adds more -- and a structure-preserving rewrite that silently changed syntax
     would make the migration's golden diff unattributable. The raw constructors
     are reachable because these are sections of [Expr] rather than separate
     units. *)

let rec map_index_reducers : type r.
    (Reduce_var.t -> Reduce_var.t) -> r Index.t -> r Index.t =
 fun f i ->
  match i with
  | Index.Output _ | Index.Zero | Index.Const _ -> i
  | Index.Reduce v -> Index.Reduce (f v)
  | Index.Of_position a -> Index.Of_position (map_index_reducers f a)
  | Index.Clamp_low a -> Index.Clamp_low (map_index_reducers f a)
  | Index.Assume_position a -> Index.Assume_position (map_index_reducers f a)
  | Index.Scale (k, a) -> Index.Scale (k, map_index_reducers f a)
  | Index.Floor_div_pos (a, d) -> Index.Floor_div_pos (map_index_reducers f a, d)
  | Index.Ceil_div_pos (a, d) -> Index.Ceil_div_pos (map_index_reducers f a, d)
  | Index.Add (a, b) ->
      Index.Add (map_index_reducers f a, map_index_reducers f b)
  | Index.Min (a, b) ->
      Index.Min (map_index_reducers f a, map_index_reducers f b)
  | Index.Max (a, b) ->
      Index.Max (map_index_reducers f a, map_index_reducers f b)

let rec subst_index : type r.
    Role.Position.t Index.t Coord.t -> r Index.t -> r Index.t =
 fun c i ->
  match i with
  (* The one substituted form. Role-preserving: an output variable is a
       position and so is its replacement. *)
  | Index.Output a -> Coord.get c a
  (* Deliberately NOT substituted -- a reducer is bound by its reduction, and
       replacing one here is precisely the capture this module prevents. *)
  | Index.Reduce _ | Index.Zero | Index.Const _ -> i
  | Index.Of_position a -> Index.Of_position (subst_index c a)
  | Index.Clamp_low a -> Index.Clamp_low (subst_index c a)
  | Index.Assume_position a -> Index.Assume_position (subst_index c a)
  | Index.Scale (k, a) -> Index.Scale (k, subst_index c a)
  | Index.Floor_div_pos (a, d) -> Index.Floor_div_pos (subst_index c a, d)
  | Index.Ceil_div_pos (a, d) -> Index.Ceil_div_pos (subst_index c a, d)
  | Index.Add (a, b) -> Index.Add (subst_index c a, subst_index c b)
  | Index.Min (a, b) -> Index.Min (subst_index c a, subst_index c b)
  | Index.Max (a, b) -> Index.Max (subst_index c a, subst_index c b)

(* Rank-2, for the same reason as [Fold.walk]: a [Load]'s components are
     [Role.Position.t Index.t] while a reduction's upper bound is
     [Role.Delta.t Index.t], and a plain function argument would be fixed at
     whichever inference saw first. *)
type 'env idx_fn = { on_index : 'r. 'env -> 'r Index.t -> 'r Index.t }

let keep_indices = { on_index = (fun _ i -> i) }

(* A structural map that rebuilds every node, parameterised by what to do at
     the leaves an individual rewrite cares about. [on_reduce] sees each binder
     and returns its replacement plus the environment its body is rewritten
     under, which is what keeps scope handling in ONE place instead of repeated
     per rewrite. *)
(* STATE-PASSING, with [Builder.state] last: [rebuild ~idx ~src ~on_reduce env
     e] then has exactly the representation of a [Value.t Builder.t], since the
     monad is [state -> 'a * state] and this is a section of the same unit. With
     the state in the middle, partial application would leave
     [state -> Value.t -> _], which is not a computation at all.

     Written directly rather than with [let*]. A monadic tower would allocate a
     closure per AST node and then walk that tree to produce the result -- two
     passes plus allocation where there is one -- on the traversal fusion is
     meant to use, where expression size is the budget.

     The state threads LEFT TO RIGHT through every multi-child node: each child
     continues from what its predecessor returned. Handing a sibling the incoming
     state instead would let two freshened binders take the same identity,
     silently, which is the collision this module exists to prevent. Index
     rewriting does NOT thread, and cannot: [on_index] is pure, so no index
     position can mint. *)
(* [on_load] sees the already-rewritten source and coordinate and returns the
     node that replaces the load. It exists so a load can become a SUBTREE
     rather than only a renamed symbol, without a second scope-aware traversal
     of [Value.t] living outside this module. Its result is returned as-is and
     never fed back through [go]: an inserted fragment is not re-traversed, and
     the composition rules that depend on that say so. *)
let keep_load s c st = (Value.Load (s, c), st)

let rec rebuild ~idx ~src ~on_load ~on_reduce env (e : Value.t) st =
  let go = rebuild ~idx ~src ~on_load ~on_reduce env in
  let idxe i = idx.on_index env i in
  let unary wrap a st =
    let a, st = go a st in
    (wrap a, st)
  in
  match e with
  | Value.Const _ -> (e, st)
  | Value.Binary (op, a, b) ->
      let a, st = go a st in
      let b, st = go b st in
      (Value.Binary (op, a, b), st)
  | Value.Unary (op, a) -> unary (fun a -> Value.Unary (op, a)) a st
  | Value.Round_f32 a -> unary (fun a -> Value.Round_f32 a) a st
  | Value.Select (c, a, b) ->
      let c, st =
        match c with
        | Bool.Value_lt (x, y) ->
            let x, st = go x st in
            let y, st = go y st in
            (Bool.Value_lt (x, y), st)
        | Bool.Index_eq (x, y) -> (Bool.Index_eq (idxe x, idxe y), st)
      in
      let a, st = go a st in
      let b, st = go b st in
      (Value.Select (c, a, b), st)
  | Value.Value_of_index i -> (Value.Value_of_index (idxe i), st)
  | Value.Load (s, c) -> on_load (src s) (Coord.map idxe c) st
  | Value.Reduce r ->
      let var, env', st = on_reduce env r.Reduction.var st in
      let body, st =
        rebuild ~idx ~src ~on_load ~on_reduce env' r.Reduction.body st
      in
      ( Value.Reduce
          {
            Reduction.kind = r.Reduction.kind;
            var;
            (* Bounds sit OUTSIDE the binder: rewritten under the enclosing
                 environment, not the body's. *)
            lo = idxe r.Reduction.lo;
            hi = idxe r.Reduction.hi;
            body;
          },
        st )
  | Value.Intrinsic (Intrinsic.Max_pool d) ->
      ( Value.Intrinsic
          (Intrinsic.Max_pool
             {
               d with
               Intrinsic.Max_pool.source = src d.Intrinsic.Max_pool.source;
               out = Coord.map idxe d.Intrinsic.Max_pool.out;
             }),
        st )

let subst_env env v =
  match Reduce_var.Map.find_opt v env with Some w -> w | None -> v

(* Replaces every BOUND reducer identity consistently and leaves free ones
     alone. Composing two independently built fragments is exactly when this is
     needed: both supplies start at [initial], so both mint ordinal 0, and the
     inner binder would otherwise capture references meant for the outer one.

     Freshen the fragment being INSERTED, before composing it. Freshening the
     combined tree afterwards cannot repair anything -- once a nominal collision
     has captured a reference there is no record of which binder it meant. *)
let freshen e s =
  (* Replacements must SKIP identities that occur free in [e]. A free reducer
       is a reference to a binder outside this expression, and minting one of
       them here binds it -- silently turning an ill-scoped term into a
       well-scoped-looking one with a different denotation, which [Check] would
       then report as [Ok].

       [alpha_normalize] makes that maximally likely, since it always starts
       from [initial]: any free ordinal near zero is directly in the way. *)
  let free = Fold.free_reducers e in
  let rec mint st =
    let w, st = Builder.run_from st Builder.fresh_reduce in
    if Reduce_var.Set.mem w free then mint st else (w, st)
  in
  let on_reduce env v st =
    let w, st = mint st in
    (w, Reduce_var.Map.add v w env, st)
  in
  rebuild
    ~idx:{ on_index = (fun env i -> map_index_reducers (subst_env env) i) }
    ~src:Fun.id ~on_load:keep_load ~on_reduce Reduce_var.Map.empty e s

(* Deterministic renaming by lexical traversal: running [freshen] from a fixed
     initial supply gives binders the LOWEST ORDINALS NOT FREE in the
     expression, in the order they are met -- 0, 1, ... for a closed expression,
     and skipping the free ones otherwise, because [freshen] must not bind them.
     Canonical either way: alpha-equivalent expressions share a free set, so
     they skip the same ordinals. Does not reorder operations or reductions. *)
let alpha_normalize e = fst (freshen e Builder.initial)

(* Replaces only output-axis variables, never reducers. If the result is
     placed beneath another reduction, the CALLER freshens it first -- this
     function cannot know that context. *)
(* Neither of these mints, so their [on_reduce] passes the state straight
     through and the supply they are run from is immaterial. [Builder.initial]
     is the arbitrary but fixed choice; the discarded final state is the proof
     that nothing was allocated. *)
let keep_reducer () v st = (v, (), st)

let substitute_output c e =
  fst
    (rebuild
       ~idx:{ on_index = (fun _ i -> subst_index c i) }
       ~src:Fun.id ~on_load:keep_load ~on_reduce:keep_reducer () e
       Builder.initial)

let map_sources f e =
  fst
    (rebuild ~idx:keep_indices ~src:f ~on_load:keep_load ~on_reduce:keep_reducer
       () e Builder.initial)

(* Replaces ordinary [Load] nodes with whole subtrees, in the SAME builder
     namespace as the destination — which is the point of returning a
     computation. Freshening each inserted fragment from [Builder.initial]
     instead would have every fragment mint ordinal 0 again, and the collision
     [freshen] exists to prevent would be reintroduced by the very act of
     composing.

     Intrinsic descriptors are left structurally intact. An [Intrinsic.Max_pool]
     holds a source and window geometry, not a load node, so there is nothing
     inside it that one scalar subtree could stand in for; its source still goes
     through [src] as always, and a caller that needs to eliminate such a
     dependency has to reject it instead. *)
let substitute_loads f e st =
  rebuild ~idx:keep_indices ~src:Fun.id
    ~on_load:(fun s c st ->
      match f s c with
      | None -> (Value.Load (s, c), st)
      | Some replacement -> Builder.run_from st replacement)
    ~on_reduce:keep_reducer () e st

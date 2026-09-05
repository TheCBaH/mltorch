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
  | Index.Add (a, b) ->
      Index.Add (map_index_reducers f a, map_index_reducers f b)
  | Index.Assume_position a -> Index.Assume_position (map_index_reducers f a)
  | Index.Ceil_div_pos (a, d) -> Index.Ceil_div_pos (map_index_reducers f a, d)
  | Index.Clamp_low a -> Index.Clamp_low (map_index_reducers f a)
  | Index.Const _ -> i
  | Index.Data (s, c, extent) ->
      Index.Data (s, Coord.map (map_index_reducers f) c, extent)
  | Index.Floor_div_pos (a, d) -> Index.Floor_div_pos (map_index_reducers f a, d)
  | Index.Max (a, b) ->
      Index.Max (map_index_reducers f a, map_index_reducers f b)
  | Index.Min (a, b) ->
      Index.Min (map_index_reducers f a, map_index_reducers f b)
  | Index.Of_position a -> Index.Of_position (map_index_reducers f a)
  | Index.Output _ -> i
  | Index.Reduce v -> Index.Reduce (f v)
  | Index.Scale (k, a) -> Index.Scale (k, map_index_reducers f a)
  | Index.Zero -> i

let rec subst_index : type r.
    Role.Position.t Index.t Coord.t -> r Index.t -> r Index.t =
 fun c i ->
  match i with
  | Index.Add (a, b) -> Index.Add (subst_index c a, subst_index c b)
  | Index.Assume_position a -> Index.Assume_position (subst_index c a)
  | Index.Ceil_div_pos (a, d) -> Index.Ceil_div_pos (subst_index c a, d)
  | Index.Clamp_low a -> Index.Clamp_low (subst_index c a)
  | Index.Const _ -> i
  | Index.Data (s, dc, extent) ->
      Index.Data (s, Coord.map (subst_index c) dc, extent)
  | Index.Floor_div_pos (a, d) -> Index.Floor_div_pos (subst_index c a, d)
  | Index.Max (a, b) -> Index.Max (subst_index c a, subst_index c b)
  | Index.Min (a, b) -> Index.Min (subst_index c a, subst_index c b)
  | Index.Of_position a -> Index.Of_position (subst_index c a)
  (* The one substituted form. Role-preserving: an output variable is a
       position and so is its replacement. *)
  | Index.Output a -> Coord.get c a
  (* Deliberately NOT substituted -- a reducer is bound by its reduction, and
       replacing one here is precisely the capture this module prevents. *)
  | Index.Reduce _ -> i
  | Index.Scale (k, a) -> Index.Scale (k, subst_index c a)
  | Index.Zero -> i

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

(* [on_local_at]/[on_local_scan_at] mirror [on_load]: they see the
     already-rewritten local id and index/indices and return the node that
     replaces the read -- both indices have already gone through [idx], same
     as [Load]'s coordinate. [lenv], threaded alongside [env], is [prev]'s
     namesake: empty except within its own scan's [update], and consulted by
     every local-reading callback so a rewrite that cares (only [freshen]
     today) can tell a bound [prev] apart from an ordinary Region local.
     [on_local_bind] mints/keeps [prev]'s replacement and extends [lenv],
     mirroring [on_reduce] for [lane]/[step]. *)
let rec rebuild ~idx ~src ~on_load ~on_local ~on_local_at ~on_local_scan_at
    ~on_reduce ~on_local_bind env lenv (e : Value.t) st =
  let go =
    rebuild ~idx ~src ~on_load ~on_local ~on_local_at ~on_local_scan_at
      ~on_reduce ~on_local_bind env lenv
  in
  let idxe i = idx.on_index env i in
  let unary wrap a st =
    let a, st = go a st in
    (wrap a, st)
  in
  match e with
  | Value.Const _ -> (e, st)
  | Value.Local v -> on_local lenv v st
  | Value.Local_at (v, i) -> on_local_at lenv v (idxe i) st
  | Value.Local_scan_at (v, row, lane) ->
      on_local_scan_at lenv v (idxe row) (idxe lane) st
  | Value.Binary (op, a, b) ->
      let a, st = go a st in
      let b, st = go b st in
      (Value.Binary (op, a, b), st)
  | Value.Unary (op, a) -> unary (fun a -> Value.Unary (op, a)) a st
  | Value.Round_f32 a -> unary (fun a -> Value.Round_f32 a) a st
  | Value.Scan_at (s, row, lane) ->
      let _lane1, env1, st = on_reduce env s.Scan.lane st in
      let init, st =
        rebuild ~idx ~src ~on_load ~on_local ~on_local_at ~on_local_scan_at
          ~on_reduce ~on_local_bind env1 lenv s.Scan.init st
      in
      let lane2, env2, st = on_reduce env s.Scan.lane st in
      let step, env3, st = on_reduce env2 s.Scan.step st in
      let prev, lenv1, st = on_local_bind lenv s.Scan.prev st in
      let update, st =
        rebuild ~idx ~src ~on_load ~on_local ~on_local_at ~on_local_scan_at
          ~on_reduce ~on_local_bind env3 lenv1 s.Scan.update st
      in
      ( Value.scan_at
          {
            Scan.width = s.Scan.width;
            steps = s.Scan.steps;
            lane = lane2;
            step;
            prev;
            init;
            update;
          }
          ~row:(idxe row) ~lane:(idxe lane),
        st )
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
        rebuild ~idx ~src ~on_load ~on_local ~on_local_at ~on_local_scan_at
          ~on_reduce ~on_local_bind env' lenv r.Reduction.body st
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

(* Neither of these mints, so their [on_reduce]/[on_local_bind] pass the
   supply straight through -- shared by every rewrite below that does not
   introduce identities of its own. *)
let keep_reducer () v st = (v, (), st)
let keep_local_bind () v st = (v, (), st)

let keep_local_scan_at _lenv v row lane st =
  (Value.local_scan_at v ~row ~lane, st)

(* Replaces every BOUND reducer identity consistently and leaves free ones
     alone. Composing two independently built fragments is exactly when this is
     needed: both supplies start at [initial], so both mint ordinal 0, and the
     inner binder would otherwise capture references meant for the outer one.

     Freshen the fragment being INSERTED, before composing it. Freshening the
     combined tree afterwards cannot repair anything -- once a nominal collision
     has captured a reference there is no record of which binder it meant. *)
let freshen e s =
  (* Replacements must SKIP identities that occur free in [e]. A free
       reducer/local is a reference to a binder outside this expression, and
       minting one of them here binds it -- silently turning an ill-scoped
       term into a well-scoped-looking one with a different denotation, which
       [Check] would then report as [Ok].

       [alpha_normalize] makes that maximally likely, since it always starts
       from [initial]: any free ordinal near zero is directly in the way. *)
  let free = Fold.free_reducers e in
  let free_locals = Fold.locals e in
  let rec mint st =
    let w, st = Builder.run_from st Builder.fresh_reduce in
    if Reduce_var.Set.mem w free then mint st else (w, st)
  in
  let rec mint_local st =
    let w, st = Builder.run_from st Builder.fresh_local in
    if Local_var.Set.mem w free_locals then mint_local st else (w, st)
  in
  let on_reduce env v st =
    let w, st = mint st in
    (w, Reduce_var.Map.add v w env, st)
  in
  let on_local_bind lenv v st =
    let w, st = mint_local st in
    (w, Local_var.Map.add v w lenv, st)
  in
  rebuild
    ~idx:{ on_index = (fun env i -> map_index_reducers (subst_env env) i) }
    ~src:Fun.id ~on_load:keep_load
    ~on_local:(fun _lenv v st -> (Value.Local v, st))
      (* [prev] is the only local this can ever rename -- [lenv] holds it only
       within its own scan's [update]; every other read is an ordinary
       external Region local, passed through unchanged exactly as before
       scan existed. *)
    ~on_local_at:(fun lenv v i st ->
      match Local_var.Map.find_opt v lenv with
      | Some w -> (Value.Local_at (w, i), st)
      | None -> (Value.Local_at (v, i), st))
    ~on_local_scan_at:(fun _lenv v row lane st ->
      (Value.local_scan_at v ~row ~lane, st))
    ~on_reduce ~on_local_bind Reduce_var.Map.empty Local_var.Map.empty e s

(* Deterministic renaming by lexical traversal: running [freshen] from a fixed
     initial supply gives binders the LOWEST ORDINALS NOT FREE in the
     expression, in the order they are met -- 0, 1, ... for a closed expression,
     and skipping the free ones otherwise, because [freshen] must not bind them.
     Canonical either way: alpha-equivalent expressions share a free set, so
     they skip the same ordinals. Does not reorder operations or reductions. *)
let alpha_normalize e = fst (freshen e Builder.initial)

(* [Rewrite]'s reusable freshen-a-standalone-descriptor helper: wraps [s] in
   a placeholder [Scan_at] (row/lane are not touched by [freshen] -- they
   carry no reducer this scan itself binds), runs the ordinary [freshen], and
   unwraps. Used wherever a [Scan.t] is spliced into a different context and
   needs its own [lane]/[step]/[prev] made fresh first, exactly as a scalar
   or vector local's stored value already is. *)
let freshen_scan (s : Scan.t) st =
  match freshen (Value.scan_at s ~row:Index.zero ~lane:Index.zero) st with
  | Value.Scan_at (s', _, _), st -> (s', st)
  | _ -> assert false

(* Replaces only output-axis variables, never reducers. If the result is
     placed beneath another reduction, the CALLER freshens it first -- this
     function cannot know that context. *)
let substitute_output c e =
  fst
    (rebuild
       ~idx:{ on_index = (fun _ i -> subst_index c i) }
       ~src:Fun.id ~on_load:keep_load
       ~on_local:(fun _lenv v st -> (Value.Local v, st))
       ~on_local_at:(fun _lenv v i st -> (Value.Local_at (v, i), st))
       ~on_local_scan_at:keep_local_scan_at ~on_reduce:keep_reducer
       ~on_local_bind:keep_local_bind () () e Builder.initial)

(* Targeted substitution of ONE specific (necessarily free) reducer identity,
     for beta-reducing a vector local's body at a read site: [var] is the
     binder [Region_local.vector] mints for "my own index", free within the
     local's stored [value] by construction (nothing inside that value binds
     it), so replacing it here is resolving an external reference, not the
     capture [subst_index]'s own [Reduce] case guards against. Generalized
     over the index role the same way [subst_index] is, since [var] can occur
     under [Of_position] at [Role.Delta.t] as well as directly at
     [Role.Position.t]. *)
let rec subst_reducer : type r.
    Reduce_var.t -> Role.Position.t Index.t -> r Index.t -> r Index.t =
 fun v repl i ->
  match i with
  | Index.Add (a, b) ->
      Index.Add (subst_reducer v repl a, subst_reducer v repl b)
  | Index.Assume_position a -> Index.Assume_position (subst_reducer v repl a)
  | Index.Ceil_div_pos (a, d) -> Index.Ceil_div_pos (subst_reducer v repl a, d)
  | Index.Clamp_low a -> Index.Clamp_low (subst_reducer v repl a)
  | Index.Const _ -> i
  | Index.Data (s, c, extent) ->
      Index.Data (s, Coord.map (subst_reducer v repl) c, extent)
  | Index.Floor_div_pos (a, d) -> Index.Floor_div_pos (subst_reducer v repl a, d)
  | Index.Max (a, b) ->
      Index.Max (subst_reducer v repl a, subst_reducer v repl b)
  | Index.Min (a, b) ->
      Index.Min (subst_reducer v repl a, subst_reducer v repl b)
  | Index.Of_position a -> Index.Of_position (subst_reducer v repl a)
  | Index.Output _ -> i
  | Index.Reduce w -> if Reduce_var.equal v w then repl else i
  | Index.Scale (k, a) -> Index.Scale (k, subst_reducer v repl a)
  | Index.Zero -> i

(* [rebuild] specialised to a pure, non-minting substitution of [var] for
     [repl] everywhere in [e] -- the same shape as [substitute_output], reused
     here instead of a hand-written recursion over [Value.t] so [Select]'s
     [Bool.Index_eq] guard and the [Max_pool] descriptor get the substitution
     too, for free, rather than by a second, separately-reviewed traversal. *)
let substitute_reducer var repl e =
  fst
    (rebuild
       ~idx:{ on_index = (fun () i -> subst_reducer var repl i) }
       ~src:Fun.id ~on_load:keep_load
       ~on_local:(fun _lenv v st -> (Value.Local v, st))
       ~on_local_at:(fun _lenv v i st -> (Value.Local_at (v, i), st))
       ~on_local_scan_at:keep_local_scan_at ~on_reduce:keep_reducer
       ~on_local_bind:keep_local_bind () () e Builder.initial)

(* [Data]'s own source is a real source dependency too (per [Fold.sources]),
   so [map_sources] cannot rewrite it via [keep_indices] the way it did before
   [Data] existed -- unlike [subst_index]/[map_index_reducers] above, this
   recursion changes a SOURCE, not a coordinate or a reducer identity, so it
   gets its own traversal rather than overloading either of those. *)
let rec map_index_sources : type r.
    (Source.t -> Source.t) -> r Index.t -> r Index.t =
 fun f i ->
  match i with
  | Index.Add (a, b) -> Index.Add (map_index_sources f a, map_index_sources f b)
  | Index.Assume_position a -> Index.Assume_position (map_index_sources f a)
  | Index.Ceil_div_pos (a, d) -> Index.Ceil_div_pos (map_index_sources f a, d)
  | Index.Clamp_low a -> Index.Clamp_low (map_index_sources f a)
  | Index.Const _ -> i
  | Index.Data (s, c, extent) ->
      Index.Data (f s, Coord.map (map_index_sources f) c, extent)
  | Index.Floor_div_pos (a, d) -> Index.Floor_div_pos (map_index_sources f a, d)
  | Index.Max (a, b) -> Index.Max (map_index_sources f a, map_index_sources f b)
  | Index.Min (a, b) -> Index.Min (map_index_sources f a, map_index_sources f b)
  | Index.Of_position a -> Index.Of_position (map_index_sources f a)
  | Index.Output _ -> i
  | Index.Reduce _ -> i
  | Index.Scale (k, a) -> Index.Scale (k, map_index_sources f a)
  | Index.Zero -> i

let map_sources f e =
  fst
    (rebuild
       ~idx:{ on_index = (fun _ i -> map_index_sources f i) }
       ~src:f ~on_load:keep_load
       ~on_local:(fun _lenv v st -> (Value.Local v, st))
       ~on_local_at:(fun _lenv v i st -> (Value.Local_at (v, i), st))
       ~on_local_scan_at:keep_local_scan_at ~on_reduce:keep_reducer
       ~on_local_bind:keep_local_bind () () e Builder.initial)

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
    ~on_local:(fun _lenv v st -> (Value.Local v, st))
    ~on_local_at:(fun _lenv v i st -> (Value.Local_at (v, i), st))
    ~on_local_scan_at:keep_local_scan_at ~on_reduce:keep_reducer
    ~on_local_bind:keep_local_bind () () e st

(* What a local resolves to during [substitute_locals]. A closed variant, not
     two callbacks or a wider [Value.t] convention, per CLAUDE.md's payload
     rule: a scalar local substitutes its whole value at a [Local] occurrence,
     a vector local instead carries the binder [var] its stored [body] is
     parameterised over, substituted at a [Local_at] occurrence's read index
     -- the beta-reduction [Region_local.vector]'s "body may mention the
     binder" depends on. Which occurrence a given local's node kind may appear
     as (shape agreement) is [Region_program.check]'s job, not this module's:
     [Rewrite] rebuilds structure, it does not validate it, so a mismatch here
     is reported the same way an already-invalid tree elsewhere would be --
     structurally, via [invalid_arg], never silently. *)
type local_binding =
  | Scalar of Value.t
  | Scan of Scan.t
  | Vector of { var : Reduce_var.t; body : Value.t }

let substitute_locals f e st =
  rebuild ~idx:keep_indices ~src:Fun.id ~on_load:keep_load
    ~on_local:(fun _lenv v st ->
      match f v with
      | None -> (Value.Local v, st)
      | Some (Scalar value) -> Builder.run_from st (freshen value)
      | Some (Vector _) ->
          invalid_arg
            "Expr.Rewrite.substitute_locals: vector local read as a scalar"
      | Some (Scan _) ->
          invalid_arg
            "Expr.Rewrite.substitute_locals: trace local read as a scalar")
    ~on_local_at:(fun _lenv v i st ->
      match f v with
      | None -> (Value.Local_at (v, i), st)
      | Some (Vector { var; body }) ->
          let body, st = Builder.run_from st (freshen body) in
          (substitute_reducer var i body, st)
      | Some (Scalar _) ->
          invalid_arg
            "Expr.Rewrite.substitute_locals: scalar local read with an index"
      | Some (Scan _) ->
          invalid_arg
            "Expr.Rewrite.substitute_locals: trace local read as a vector")
      (* This is what turns a cached [Local_scan_at] read into the inline
       [Scan_at] descriptor [specialize_pixel] needs: the same freshen-
       before-splice discipline as [Scalar]/[Vector] above, applied to the
       whole descriptor via [freshen_scan]. *)
    ~on_local_scan_at:(fun _lenv v row lane st ->
      match f v with
      | None -> (Value.local_scan_at v ~row ~lane, st)
      | Some (Scan s) ->
          let s, st = freshen_scan s st in
          (Value.scan_at s ~row ~lane, st)
      | Some (Scalar _) ->
          invalid_arg
            "Expr.Rewrite.substitute_locals: scalar local read as a trace"
      | Some (Vector _) ->
          invalid_arg
            "Expr.Rewrite.substitute_locals: vector local read as a trace")
    ~on_reduce:keep_reducer ~on_local_bind:keep_local_bind () () e st

(* Concrete, reviewed queries rather than a generic visitor: binder behaviour
     stays visible in each signature, and a new constructor breaks the ones that
     must handle it instead of silently falling through a default. *)

let rec index_reducers : type r. Reduce_var.Set.t -> r Index.t -> _ =
 fun acc -> function
  | Index.Add (a, b) -> index_reducers (index_reducers acc a) b
  | Index.Assume_position a -> index_reducers acc a
  | Index.Ceil_div_pos (a, _) -> index_reducers acc a
  | Index.Clamp_low a -> index_reducers acc a
  | Index.Const _ -> acc
  | Index.Data (_, c, _) -> Coord.fold (fun acc i -> index_reducers acc i) acc c
  | Index.Floor_div_pos (a, _) -> index_reducers acc a
  | Index.Max (a, b) -> index_reducers (index_reducers acc a) b
  | Index.Min (a, b) -> index_reducers (index_reducers acc a) b
  | Index.Of_position a -> index_reducers acc a
  | Index.Output _ -> acc
  | Index.Reduce v -> Reduce_var.Set.add v acc
  | Index.Scale (_, a) -> index_reducers acc a
  | Index.Zero -> acc

let rec index_axes : type r. Axis.t list -> r Index.t -> Axis.t list =
 fun acc -> function
  | Index.Add (a, b) -> index_axes (index_axes acc a) b
  | Index.Assume_position a -> index_axes acc a
  | Index.Ceil_div_pos (a, _) -> index_axes acc a
  | Index.Clamp_low a -> index_axes acc a
  | Index.Const _ -> acc
  | Index.Data (_, c, _) -> Coord.fold (fun acc i -> index_axes acc i) acc c
  | Index.Floor_div_pos (a, _) -> index_axes acc a
  | Index.Max (a, b) -> index_axes (index_axes acc a) b
  | Index.Min (a, b) -> index_axes (index_axes acc a) b
  | Index.Of_position a -> index_axes acc a
  | Index.Output a -> if List.mem a acc then acc else a :: acc
  | Index.Reduce _ -> acc
  | Index.Scale (_, a) -> index_axes acc a
  | Index.Zero -> acc

let rec index_assume_sites : type r. int -> r Index.t -> int =
 fun acc -> function
  | Index.Add (a, b) -> index_assume_sites (index_assume_sites acc a) b
  | Index.Assume_position a -> index_assume_sites (acc + 1) a
  | Index.Ceil_div_pos (a, _) -> index_assume_sites acc a
  | Index.Clamp_low a -> index_assume_sites acc a
  | Index.Const _ -> acc
  | Index.Data (_, c, _) ->
      Coord.fold (fun acc i -> index_assume_sites acc i) acc c
  | Index.Floor_div_pos (a, _) -> index_assume_sites acc a
  | Index.Max (a, b) -> index_assume_sites (index_assume_sites acc a) b
  | Index.Min (a, b) -> index_assume_sites (index_assume_sites acc a) b
  | Index.Of_position a -> index_assume_sites acc a
  | Index.Output _ -> acc
  | Index.Reduce _ -> acc
  | Index.Scale (_, a) -> index_assume_sites acc a
  | Index.Zero -> acc

(* [Data]'s own source, plus whatever further [Data] nodes are nested inside
   its coordinate (an ordinary [Role.Position.t Index.t] recursion, per
   [Index.Data]'s self-recursive shape). Unlike [index_reducers]/[index_axes]
   above, this is not threaded through [Fold.walk]'s no-op [no_index] --
   [sources] passes it as a real [idx_fn] below, since a [Data] embedded in a
   [Value.Load]'s coordinate would otherwise be invisible to a traversal that
   is supposed to answer "every source this expression depends on". *)
let rec index_sources : type r. Source.Set.t -> r Index.t -> Source.Set.t =
 fun acc -> function
  | Index.Add (a, b) -> index_sources (index_sources acc a) b
  | Index.Assume_position a -> index_sources acc a
  | Index.Ceil_div_pos (a, _) -> index_sources acc a
  | Index.Clamp_low a -> index_sources acc a
  | Index.Const _ -> acc
  | Index.Data (s, c, _) ->
      Coord.fold (fun acc i -> index_sources acc i) (Source.Set.add s acc) c
  | Index.Floor_div_pos (a, _) -> index_sources acc a
  | Index.Max (a, b) -> index_sources (index_sources acc a) b
  | Index.Min (a, b) -> index_sources (index_sources acc a) b
  | Index.Of_position a -> index_sources acc a
  | Index.Output _ -> acc
  | Index.Reduce _ -> acc
  | Index.Scale (_, a) -> index_sources acc a
  | Index.Zero -> acc

(* The index callback has to be RANK-2: a [Load]'s coordinate components are
     [Role.Position.t Index.t] while a reduction's upper bound is
     [Role.Delta.t Index.t], and an ordinary function argument would be fixed at
     whichever the inference engine saw first. Hence the record with an
     explicitly quantified field. *)
type 'acc idx_fn = { idx : 'r. 'acc -> 'r Index.t -> 'acc }

(* Generic bottom-up walk over a value, threading an accumulator. Every
     traversal below is written in terms of it, so a new [Value] constructor is
     handled in exactly one place. Reduction bounds are visited as indices and
     the body as a value; the BINDER is not interpreted here -- callers that care
     about scope (free variables, [Check]) handle it themselves. *)
let rec walk ~value ~index ~intrinsic acc (e : Value.t) =
  let acc = value acc e in
  let recur = walk ~value ~index ~intrinsic in
  match e with
  | Value.Binary (_, a, b) -> recur (recur acc a) b
  | Value.Const _ -> acc
  | Value.Intrinsic i ->
      let acc = intrinsic acc i in
      let (Intrinsic.Max_pool d) = i in
      Coord.fold (fun acc x -> index.idx acc x) acc d.Intrinsic.Max_pool.out
  | Value.Local _ -> acc
  | Value.Local_at (_, i) -> index.idx acc i
  | Value.Local_scan_at (_, row, lane) -> index.idx (index.idx acc row) lane
  | Value.Load (_, c) -> Coord.fold (fun acc i -> index.idx acc i) acc c
  | Value.Reduce r ->
      let acc = index.idx (index.idx acc r.Reduction.lo) r.Reduction.hi in
      recur acc r.Reduction.body
  | Value.Round_f32 a -> recur acc a
  (* [init]/[update] are ordinary structural children for every query built on
     [walk]: a source, load or intrinsic reached through them is a real
     dependency regardless of [lane]/[step]/[prev]'s binder scope, which none
     of these queries interpret anyway (see the header comment). Scope-aware
     queries ([free_reducers], [binders], the locals family) do not use
     [walk] and mask it themselves. *)
  | Value.Scan_at (s, row, lane) ->
      let acc = recur acc s.Scan.init in
      let acc = recur acc s.Scan.update in
      index.idx (index.idx acc row) lane
  | Value.Select (c, a, b) ->
      let acc =
        match c with
        | Bool.Index_eq (x, y) -> index.idx (index.idx acc x) y
        | Bool.Value_lt (x, y) -> recur (recur acc x) y
      in
      recur (recur acc a) b
  | Value.Unary (_, a) -> recur acc a
  | Value.Value_of_index i -> index.idx acc i

let nothing acc _ = acc
let no_index = { idx = (fun acc _ -> acc) }

(* ONE metered traversal, computing both measures and carrying both budgets.
     [Check]'s limits exist to reject an oversized tree, so measuring first and
     comparing after would exhaust the stack on exactly the input the limit is
     there to refuse. But a walk per limit is not enough either: whichever runs
     first still descends the full input whenever its OWN bound is loose, so a
     loose size limit defeats a tight depth limit, and swapping the order
     defeats the dual case. Carrying both on one walk bounds the recursion by
     the TIGHTER of the two.

     Index trees are metered too. They are where a load's addressing lives, so a
     limit that treated them as leaves would bound nothing useful: a single
     [Value_of_index] can carry an arbitrarily deep affine expression.

     [size] and [depth] are this same walk with both budgets at [max_int], which
     never trip — so there is exactly one description of what counts as a node
     and what counts as a level. *)
exception Over of [ `Depth | `Size ]

(* The node budget is THREADED, not held in a ref: every traversal takes the
     count still available and returns what it left, so a sibling is measured
     against what its predecessor consumed rather than against a shared cell.
     Each function returns [(depth, left)]. *)
let measure_with_locals ~local ~max_size ~max_depth e =
  (* Charged once per node, before descending: that is what keeps the
       recursion inside the budget rather than merely reporting on it. *)
  let node budget left ~cost ~depth =
    if cost <= 0 || depth <= 0 then
      invalid_arg "Expr.Fold.measure_with_locals: non-positive local measure";
    if budget < depth then raise_notrace (Over `Depth);
    if left < cost then raise_notrace (Over `Size);
    left - cost
  in
  let rec index : type r. int -> int -> r Index.t -> int * int =
   fun budget left i ->
    let left = node budget left ~cost:1 ~depth:1 in
    let sub = budget - 1 in
    let one a =
      let d, left = index sub left a in
      (1 + d, left)
    in
    match i with
    | Index.Add (a, b) ->
        let da, left = index sub left a in
        let db, left = index sub left b in
        (1 + Stdlib.max da db, left)
    | Index.Assume_position a -> one a
    | Index.Ceil_div_pos (a, _) -> one a
    | Index.Clamp_low a -> one a
    | Index.Const _ -> (1, left)
    | Index.Data (_, c, _) ->
        let dmax, left =
          Coord.fold
            (fun (m, left) x ->
              let d, left = index sub left x in
              (Stdlib.max m d, left))
            (0, left) c
        in
        (1 + dmax, left)
    | Index.Floor_div_pos (a, _) -> one a
    | Index.Max (a, b) ->
        let da, left = index sub left a in
        let db, left = index sub left b in
        (1 + Stdlib.max da db, left)
    | Index.Min (a, b) ->
        let da, left = index sub left a in
        let db, left = index sub left b in
        (1 + Stdlib.max da db, left)
    (* [Of_position]'s operand is a position, unlike the delta operands in the
       surrounding unary cases. *)
    | Index.Of_position a -> one a
    | Index.Output _ -> (1, left)
    | Index.Reduce _ -> (1, left)
    | Index.Scale (_, a) -> one a
    | Index.Zero -> (1, left)
  in
  let coord budget left c =
    Coord.fold
      (fun (m, left) i ->
        let d, left = index budget left i in
        (Stdlib.max m d, left))
      (0, left) c
  in
  (* [bound] is the enclosing scan(s)' own [prev] binder(s) -- a [Local_at]
     occurrence of one is a plain bound-variable reference, not a Region
     local, and must not reach [local] (which has no entry for it and, in
     [Region_program.specialize_pixel]'s caller, would raise). Same
     bound-tracking convention as [scoped_locals]/[keep]: the callback isn't
     told to skip anything, the traversal simply never calls it for a bound
     id. *)
  let rec value bound budget left (e : Value.t) =
    let local_size, local_depth =
      match e with
      | (Value.Local v | Value.Local_at (v, _) | Value.Local_scan_at (v, _, _))
        when not (Local_var.Set.mem v bound) ->
          local v
      | _ -> (1, 1)
    in
    let left = node budget left ~cost:local_size ~depth:local_depth in
    let sub = budget - 1 in
    match e with
    | Value.Binary (_, a, b) ->
        let da, left = value bound sub left a in
        let db, left = value bound sub left b in
        (1 + Stdlib.max da db, left)
    | Value.Const _ -> (1, left)
    | Value.Intrinsic (Intrinsic.Max_pool d) ->
        let dc, left = coord sub left d.Intrinsic.Max_pool.out in
        (1 + dc, left)
    | Value.Local _ -> (local_depth, left)
    | Value.Local_at (_, i) ->
        let d, left = index sub left i in
        (1 + Stdlib.max local_depth d, left)
    | Value.Local_scan_at (_, row, lane) ->
        let dr, left = index sub left row in
        let dl, left = index sub left lane in
        (1 + Stdlib.max local_depth (Stdlib.max dr dl), left)
    | Value.Load (_, c) ->
        let d, left = coord sub left c in
        (1 + d, left)
    | Value.Reduce r ->
        let dlo, left = index sub left r.Reduction.lo in
        let dhi, left = index sub left r.Reduction.hi in
        let dbody, left = value bound sub left r.Reduction.body in
        (1 + Stdlib.max (Stdlib.max dlo dhi) dbody, left)
    (* Both [init] and [update] are real embedded subtrees, so both are
       measured -- an inline [Scan_at] is what specialization turns a cached
       [Local_scan_at] read into, and undercounting it here would let a
       program past the size/depth budget it exists to enforce. [update]
       alone adds [s.Scan.prev] to [bound], matching [prev]'s own scope: free
       in [init], bound in [update]. *)
    | Value.Scan_at (s, row, lane) ->
        let dr, left = index sub left row in
        let dl, left = index sub left lane in
        let dinit, left = value bound sub left s.Scan.init in
        let dupdate, left =
          value (Local_var.Set.add s.Scan.prev bound) sub left s.Scan.update
        in
        (1 + Stdlib.max (Stdlib.max dr dl) (Stdlib.max dinit dupdate), left)
    | Value.Round_f32 a ->
        let d, left = value bound sub left a in
        (1 + d, left)
    | Value.Select (c, a, b) ->
        let g, left =
          match c with
          | Bool.Index_eq (x, y) ->
              let dx, left = index sub left x in
              let dy, left = index sub left y in
              (Stdlib.max dx dy, left)
          | Bool.Value_lt (x, y) ->
              let dx, left = value bound sub left x in
              let dy, left = value bound sub left y in
              (Stdlib.max dx dy, left)
        in
        let da, left = value bound sub left a in
        let db, left = value bound sub left b in
        (1 + Stdlib.max g (Stdlib.max da db), left)
    | Value.Unary (_, a) ->
        let d, left = value bound sub left a in
        (1 + d, left)
    | Value.Value_of_index i ->
        let d, left = index sub left i in
        (1 + d, left)
  in
  let d, left = value Local_var.Set.empty max_depth max_size e in
  (max_size - left, d)

let measure ~max_size ~max_depth e =
  measure_with_locals ~local:(fun _ -> (1, 1)) ~max_size ~max_depth e

let unmetered e = measure ~max_size:Stdlib.max_int ~max_depth:Stdlib.max_int e
let size e = fst (unmetered e)
let depth e = snd (unmetered e)

(* Which limit was passed, without measuring the rest. Both are enforced
     together for the reason above, so an absent limit is [max_int] rather than
     a skipped budget. *)
let exceeds_with_locals ~local ~max_size ~max_depth e =
  match measure_with_locals ~local ~max_size ~max_depth e with
  | _ -> None
  | exception Over w -> Some w

let exceeds ~max_size ~max_depth e =
  exceeds_with_locals ~local:(fun _ -> (1, 1)) ~max_size ~max_depth e

let sources e =
  walk
    ~value:(fun acc -> function
      | Value.Load (s, _) -> Source.Set.add s acc
      | Value.Intrinsic (Intrinsic.Max_pool d) ->
          Source.Set.add d.Intrinsic.Max_pool.source acc
      | _ -> acc)
    ~index:{ idx = index_sources } ~intrinsic:nothing Source.Set.empty e

(* Ordinary [Load] SITES, with their coordinates, in lexical order and with
     repeats. [sources] answers a different question and cannot serve here: it
     is a set, so it loses both multiplicity and addressing, and it folds in the
     source of an intrinsic descriptor — which is a real dependency but not a
     substitutable load. A consumer deciding what it may inline needs exactly
     this list; one deciding what must be resolved and ordered needs [sources]. *)
let loads e =
  List.rev
    (walk
       ~value:(fun acc -> function
         | Value.Load (s, c) -> (s, c) :: acc | _ -> acc)
       ~index:no_index ~intrinsic:nothing [] e)

let intrinsic_sources e =
  List.rev
    (walk
       ~value:(fun acc -> function
         | Value.Intrinsic (Intrinsic.Max_pool d) ->
             d.Intrinsic.Max_pool.source :: acc
         | _ -> acc)
       ~index:no_index ~intrinsic:nothing [] e)

type local_ref =
  | Scalar_ref of Local_var.t
  | Vector_ref of Local_var.t
  | Scan_ref of Local_var.t

(* Scope-aware, unlike most of [walk]'s consumers: [prev] is bound within its
   own scan's [update], so an occurrence there is not a free/declared-local
   reference the way it would be anywhere else. Shared by [locals],
   [scalar_locals], [vector_locals] and [scan_locals] below -- they differ
   only in which node kind [f] keeps, matching the pre-scan code's shape of
   one [walk] callback per query. *)
let rec scoped_locals ~f bound acc (e : Value.t) =
  let go = scoped_locals ~f bound in
  match e with
  | Value.Const _ | Value.Value_of_index _ | Value.Load _ | Value.Intrinsic _ ->
      acc
  | Value.Local v -> f bound acc (Scalar_ref v)
  | Value.Local_at (v, _) -> f bound acc (Vector_ref v)
  | Value.Local_scan_at (v, _, _) -> f bound acc (Scan_ref v)
  | Value.Binary (_, a, b) -> go (go acc a) b
  | Value.Unary (_, a) | Value.Round_f32 a -> go acc a
  | Value.Select (c, a, b) ->
      let acc =
        match c with
        | Bool.Value_lt (x, y) -> go (go acc x) y
        | Bool.Index_eq _ -> acc
      in
      go (go acc a) b
  | Value.Reduce r -> go acc r.Reduction.body
  | Value.Scan_at (s, _, _) ->
      let acc = scoped_locals ~f bound acc s.Scan.init in
      scoped_locals ~f (Local_var.Set.add s.Scan.prev bound) acc s.Scan.update

(* [keep bound acc v] adds [v] unless the scope traversal found it bound
   (a [prev] occurrence within its own scan's [update]). Each query below
   picks which node kind(s) to keep and ignores the rest, matching the
   pre-scan code's shape of one [walk] callback per query. *)
let keep bound acc v =
  if Local_var.Set.mem v bound then acc else Local_var.Set.add v acc

let locals e =
  scoped_locals
    ~f:(fun bound acc -> function
      | Scalar_ref v | Vector_ref v | Scan_ref v -> keep bound acc v)
    Local_var.Set.empty Local_var.Set.empty e

(* Split by NODE KIND, not merged into [locals]: the host's shape-agreement
   rule (a [Local] on a vector-shaped local, a [Local_at] on a scalar-shaped
   one, or either on a trace, is a typed error) needs to know WHICH form
   referenced a given id, and a single set that unions all three loses
   exactly that. *)
let scalar_locals e =
  scoped_locals
    ~f:(fun bound acc -> function
      | Scalar_ref v -> keep bound acc v | Vector_ref _ | Scan_ref _ -> acc)
    Local_var.Set.empty Local_var.Set.empty e

let vector_locals e =
  scoped_locals
    ~f:(fun bound acc -> function
      | Vector_ref v -> keep bound acc v | Scalar_ref _ | Scan_ref _ -> acc)
    Local_var.Set.empty Local_var.Set.empty e

let scan_locals e =
  scoped_locals
    ~f:(fun bound acc -> function
      | Scan_ref v -> keep bound acc v | Scalar_ref _ | Vector_ref _ -> acc)
    Local_var.Set.empty Local_var.Set.empty e

(* Saturating, not wrapping: this repository's 32-bit rule for
   js_of_ocaml-reachable code (this library is one) requires that an
   aggregate never silently wrap, and [scan_cost] is a cost ESTIMATE, not a
   value anything stores or replays, so capping at [Int64.max_int] is exactly
   as sound as failing outright while needing no error channel here -- the
   caller compares the result against a configured limit and every limit sits
   far below this ceiling. *)
let sat_add_i64 a b =
  if Int64.compare a (Int64.sub Int64.max_int b) > 0 then Int64.max_int
  else Int64.add a b

let sat_mul_i64 a b =
  if Int64.equal a 0L || Int64.equal b 0L then 0L
  else if Int64.compare a (Int64.div Int64.max_int b) > 0 then Int64.max_int
  else Int64.mul a b

(* [(updates, state)]: the lane-update count and peak live scan state one
   evaluation of [e] costs through the standalone inline evaluator
   ([Eval.value]'s [Scan_at] arm). Deliberately does NOT multiply through an
   enclosing [Reduce]'s extent -- [Scan_admission.check] is the complementary,
   reduction-aware guard for a scan actually composed under one; every scan
   this measure targets sits at Region-local top level, per the scan design
   record's own census. Since neither [Role.Position.t Index.t] nor
   [Role.Delta.t Index.t] can embed a [value] (the two languages are not
   mutually recursive), a [Scan_at]/[Local_scan_at] read's row/lane/step
   arguments and a [Reduce]'s bounds can never hide a scan, so every other
   node contributes only its children's cost. *)
let rec scan_cost (e : Value.t) : int64 * int =
  match e with
  | Value.Const _ | Value.Intrinsic _ | Value.Load _ | Value.Local _
  | Value.Local_at _ | Value.Local_scan_at _ | Value.Value_of_index _ ->
      (0L, 0)
  | Value.Unary (_, a) | Value.Round_f32 a -> scan_cost a
  | Value.Reduce r -> scan_cost r.Reduction.body
  | Value.Binary (_, a, b) ->
      let ua, sa = scan_cost a and ub, sb = scan_cost b in
      (sat_add_i64 ua ub, Stdlib.max sa sb)
  | Value.Select (c, a, b) ->
      let uc, sc =
        match c with
        | Bool.Value_lt (x, y) ->
            let ux, sx = scan_cost x and uy, sy = scan_cost y in
            (sat_add_i64 ux uy, Stdlib.max sx sy)
        | Bool.Index_eq _ -> (0L, 0)
      in
      let ua, sa = scan_cost a and ub, sb = scan_cost b in
      (sat_add_i64 uc (sat_add_i64 ua ub), Stdlib.max sc (Stdlib.max sa sb))
  | Value.Scan_at (s, _, _) ->
      let u_init, s_init = scan_cost s.Scan.init in
      let u_update, s_update = scan_cost s.Scan.update in
      let width = Int64.of_int s.Scan.width
      and steps = Int64.of_int s.Scan.steps in
      let updates =
        sat_add_i64 (sat_mul_i64 width u_init)
          (sat_mul_i64 steps (sat_mul_i64 width (Int64.add 1L u_update)))
      in
      let state = (2 * s.Scan.width) + Stdlib.max s_init s_update in
      (updates, state)

let output_axes e =
  walk ~value:nothing ~index:{ idx = index_axes } ~intrinsic:nothing [] e
  |> List.sort Axis.compare

let assume_sites e =
  walk ~value:nothing ~index:{ idx = index_assume_sites } ~intrinsic:nothing 0 e

let intrinsics e =
  walk ~value:nothing ~index:no_index ~intrinsic:(fun n _ -> n + 1) 0 e

(* Scope-aware, unlike the queries above: a reducer mentioned under its own
     binder is bound, not free. A well-formed top-level expression has none. *)
let free_reducers e =
  let rec go bound acc (e : Value.t) =
    let idx acc i =
      Reduce_var.Set.diff (index_reducers Reduce_var.Set.empty i) bound
      |> Reduce_var.Set.union acc
    in
    match e with
    | Value.Binary (_, a, b) -> go bound (go bound acc a) b
    | Value.Const _ -> acc
    | Value.Intrinsic (Intrinsic.Max_pool d) ->
        Coord.fold idx acc d.Intrinsic.Max_pool.out
    | Value.Local _ -> acc
    | Value.Local_at (_, i) -> idx acc i
    | Value.Local_scan_at (_, row, lane) -> idx (idx acc row) lane
    | Value.Load (_, c) -> Coord.fold idx acc c
    | Value.Reduce r ->
        (* The bounds are OUTSIDE the binder: they may mention enclosing
             reducers but not this one. *)
        let acc = idx (idx acc r.Reduction.lo) r.Reduction.hi in
        go (Reduce_var.Set.add r.Reduction.var bound) acc r.Reduction.body
    | Value.Round_f32 a -> go bound acc a
    (* [row]/[lane] (the READ site) sit OUTSIDE both scopes, like a
       reduction's bounds. [lane] is bound in [init]; [lane] and [step] are
       both bound in [update] -- two SIBLING scopes, so [lane] is added to
       [bound] independently for each. *)
    | Value.Scan_at (s, row, lane) ->
        let acc = idx (idx acc row) lane in
        let acc = go (Reduce_var.Set.add s.Scan.lane bound) acc s.Scan.init in
        go
          (Reduce_var.Set.add s.Scan.lane
             (Reduce_var.Set.add s.Scan.step bound))
          acc s.Scan.update
    | Value.Select (c, a, b) ->
        let acc =
          match c with
          | Bool.Index_eq (x, y) -> idx (idx acc x) y
          | Bool.Value_lt (x, y) -> go bound (go bound acc x) y
        in
        go bound (go bound acc a) b
    | Value.Unary (_, a) -> go bound acc a
    | Value.Value_of_index i -> idx acc i
  in
  go Reduce_var.Set.empty Reduce_var.Set.empty e

(* Binders in lexical (pre-)order, with repeats: an identity bound in two
     sibling scopes appears twice, which is what makes this usable for counting
     binders as distinct from counting identities. Inspection only -- [Pp] and
     the structural comparison each carry their own SCOPED environment, because
     a list keyed by identity cannot distinguish those siblings. *)
let binders e =
  let rec go acc (e : Value.t) =
    match e with
    | Value.Binary (_, a, b) -> go (go acc a) b
    | Value.Const _ -> acc
    | Value.Intrinsic _ -> acc
    | Value.Local _ -> acc
    | Value.Local_at _ -> acc
    | Value.Local_scan_at _ -> acc
    | Value.Load _ -> acc
    | Value.Reduce r -> go (r.Reduction.var :: acc) r.Reduction.body
    | Value.Round_f32 a -> go acc a
    (* [lane] once for [init]'s scope, then [lane] again and [step] for
       [update]'s -- two sibling scopes, reported in that order, matching
       [Reduce]'s "named before descending" lexical convention. *)
    | Value.Scan_at (s, _, _) ->
        let acc = go (s.Scan.lane :: acc) s.Scan.init in
        go (s.Scan.step :: s.Scan.lane :: acc) s.Scan.update
    | Value.Select (c, a, b) ->
        let acc =
          match c with
          | Bool.Index_eq _ -> acc
          | Bool.Value_lt (x, y) -> go (go acc x) y
        in
        go (go acc a) b
    | Value.Unary (_, a) -> go acc a
    | Value.Value_of_index _ -> acc
  in
  List.rev (go [] e)

(* [Fold.binders]'s local-namespace sibling: only [prev] is ever a local
   binder, introduced once per [Scan_at], for [update]'s scope. *)
let local_binders e =
  let rec go acc (e : Value.t) =
    match e with
    | Value.Binary (_, a, b) -> go (go acc a) b
    | Value.Const _ -> acc
    | Value.Intrinsic _ -> acc
    | Value.Local _ -> acc
    | Value.Local_at _ -> acc
    | Value.Local_scan_at _ -> acc
    | Value.Load _ -> acc
    | Value.Reduce r -> go acc r.Reduction.body
    | Value.Round_f32 a -> go acc a
    | Value.Scan_at (s, _, _) ->
        let acc = go acc s.Scan.init in
        go (s.Scan.prev :: acc) s.Scan.update
    | Value.Select (c, a, b) ->
        let acc =
          match c with
          | Bool.Index_eq _ -> acc
          | Bool.Value_lt (x, y) -> go (go acc x) y
        in
        go (go acc a) b
    | Value.Unary (_, a) -> go acc a
    | Value.Value_of_index _ -> acc
  in
  List.rev (go [] e)

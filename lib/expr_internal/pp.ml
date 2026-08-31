(* Reducer display names are supplied by the caller rather than derived from
     the opaque identity, so output never depends on which ordinals a supply
     handed out. [Pp.value] threads the lexical r0, r1, ... assignment in stage
     4; here the naming function is explicit, with no default to fall back on.

     Every form below renders exactly as the representation it replaces, which
     is what keeps the migration's golden diff attributable to a single cause:
     [Zero] as [0] (it was [Index_const 0]), [Of_position] and [Assume_position]
     transparently (both were the identity), and [Clamp_low] as [max(0,_)] (it
     was [Index_max (Index_const 0, _)]). *)
let index ~names fmt e =
  let rec go : type r. Format.formatter -> r Index.t -> unit =
   fun fmt -> function
     | Index.Add (a, b) -> Fmt.pf fmt "%a+%a" go a go b
     | Index.Assume_position a -> go fmt a
     | Index.Ceil_div_pos (a, d) -> Fmt.pf fmt "ceil_div(%a,%d)" go a d
     | Index.Clamp_low a -> Fmt.pf fmt "max(0,%a)" go a
     | Index.Const n -> Fmt.int fmt n
     | Index.Data (s, c, extent) ->
         Fmt.pf fmt "data(%a[%a],%d)" Source.pp s (Coord.pp go) c extent
     | Index.Floor_div_pos (a, d) -> Fmt.pf fmt "floor_div(%a,%d)" go a d
     | Index.Max (a, b) -> Fmt.pf fmt "max(%a,%a)" go a go b
     | Index.Min (a, b) -> Fmt.pf fmt "min(%a,%a)" go a go b
     | Index.Of_position i -> go fmt i
     | Index.Output a -> Axis.pp fmt a
     | Index.Reduce v -> Fmt.string fmt (names v)
     | Index.Scale (k, a) -> Fmt.pf fmt "%d*%a" k go a
     | Index.Zero -> Fmt.int fmt 0
  in
  go fmt e

(* Display names are assigned r1, r2, ... in LEXICAL order, so output is
     stable across expressions that differ only in which ordinals their supply
     handed out -- two structurally identical formulas built by independent
     builders print identically. *)
(* Numbered from 1, not 0. The representation being replaced numbered reducers
     from an allocation counter that incremented BEFORE building the body, so its
     order was already lexical and only the base differed. Matching it means the
     migration moves no goldens at all, which turns "did the cutover change
     behaviour?" into a question the diff answers by itself. Renumbering from 0
     is a separate, labelled change if it is ever wanted. *)

(* The naming environment is SCOPED, not a global identity-to-name map. Two
     independently built fragments can bind the same ordinal in sibling scopes
     -- well-scoped, and [Check] accepts it, since neither shadows the other --
     and one map keyed by identity would then give both binders whichever name
     came last, and would render a free reference in one sibling as bound
     because its twin binds that identity. So the environment descends with the
     tree and the counter runs across it, which is what makes the names lexical
     rather than merely distinct. *)
let names_in env v =
  match Reduce_var.Map.find_opt v env with
  | Some n -> n
  | None -> Fmt.str "?%a" Reduce_var.pp v

(* Renders exactly as the representation being replaced, so the migration's
     golden diff stays attributable to a single cause. [Round_f32] is the one
     genuinely new form and has no old spelling to preserve. *)
(* The display counter is RETURNED, not held in a ref. A top-down environment
     cannot number siblings lexically -- the left sibling's count has to reach
     the right one -- so [at] hands back the next number, and every arm with
     more than one value child sequences its children explicitly instead of
     putting several [%a] in one format string.

     That covers more than the reductions: [Binary]'s operands, [Select]'s guard
     AND both branches, and inside the guard [Bool.Value_lt]'s two operands, any
     of which can contain a binder. ([Bool.Index_eq] holds indices, which bind
     nothing.) Dropping a hand-off is silent -- two binders would share a name
     -- so the goldens are the guard. *)
let value_open ~names fmt e =
  let idx env fmt i = index ~names:(names_in env) fmt i in
  let rec at env n fmt (e : Value.t) =
    (* Eta-expanded so it stays polymorphic in the role: a reduction's [lo] is
         a position and its [hi] a delta. *)
    let idxe fmt i = idx env fmt i in
    match e with
    | Value.Binary (op, a, b) ->
        Fmt.pf fmt "(";
        let n = at env n fmt a in
        Fmt.pf fmt " %s " (Value.binary_sym op);
        let n = at env n fmt b in
        Fmt.pf fmt ")";
        n
    | Value.Const x ->
        Fmt.float fmt x;
        n
    | Value.Intrinsic (Intrinsic.Max_pool d) ->
        Fmt.pf fmt "max_pool2d_%s(%a; k=%dx%d s=%dx%d p=%dx%d; out=[%a])"
          (Intrinsic.Max_pool.result_name d.Intrinsic.Max_pool.result)
          Source.pp d.Intrinsic.Max_pool.source d.Intrinsic.Max_pool.kernel_h
          d.Intrinsic.Max_pool.kernel_w d.Intrinsic.Max_pool.stride_h
          d.Intrinsic.Max_pool.stride_w d.Intrinsic.Max_pool.pad_h
          d.Intrinsic.Max_pool.pad_w (Coord.pp idxe) d.Intrinsic.Max_pool.out;
        n
    | Value.Local v ->
        Fmt.string fmt
          (match names v with
          | Some name -> name
          | None -> Fmt.str "?%a" Local_var.pp v);
        n
    | Value.Load (s, c) ->
        Fmt.pf fmt "%a[%a]" Source.pp s (Coord.pp idxe) c;
        n
    | Value.Reduce r ->
        (* Named before descending, so the counter follows lexical order --
             and the bounds print under the OUTER environment, since they are
             evaluated outside the binder and cannot mention it. *)
        let name = Fmt.str "r%d" n in
        let inner = Reduce_var.Map.add r.Reduction.var name env in
        Fmt.pf fmt "%s(%s=%a..%a: "
          (Reduction.kind_name r.Reduction.kind)
          name idxe r.Reduction.lo idxe r.Reduction.hi;
        let n = at inner (n + 1) fmt r.Reduction.body in
        Fmt.pf fmt ")";
        n
    | Value.Round_f32 a ->
        Fmt.pf fmt "f32(";
        let n = at env n fmt a in
        Fmt.pf fmt ")";
        n
    | Value.Select (c, a, b) ->
        Fmt.pf fmt "select(";
        let n = guard_at env n fmt c in
        Fmt.pf fmt ", ";
        let n = at env n fmt a in
        Fmt.pf fmt ", ";
        let n = at env n fmt b in
        Fmt.pf fmt ")";
        n
    | Value.Unary (op, a) ->
        Fmt.pf fmt "%s(" (Value.unary_name op);
        let n = at env n fmt a in
        Fmt.pf fmt ")";
        n
    | Value.Value_of_index i ->
        Fmt.pf fmt "value_of_index(%a)" idxe i;
        n
  and guard_at env n fmt = function
    | Bool.Index_eq (a, b) ->
        Fmt.pf fmt "(%a = %a)" (idx env) a (idx env) b;
        n
    | Bool.Value_lt (a, b) ->
        Fmt.pf fmt "(";
        let n = at env n fmt a in
        Fmt.pf fmt " < ";
        let n = at env n fmt b in
        Fmt.pf fmt ")";
        n
  in
  ignore (at Reduce_var.Map.empty 1 fmt e : int)

let value = value_open ~names:(fun _ -> None)

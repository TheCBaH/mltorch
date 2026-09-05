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
let local_name lenv names v =
  match Local_var.Map.find_opt v lenv with
  | Some name -> name
  | None -> (
      match names v with
      | Some name -> name
      | None -> Fmt.str "?%a" Local_var.pp v)

let idx env fmt i = index ~names:(names_in env) fmt i

(* [lenv] is [Pp]'s local-namespace sibling of [env]: it names only [prev]
   binders, scoped to their own scan's [update], so a name it holds always
   wins over [names]'s external lookup for that occurrence. Elsewhere it is
   empty and every [Local]/[Local_at] renders exactly as before scan existed.

   [at] and [scan_body] are mutually recursive rather than [scan_body] living
   inside [value_open]'s closure: [scan_open] below needs the SAME rendering
   -- an unspecialized scan's [init]/[update], with proper [lane]/[step]/
   [prev] naming -- without [at]'s [Value.Scan_at] case's trailing
   [row,lane] projection, which is fabricated for any caller that has no
   real read site (see the scan design record). *)
let rec at ~names env lenv n fmt (e : Value.t) =
  (* Eta-expanded so it stays polymorphic in the role: a reduction's [lo] is
       a position and its [hi] a delta. *)
  let idxe fmt i = idx env fmt i in
  match e with
  | Value.Binary (op, a, b) ->
      Fmt.pf fmt "(";
      let n = at ~names env lenv n fmt a in
      Fmt.pf fmt " %s " (Value.binary_sym op);
      let n = at ~names env lenv n fmt b in
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
  | Value.Local_at (v, i) ->
      Fmt.pf fmt "%s[%a]" (local_name lenv names v) idxe i;
      n
  | Value.Local_scan_at (v, row, lane) ->
      Fmt.pf fmt "%s@@[%a,%a]" (local_name lenv names v) idxe row idxe lane;
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
      let n = at ~names inner lenv (n + 1) fmt r.Reduction.body in
      Fmt.pf fmt ")";
      n
  | Value.Round_f32 a ->
      Fmt.pf fmt "f32(";
      let n = at ~names env lenv n fmt a in
      Fmt.pf fmt ")";
      n
  | Value.Scan_at (s, row, lane) ->
      let n = scan_body ~names env lenv n fmt s in
      Fmt.pf fmt "@@[%a,%a]" idxe row idxe lane;
      n
  | Value.Select (c, a, b) ->
      Fmt.pf fmt "select(";
      let n = guard_at ~names env lenv n fmt c in
      Fmt.pf fmt ", ";
      let n = at ~names env lenv n fmt a in
      Fmt.pf fmt ", ";
      let n = at ~names env lenv n fmt b in
      Fmt.pf fmt ")";
      n
  | Value.Unary (op, a) ->
      Fmt.pf fmt "%s(" (Value.unary_name op);
      let n = at ~names env lenv n fmt a in
      Fmt.pf fmt ")";
      n
  | Value.Value_of_index i ->
      Fmt.pf fmt "value_of_index(%a)" idxe i;
      n

(* The shared body of an unspecialized scan: [init]/[update], scoped and
   named exactly as a real [Value.Scan_at] read renders them, but with no
   trailing projection -- there is no row/lane to show for a plain
   declaration. [lane] gets a fresh display name for EACH sibling scope
   ([init] and [update]) even though it is one identity -- the same
   lexical-naming rule [Reduce] already applies, extended to a binder
   appearing twice. *)
and scan_body ~names env lenv n fmt (s : Scan.t) =
  let lane1 = Fmt.str "r%d" n in
  Fmt.pf fmt "scan[w=%d,s=%d](init[%s]=" s.Scan.width s.Scan.steps lane1;
  let n =
    at ~names
      (Reduce_var.Map.add s.Scan.lane lane1 env)
      lenv (n + 1) fmt s.Scan.init
  in
  let lane2 = Fmt.str "r%d" n and step = Fmt.str "r%d" (n + 1) in
  let prev = Fmt.str "p%d" (n + 1) in
  Fmt.pf fmt "; update[%s,%s,%s]=" lane2 step prev;
  let n =
    at ~names
      (env
      |> Reduce_var.Map.add s.Scan.lane lane2
      |> Reduce_var.Map.add s.Scan.step step)
      (Local_var.Map.add s.Scan.prev prev lenv)
      (n + 2) fmt s.Scan.update
  in
  Fmt.pf fmt ")";
  n

and guard_at ~names env lenv n fmt = function
  | Bool.Index_eq (a, b) ->
      Fmt.pf fmt "(%a = %a)" (idx env) a (idx env) b;
      n
  | Bool.Value_lt (a, b) ->
      Fmt.pf fmt "(";
      let n = at ~names env lenv n fmt a in
      Fmt.pf fmt " < ";
      let n = at ~names env lenv n fmt b in
      Fmt.pf fmt ")";
      n

let value_open ~names fmt e =
  ignore (at ~names Reduce_var.Map.empty Local_var.Map.empty 1 fmt e : int)

let scan_open ~names fmt (s : Scan.t) =
  ignore
    (scan_body ~names Reduce_var.Map.empty Local_var.Map.empty 1 fmt s : int)

let value = value_open ~names:(fun _ -> None)
let scan = scan_open ~names:(fun _ -> None)

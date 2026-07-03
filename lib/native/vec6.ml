(* See vec6.mli. Representation is a record (fields n/t/d/h/w/c; [private] in
   the .mli — readable everywhere, constructible only here), not an abstract
   array: axis count is fixed and hot-path callers know their axis
   statically, so they can read a component as a direct field projection
   instead of going through [get]. The [role] parameter is phantom. *)

type 'd t = { n : 'd; t : 'd; d : 'd; h : 'd; w : 'd; c : 'd }
type shape = Dim.extent Dim.t t
type coord = Dim.index Dim.t t
type deltas = Dim.delta Dim.t t

let get (v : 'd t) (a : Axis.t) : 'd =
  match a with N -> v.n | T -> v.t | D -> v.d | H -> v.h | W -> v.w | C -> v.c

let set (v : 'd t) (a : Axis.t) (x : 'd) : 'd t =
  match a with
  | N -> { v with n = x }
  | T -> { v with t = x }
  | D -> { v with d = x }
  | H -> { v with h = x }
  | W -> { v with w = x }
  | C -> { v with c = x }

(* Per-axis setters, new value first / vec last (not [set]'s vec-first order):
   that's what makes [base |> set_h h' |> set_w w' |> set_c c'] actually
   pipe — [set_h h'] partially applies to a ['d t -> 'd t] transformer ready
   to receive the piped vec, whereas [set]'s order would need an explicit
   [fun v -> set v Axis.H h'] at each step. For op code that only overrides
   one or two axes of an existing coordinate (e.g. a base tensor's axes with
   a couple of axes replaced), this reads as a straight-line pipeline instead
   of a closure or listing all 6 fields via [make]. *)
let set_n x (v : 'd t) : 'd t = { v with n = x }
let set_t x (v : 'd t) : 'd t = { v with t = x }
let set_d x (v : 'd t) : 'd t = { v with d = x }
let set_h x (v : 'd t) : 'd t = { v with h = x }
let set_w x (v : 'd t) : 'd t = { v with w = x }
let set_c x (v : 'd t) : 'd t = { v with c = x }

(* Copies [src]'s value at [axis] into the same [axis] of the piped vec — one
   call and one axis dispatch (each branch both reads [src] and writes [v]),
   not the two dispatches [set v axis (get src axis)] would do. [src] is
   positional, not [~src]-labeled: with only one axis role there's nothing
   to disambiguate, and it keeps [src |> copy_axis ...] symmetric with
   [set_h]/[set_c]'s "pre-filled args first, piped vec last" pipe shape. For
   op code overriding some of [v]'s axes from another vec at the same axis
   (the common case — see [copy] below for the rarer cross-axis one). *)
let copy_axis (src : 'd t) (axis : Axis.t) (v : 'd t) : 'd t =
  match axis with
  | N -> { v with n = src.n }
  | T -> { v with t = src.t }
  | D -> { v with d = src.d }
  | H -> { v with h = src.h }
  | W -> { v with w = src.w }
  | C -> { v with c = src.c }

(* General form: the source vec's value at axis [~src] into axis [~dst] of
   the piped vec — two axis dispatches, since the axes can differ. The
   source vec is positional (see [copy_axis]); [~src]/[~dst] the labels
   belong to the two axis roles, which are what actually need disambiguating
   at a call site. *)
let copy (source : 'd t) ~(src : Axis.t) ~(dst : Axis.t) (v : 'd t) : 'd t =
  set v dst (get source src)

let map (f : 'a -> 'b) (v : 'a t) : 'b t =
  { n = f v.n; t = f v.t; d = f v.d; h = f v.h; w = f v.w; c = f v.c }

let mapi (f : Axis.t -> 'a -> 'b) (v : 'a t) : 'b t =
  {
    n = f N v.n;
    t = f T v.t;
    d = f D v.d;
    h = f H v.h;
    w = f W v.w;
    c = f C v.c;
  }

(* Fold order matches [Axis.all]/[get]'s N/T/D/H/W/C convention. *)
let fold (f : 'acc -> 'a -> 'acc) (acc : 'acc) (v : 'a t) : 'acc =
  f (f (f (f (f (f acc v.n) v.t) v.d) v.h) v.w) v.c

let foldi (f : 'acc -> Axis.t -> 'a -> 'acc) (acc : 'acc) (v : 'a t) : 'acc =
  f (f (f (f (f (f acc N v.n) T v.t) D v.d) H v.h) W v.w) C v.c

(* Materializes an [Axis.t -> 'd] function into a ['d t] by calling it once
   per axis — for callers holding a per-axis function (e.g. a [SEMANTICS]
   op's base output coordinate) that now need a [Vec6.t] to pipe through
   [set_h] etc. or pass to [SEMANTICS.load]. *)
let of_fn (f : Axis.t -> 'd) : 'd t =
  { n = f N; t = f T; d = f D; h = f H; w = f W; c = f C }

(* Unchecked, for any payload type — the general-purpose "6 named slots"
   constructor. [shape]/[coord]/[deltas] below are the checked, [Dim.t]-role
   form (going through [Dim.extent]/[Dim.index]/[Dim.delta]); a caller with a
   payload that isn't a [Dim.t] role at all (e.g. a symbolic sub-expression —
   see [Expr.Load]) has nothing to check, so uses this directly. *)
let make ~n ~t ~d ~h ~w ~c = { n; t; d; h; w; c }

let shape ~n ~t ~d ~h ~w ~c =
  make ~n:(Dim.extent n) ~t:(Dim.extent t) ~d:(Dim.extent d) ~h:(Dim.extent h)
    ~w:(Dim.extent w) ~c:(Dim.extent c)

let coord ~n ~t ~d ~h ~w ~c =
  make ~n:(Dim.index n) ~t:(Dim.index t) ~d:(Dim.index d) ~h:(Dim.index h)
    ~w:(Dim.index w) ~c:(Dim.index c)

let deltas ~n ~t ~d ~h ~w ~c =
  make ~n:(Dim.delta n) ~t:(Dim.delta t) ~d:(Dim.delta d) ~h:(Dim.delta h)
    ~w:(Dim.delta w) ~c:(Dim.delta c)

let origin = coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0

let numel (s : shape) =
  List.fold_left (fun acc a -> Dim.( *@ ) acc (get s a)) Dim.one_count Axis.all

(* Unrolled over the 6 fixed fields instead of folding a closure over
   [Axis.all]: this is the innermost per-element tensor-read path, called
   millions of times for a real conv, and profiling (memtrace, then landmarks
   for real cycle counts) showed the fold's closure allocation, and later
   [get]'s [Axis.to_int] dispatch, were substantial shares of total time
   there — see .ai/pt2_inference_perf.md. *)
let offset (s : shape) (c : coord) =
  let acc = Dim.lin Dim.zero_offset s.n c.n in
  let acc = Dim.lin acc s.t c.t in
  let acc = Dim.lin acc s.d c.d in
  let acc = Dim.lin acc s.h c.h in
  let acc = Dim.lin acc s.w c.w in
  Dim.lin acc s.c c.c

let in_bounds (s : shape) (c : coord) =
  (* indices are >= 0 by construction (Dim.index); just check the upper bound *)
  Dim.to_int c.n < Dim.to_int s.n
  && Dim.to_int c.t < Dim.to_int s.t
  && Dim.to_int c.d < Dim.to_int s.d
  && Dim.to_int c.h < Dim.to_int s.h
  && Dim.to_int c.w < Dim.to_int s.w
  && Dim.to_int c.c < Dim.to_int s.c

(* [in_bounds]+[offset] fused, and taking the 6 components directly instead
   of a [coord]: [Tensor.read_at] (the innermost per-element path) computes
   its index from an [Axis.t -> Dim.index Dim.t] closure and only ever needs
   the final linear offset, so building a [coord] just to hand it straight
   to [in_bounds]/[offset] would allocate a record for no reason on every one
   of the ~232M calls a real conv makes here. Components are already
   [Dim.index Dim.t] (validated non-negative by construction — see
   direct.ml, where [position index = Dim.index Dim.t] and every producer of
   one goes through [Dim.index]/[Dim.succ]), so only the upper bound is
   checked here — no re-validating what the caller already guarantees.
   Returns [-1] (not [Dim.offset Dim.t option]) for a point that isn't fully
   inside [s]: a valid offset is always >= 0, so it's an unambiguous,
   allocation-free sentinel — see .ai/pt2_inference_perf.md for why this
   exists as a separate function instead of composing
   [coord]+[in_bounds]+[offset]. *)
let offset_of (s : shape) ~(n : Dim.index Dim.t) ~(t : Dim.index Dim.t)
    ~(d : Dim.index Dim.t) ~(h : Dim.index Dim.t) ~(w : Dim.index Dim.t)
    ~(c : Dim.index Dim.t) =
  let n = (n :> int) and t = (t :> int) and d = (d :> int) in
  let h = (h :> int) and w = (w :> int) and c = (c :> int) in
  let en = Dim.to_int s.n in
  let et = Dim.to_int s.t in
  let ed = Dim.to_int s.d in
  let eh = Dim.to_int s.h in
  let ew = Dim.to_int s.w in
  let ec = Dim.to_int s.c in
  if n < en && t < et && d < ed && h < eh && w < ew && c < ec then
    let acc = (((((((n * et) + t) * ed) + d) * eh) + h) * ew) + w in
    (acc * ec) + c
  else -1

let iter (s : shape) (f : coord -> unit) =
  let en = Dim.to_int s.n in
  let et = Dim.to_int s.t in
  let ed = Dim.to_int s.d in
  let eh = Dim.to_int s.h in
  let ew = Dim.to_int s.w in
  let ec = Dim.to_int s.c in
  for n = 0 to en - 1 do
    for t = 0 to et - 1 do
      for d = 0 to ed - 1 do
        for h = 0 to eh - 1 do
          for w = 0 to ew - 1 do
            for c = 0 to ec - 1 do
              f (coord ~n ~t ~d ~h ~w ~c)
            done
          done
        done
      done
    done
  done

let drop_leading pred axes =
  let rec aux = function
    | [] -> [ List.nth axes (List.length axes - 1) ]
    | a :: rest -> if pred a then aux rest else a :: rest
  in
  aux axes

let pp_axis_binding get ppf a = Fmt.pf ppf "%a=%a" Axis.pp a Dim.pp (get a)

let pp_shape fmt (s : shape) =
  let axes = drop_leading (fun a -> Dim.to_int (get s a) = 1) Axis.all in
  Fmt.brackets (Fmt.list ~sep:Fmt.sp (pp_axis_binding (get s))) fmt axes

(* Encoded as a 6-element array [n, t, d, h, w, c]. *)
let shape_jsont : shape Jsont.t =
  Jsont.map ~kind:"shape"
    ~dec:(fun xs ->
      match xs with
      | [ n; t; d; h; w; c ] -> shape ~n ~t ~d ~h ~w ~c
      | _ ->
          Jsont.Error.msgf Jsont.Meta.none
            "shape: expected 6-element array, got %d" (List.length xs))
    ~enc:(fun s -> List.map (fun a -> (get s a :> int)) Axis.all)
    (Jsont.list Jsont.int)

let pp_coord fmt (c : coord) =
  let axes = drop_leading (fun a -> Dim.to_int (get c a) = 0) Axis.all in
  Fmt.parens (Fmt.list ~sep:(Fmt.any ",") Dim.pp) fmt (List.map (get c) axes)

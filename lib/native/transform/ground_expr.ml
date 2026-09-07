(* See ground_expr.mli. *)

module Origin = struct
  type t =
    | Boundary of Cluster_var.t
    | Capture of Const_ssa.Capture.t
    | Dst of Tensor_id.t
    | Src of Tensor_id.t

  let rank = function
    | Boundary _ -> 0
    | Capture _ -> 1
    | Dst _ -> 2
    | Src _ -> 3

  let compare a b =
    match (a, b) with
    | Boundary x, Boundary y -> Cluster_var.compare x y
    | Capture x, Capture y -> Const_ssa.Capture.compare x y
    | Dst x, Dst y | Src x, Src y -> Tensor_id.compare x y
    | _ -> Int.compare (rank a) (rank b)

  let equal a b = compare a b = 0

  (* The raw edge this stands for, where there is one. A boundary variable names
     no single edge — that is the point of it, since the two graphs number the
     one value differently — which is why this is an option rather than a total
     projection, and why a boundary cell has no stage to expand. *)
  let edge = function
    | Dst id | Src id -> Some id
    | Boundary _ | Capture _ -> None

  let pp fmt = function
    | Boundary v -> Cluster_var.pp fmt v
    | Capture capture -> Fmt.pf fmt "capture.%a" Const_ssa.Capture.pp capture
    | Dst id -> Fmt.pf fmt "dst.%a" Tensor_id.pp id
    | Src id -> Fmt.pf fmt "src.%a" Tensor_id.pp id

  module Map = Map.Make (struct
    type nonrec t = t

    let compare = compare
  end)
end

module Cell = struct
  type t = { coord : Vec6.coord; origin : Origin.t }

  let compare a b =
    match Origin.compare a.origin b.origin with
    | 0 -> Stdlib.compare a.coord b.coord
    | n -> n

  let pp fmt c = Fmt.pf fmt "%a%a" Origin.pp c.origin Vec6.pp_coord c.coord

  module Set = Set.Make (struct
    type nonrec t = t

    let compare = compare
  end)

  module Map = Map.Make (struct
    type nonrec t = t

    let compare = compare
  end)
end

module Valuation = struct
  type t = float Cell.Map.t

  let empty = Cell.Map.empty
  let find t cell = Cell.Map.find cell t
  let of_list l = List.to_seq l |> Cell.Map.of_seq

  let pp fmt t =
    Fmt.pf fmt "@[<h>{%a}@]"
      (Fmt.list ~sep:Fmt.comma (fun fmt (cell, v) ->
           Fmt.pf fmt "%a=%h" Cell.pp cell v))
      (Cell.Map.bindings t)

  (* The whole cell, not a digest of it. A weighted sum of the coordinate
     collides — with weights 7 on W and 1 on C, (W=1,C=0) and (W=0,C=7) land on
     the same value — and an indexing error that swapped exactly those two cells
     would then be invisible to every draw, since the later ones keyed off that
     same sum. *)
  (* The ORIGIN, not a raw id: a boundary variable and an edge that happen to
     share a number are different cells, and so are the two graphs' edges that
     do, and keying off the number alone would have them draw the same value. *)
  let origin_key (o : Origin.t) =
    match o with
    | Origin.Boundary v -> (0, Cluster_var.to_int v)
    | Origin.Capture capture ->
        (1, Hashtbl.hash (Const_ssa.Capture.to_string capture))
    | Origin.Dst id -> (2, Tensor_id.to_int id)
    | Origin.Src id -> (3, Tensor_id.to_int id)

  let key (c : Cell.t) =
    let axis a = Dim.to_int (Vec6.get c.coord a) in
    ( origin_key c.origin,
      axis Axis.N,
      axis Axis.T,
      axis Axis.D,
      axis Axis.H,
      axis Axis.W,
      axis Axis.C )

  let pseudo_random n (c : Cell.t) =
    let h = Hashtbl.hash (n, key c) in
    let v = float_of_int ((h mod 1999) - 999) /. 250. in
    if Float.equal v 0. then 1. else v

  (* Draw 0 numbers the cells 1, 2, 3, … in [Cell.Set] order, which is
     collision-free by construction rather than by choice of weights, and still
     deterministic — the set order is [Cell.compare], not insertion order. A
     distinct value per cell is what makes an index swap visible at all. *)
  let draw n cells =
    if n = 0 then
      snd
        (Cell.Set.fold
           (fun c (i, acc) -> (i + 1, Cell.Map.add c (float_of_int i) acc))
           cells (1, Cell.Map.empty))
    else
      Cell.Set.fold
        (fun c acc -> Cell.Map.add c (pseudo_random n c) acc)
        cells Cell.Map.empty
end

(* ---- the hash-consed DAG ---------------------------------------------------

   A node's children always have a strictly smaller id than the node itself —
   the arena is append-only and every constructor requires its children to
   already exist — so every traversal below is a plain iterative walk (an
   explicit heap-allocated stack, never OCaml call-stack recursion) rather than
   a recursive descent that a deep or wide DAG could blow. *)

type node =
  | NBinary of Expr.Value.binary_op * int * int
  | NCell of Cell.t
  | NConst of float
  | NMax of Expr.Max_op.t * int * int
  | NRound of int
  | NSelect of int (* guard id *) * int * int
  | NUnary of Expr.Value.unary_op * int

type gnode = GLt of int * int | GPoolBetter of int * int

type key =
  | KBinary of Expr.Value.binary_op * int * int
  | KCell of Cell.t
  | KConst of int64 (* exact bits, never the float itself *)
  | KMax of Expr.Max_op.t * int * int
  | KRound of int
  | KSelect of int * int * int
  | KUnary of Expr.Value.unary_op * int

type gkey = KLt of int * int | KPoolBetter of int * int

let tag_of = function
  | NBinary _ -> 0
  | NCell _ -> 1
  | NConst _ -> 2
  | NMax _ -> 3
  | NRound _ -> 4
  | NSelect _ -> 5
  | NUnary _ -> 6

module Arena = struct
  type t = {
    token : int;
    mutable nodes : node array;
    mutable digests : int array;
    mutable n_count : int;
    intern : (key, int) Hashtbl.t;
    mutable gnodes : gnode array;
    mutable gdigests : int array;
    mutable g_count : int;
    gintern : (gkey, int) Hashtbl.t;
  }

  let next_token = ref 0

  let create () =
    incr next_token;
    {
      token = !next_token;
      nodes = Array.make 16 (NConst 0.);
      digests = Array.make 16 0;
      n_count = 0;
      intern = Hashtbl.create 64;
      gnodes = Array.make 4 (GLt (0, 0));
      gdigests = Array.make 4 0;
      g_count = 0;
      gintern = Hashtbl.create 16;
    }
end

type t = { arena : Arena.t; id : int }
type guard = { garena : Arena.t; gid : int }

(* [Sys.max_array_length] under js_of_ocaml/node (see project_todo.md step 4's
   own probe): every arena array is bounded by it regardless of host, and no
   verifier budget in this repository comes remotely close, so hitting this is
   a caller bug (an unbounded construction loop with no meter), not a resource
   outcome to report through [Err.t]. *)
let hard_cap = 536_870_911

let digest_key (arena : Arena.t) (k : key) =
  let d id = arena.digests.(id) in
  match k with
  | KBinary (op, a, b) -> Hashtbl.hash (0, op, d a, d b)
  | KCell c -> Hashtbl.hash (1, c)
  | KConst bits -> Hashtbl.hash (2, bits)
  | KMax (op, a, b) -> Hashtbl.hash (3, op, d a, d b)
  | KRound a -> Hashtbl.hash (4, d a)
  | KSelect (g, a, b) -> Hashtbl.hash (5, arena.gdigests.(g), d a, d b)
  | KUnary (op, a) -> Hashtbl.hash (6, op, d a)

let gdigest_key (arena : Arena.t) (k : gkey) =
  let d id = arena.digests.(id) in
  match k with
  | KLt (a, b) -> Hashtbl.hash (10, d a, d b)
  | KPoolBetter (a, b) -> Hashtbl.hash (11, d a, d b)

let ensure_v_capacity (arena : Arena.t) needed =
  let cap = Array.length arena.Arena.nodes in
  if needed > cap then begin
    let new_cap =
      Stdlib.min hard_cap (Stdlib.max needed (Stdlib.max 16 (cap * 2)))
    in
    let nodes' = Array.make new_cap (NConst 0.) in
    Array.blit arena.Arena.nodes 0 nodes' 0 arena.Arena.n_count;
    arena.Arena.nodes <- nodes';
    let digests' = Array.make new_cap 0 in
    Array.blit arena.Arena.digests 0 digests' 0 arena.Arena.n_count;
    arena.Arena.digests <- digests'
  end

let ensure_g_capacity (arena : Arena.t) needed =
  let cap = Array.length arena.Arena.gnodes in
  if needed > cap then begin
    let new_cap =
      Stdlib.min hard_cap (Stdlib.max needed (Stdlib.max 16 (cap * 2)))
    in
    let gnodes' = Array.make new_cap (GLt (0, 0)) in
    Array.blit arena.Arena.gnodes 0 gnodes' 0 arena.Arena.g_count;
    arena.Arena.gnodes <- gnodes';
    let gdigests' = Array.make new_cap 0 in
    Array.blit arena.Arena.gdigests 0 gdigests' 0 arena.Arena.g_count;
    arena.Arena.gdigests <- gdigests'
  end

let intern (arena : Arena.t) (k : key) (mk : unit -> node) : int =
  match Hashtbl.find_opt arena.Arena.intern k with
  | Some id -> id
  | None ->
      if arena.Arena.n_count >= hard_cap then
        invalid_arg "Ground_expr.Arena: too many nodes";
      let dig = digest_key arena k in
      ensure_v_capacity arena (arena.Arena.n_count + 1);
      let id = arena.Arena.n_count in
      arena.Arena.nodes.(id) <- mk ();
      arena.Arena.digests.(id) <- dig;
      arena.Arena.n_count <- id + 1;
      Hashtbl.add arena.Arena.intern k id;
      id

let intern_g (arena : Arena.t) (k : gkey) (mk : unit -> gnode) : int =
  match Hashtbl.find_opt arena.Arena.gintern k with
  | Some id -> id
  | None ->
      if arena.Arena.g_count >= hard_cap then
        invalid_arg "Ground_expr.Arena: too many guard nodes";
      let dig = gdigest_key arena k in
      ensure_g_capacity arena (arena.Arena.g_count + 1);
      let id = arena.Arena.g_count in
      arena.Arena.gnodes.(id) <- mk ();
      arena.Arena.gdigests.(id) <- dig;
      arena.Arena.g_count <- id + 1;
      Hashtbl.add arena.Arena.gintern k id;
      id

let check_owner (arena : Arena.t) (x : t) =
  if x.arena != arena then
    invalid_arg "Ground_expr: node belongs to a different arena"

let check_owner_g (arena : Arena.t) (g : guard) =
  if g.garena != arena then
    invalid_arg "Ground_expr: guard belongs to a different arena"

(* ---- smart constructors ----------------------------------------------------- *)

let binary arena op (x : t) (y : t) : t =
  check_owner arena x;
  check_owner arena y;
  {
    arena;
    id =
      intern arena
        (KBinary (op, x.id, y.id))
        (fun () -> NBinary (op, x.id, y.id));
  }

let cell arena (c : Cell.t) : t =
  { arena; id = intern arena (KCell c) (fun () -> NCell c) }

let const arena (v : float) : t =
  let bits = Int64.bits_of_float v in
  { arena; id = intern arena (KConst bits) (fun () -> NConst v) }

let max arena op (x : t) (y : t) : t =
  check_owner arena x;
  check_owner arena y;
  {
    arena;
    id = intern arena (KMax (op, x.id, y.id)) (fun () -> NMax (op, x.id, y.id));
  }

let round arena (x : t) : t =
  check_owner arena x;
  { arena; id = intern arena (KRound x.id) (fun () -> NRound x.id) }

let select arena (g : guard) (x : t) (y : t) : t =
  check_owner arena x;
  check_owner arena y;
  check_owner_g arena g;
  {
    arena;
    id =
      intern arena
        (KSelect (g.gid, x.id, y.id))
        (fun () -> NSelect (g.gid, x.id, y.id));
  }

let unary arena op (x : t) : t =
  check_owner arena x;
  { arena; id = intern arena (KUnary (op, x.id)) (fun () -> NUnary (op, x.id)) }

let lt arena (x : t) (y : t) : guard =
  check_owner arena x;
  check_owner arena y;
  {
    garena = arena;
    gid = intern_g arena (KLt (x.id, y.id)) (fun () -> GLt (x.id, y.id));
  }

let pool_better arena ~best ~value : guard =
  check_owner arena best;
  check_owner arena value;
  {
    garena = arena;
    gid =
      intern_g arena
        (KPoolBetter (best.id, value.id))
        (fun () -> GPoolBetter (best.id, value.id));
  }

(* ---- views ------------------------------------------------------------------- *)

type view =
  | Binary of Expr.Value.binary_op * t * t
  | Cell of Cell.t
  | Const of float
  | Max of Expr.Max_op.t * t * t
  | Round of t
  | Select of guard * t * t
  | Unary of Expr.Value.unary_op * t

type guard_view = Lt of t * t | Pool_better of { best : t; value : t }

let out (x : t) : view =
  let v arena id = { arena; id } in
  match x.arena.Arena.nodes.(x.id) with
  | NBinary (op, a, b) -> Binary (op, v x.arena a, v x.arena b)
  | NCell c -> Cell c
  | NConst c -> Const c
  | NMax (op, a, b) -> Max (op, v x.arena a, v x.arena b)
  | NRound a -> Round (v x.arena a)
  | NSelect (g, a, b) ->
      Select ({ garena = x.arena; gid = g }, v x.arena a, v x.arena b)
  | NUnary (op, a) -> Unary (op, v x.arena a)

let guard_out (g : guard) : guard_view =
  let v arena id = { arena; id } in
  match g.garena.Arena.gnodes.(g.gid) with
  | GLt (a, b) -> Lt (v g.garena a, v g.garena b)
  | GPoolBetter (best, value) ->
      Pool_better { best = v g.garena best; value = v g.garena value }

let arena (x : t) = x.arena

(* ---- comparison, equality, hashing -------------------------------------------

   [hash]/[hash_guard] are O(1): every node's digest is computed bottom-up at
   construction (see [digest_key]/[gdigest_key]) and merely read back here.
   [compare]/[equal] recurse structurally with per-call pair memoisation, so a
   shared node compared against the same counterpart more than once (a diamond,
   a repeated accumulator) is decided once. *)

let hash (x : t) = x.arena.Arena.digests.(x.id)
let hash_guard (g : guard) = g.garena.Arena.gdigests.(g.gid)

let lex3 first second third =
  if first <> 0 then first
  else
    let s = second () in
    if s <> 0 then s else third ()

(* Exact bits, not [Float.compare]: [Identical] is a claim about bits, and the
   ordinary comparisons equate -0. with +0. and NaN with NaN. *)
let compare_const = Core.Float_bits.compare_exact

let make_comparators () =
  let memo : (int * int * int * int, int) Hashtbl.t = Hashtbl.create 64 in
  let gmemo : (int * int * int * int, int) Hashtbl.t = Hashtbl.create 16 in
  let rec cmp (x : t) (y : t) : int =
    if x.arena == y.arena && x.id = y.id then 0
    else
      let k = (x.arena.Arena.token, x.id, y.arena.Arena.token, y.id) in
      match Hashtbl.find_opt memo k with
      | Some r -> r
      | None ->
          let r = cmp_nodes x y in
          Hashtbl.add memo k r;
          r
  and cmp_nodes (x : t) (y : t) : int =
    let mkx id = { arena = x.arena; id } and mky id = { arena = y.arena; id } in
    let nx = x.arena.Arena.nodes.(x.id) and ny = y.arena.Arena.nodes.(y.id) in
    match (nx, ny) with
    | NBinary (o1, a1, b1), NBinary (o2, a2, b2) ->
        lex3 (Stdlib.compare o1 o2)
          (fun () -> cmp (mkx a1) (mky a2))
          (fun () -> cmp (mkx b1) (mky b2))
    | NCell c1, NCell c2 -> Cell.compare c1 c2
    | NConst v1, NConst v2 -> compare_const v1 v2
    | NMax (o1, a1, b1), NMax (o2, a2, b2) ->
        lex3 (Stdlib.compare o1 o2)
          (fun () -> cmp (mkx a1) (mky a2))
          (fun () -> cmp (mkx b1) (mky b2))
    | NRound a1, NRound a2 -> cmp (mkx a1) (mky a2)
    | NSelect (g1, a1, b1), NSelect (g2, a2, b2) ->
        lex3
          (cmp_guard
             { garena = x.arena; gid = g1 }
             { garena = y.arena; gid = g2 })
          (fun () -> cmp (mkx a1) (mky a2))
          (fun () -> cmp (mkx b1) (mky b2))
    | NUnary (o1, a1), NUnary (o2, a2) ->
        lex3 (Stdlib.compare o1 o2)
          (fun () -> cmp (mkx a1) (mky a2))
          (fun () -> 0)
    | _ -> Stdlib.compare (tag_of nx) (tag_of ny)
  and cmp_guard (gx : guard) (gy : guard) : int =
    if gx.garena == gy.garena && gx.gid = gy.gid then 0
    else
      let k = (gx.garena.Arena.token, gx.gid, gy.garena.Arena.token, gy.gid) in
      match Hashtbl.find_opt gmemo k with
      | Some r -> r
      | None ->
          let r = cmp_guard_nodes gx gy in
          Hashtbl.add gmemo k r;
          r
  and cmp_guard_nodes (gx : guard) (gy : guard) : int =
    let mkx id = { arena = gx.garena; id }
    and mky id = { arena = gy.garena; id } in
    match
      (gx.garena.Arena.gnodes.(gx.gid), gy.garena.Arena.gnodes.(gy.gid))
    with
    | GLt (a1, b1), GLt (a2, b2) ->
        lex3
          (cmp (mkx a1) (mky a2))
          (fun () -> cmp (mkx b1) (mky b2))
          (fun () -> 0)
    | GPoolBetter (b1, v1), GPoolBetter (b2, v2) ->
        lex3
          (cmp (mkx b1) (mky b2))
          (fun () -> cmp (mkx v1) (mky v2))
          (fun () -> 0)
    | GLt _, GPoolBetter _ -> -1
    | GPoolBetter _, GLt _ -> 1
  in
  (cmp, cmp_guard)

let compare a b =
  let cmp, _ = make_comparators () in
  cmp a b

let compare_guard a b =
  let _, cmp_guard = make_comparators () in
  cmp_guard a b

let equal a b = compare a b = 0
let equal_guard a b = compare_guard a b = 0

(* ---- reachability, size, cells ------------------------------------------------

   Both walks are plain iterative marking over an explicit stack — never OCaml
   call-stack recursion — exploiting that a node's children always have a
   strictly smaller id, so nothing here can be led into unbounded depth by a
   wide or deep DAG. *)

type item = V of int | G of int

let children_of (arena : Arena.t) = function
  | V id -> (
      match arena.Arena.nodes.(id) with
      | NBinary (_, a, b) | NMax (_, a, b) -> [ V a; V b ]
      | NCell _ | NConst _ -> []
      | NRound a | NUnary (_, a) -> [ V a ]
      | NSelect (g, a, b) -> [ G g; V a; V b ])
  | G g -> (
      match arena.Arena.gnodes.(g) with
      | GLt (a, b) -> [ V a; V b ]
      | GPoolBetter (a, b) -> [ V a; V b ])

let reachable (root : t) : bool array * bool array =
  let arena = root.arena in
  let vis = Array.make (Stdlib.max 1 arena.Arena.n_count) false in
  let gvis = Array.make (Stdlib.max 1 arena.Arena.g_count) false in
  let stack : item Stack.t = Stack.create () in
  Stack.push (V root.id) stack;
  while not (Stack.is_empty stack) do
    match Stack.pop stack with
    | V id ->
        if not vis.(id) then begin
          vis.(id) <- true;
          List.iter (fun c -> Stack.push c stack) (children_of arena (V id))
        end
    | G g ->
        if not gvis.(g) then begin
          gvis.(g) <- true;
          List.iter (fun c -> Stack.push c stack) (children_of arena (G g))
        end
  done;
  (vis, gvis)

let size (root : t) : int =
  let vis, _ = reachable root in
  Array.fold_left (fun acc b -> if b then acc + 1 else acc) 0 vis

let cells (root : t) : Cell.Set.t =
  let vis, _ = reachable root in
  let acc = ref Cell.Set.empty in
  Array.iteri
    (fun id ok ->
      if ok then
        match root.arena.Arena.nodes.(id) with
        | NCell c -> acc := Cell.Set.add c !acc
        | _ -> ())
    vis;
  !acc

(* Dependencies-before-dependents order over the reachable set, mixing value
   and guard items — the standard "push twice" iterative postorder for a DAG:
   an item popped a second time (marked [expanded]) is only then emitted, and
   an item already finished is skipped outright, so a node reachable from many
   parents is still expanded exactly once. *)
let postorder (root : t) : item list =
  let arena = root.arena in
  let done_v = Array.make (Stdlib.max 1 arena.Arena.n_count) false in
  let done_g = Array.make (Stdlib.max 1 arena.Arena.g_count) false in
  let is_done = function V id -> done_v.(id) | G g -> done_g.(g) in
  let mark_done = function
    | V id -> done_v.(id) <- true
    | G g -> done_g.(g) <- true
  in
  let order = ref [] in
  let stack : (item * bool) Stack.t = Stack.create () in
  Stack.push (V root.id, false) stack;
  while not (Stack.is_empty stack) do
    let item, expanded = Stack.pop stack in
    if is_done item then ()
    else if expanded then begin
      mark_done item;
      order := item :: !order
    end
    else begin
      Stack.push (item, true) stack;
      List.iter (fun c -> Stack.push (c, false) stack) (children_of arena item)
    end
  done;
  List.rev !order

(* ---- evaluation ---------------------------------------------------------------

   Memoised recursion, not a forward fold over every reachable node: [Select]
   must evaluate only the branch its guard picks, so an unselected branch may
   read a cell absent from [v] with no error — the same discipline
   [Expr.Eval.value] and the old unshared [eval] both followed. Memoisation is
   still what stops a shared subterm (read from many places) being evaluated
   more than once. *)

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let eval (root : t) (v : Valuation.t) : float =
  let arena = root.arena in
  let memo : float option array =
    Array.make (Stdlib.max 1 arena.Arena.n_count) None
  in
  let gmemo : bool option array =
    Array.make (Stdlib.max 1 arena.Arena.g_count) None
  in
  let rec go id =
    match memo.(id) with
    | Some r -> r
    | None ->
        let r =
          match arena.Arena.nodes.(id) with
          | NBinary (op, a, b) -> Expr.Value.apply_binary op (go a) (go b)
          | NCell c -> Valuation.find v c
          | NConst x -> x
          | NMax (op, a, b) -> Expr.Max_op.apply op (go a) (go b)
          | NRound a -> to_f32 (go a)
          | NSelect (g, a, b) -> if go_guard g then go a else go b
          | NUnary (op, a) -> Expr.Value.apply_unary op (go a)
        in
        memo.(id) <- Some r;
        r
  and go_guard g =
    match gmemo.(g) with
    | Some r -> r
    | None ->
        let r =
          match arena.Arena.gnodes.(g) with
          | GLt (a, b) -> go a < go b
          | GPoolBetter (best, value) ->
              Expr.Max_op.pool_better ~best:(go best) ~value:(go value)
        in
        gmemo.(g) <- Some r;
        r
  in
  go root.id

(* ---- printing -------------------------------------------------------------- *)

let max_name = function
  | Expr.Max_op.Float_max -> "fmax"
  | Expr.Max_op.Pool_max -> "pmax"

let pp fmt (root : t) =
  let arena = root.arena in
  let order = postorder root in
  let refs_v = Array.make (Stdlib.max 1 arena.Arena.n_count) 0 in
  let refs_g = Array.make (Stdlib.max 1 arena.Arena.g_count) 0 in
  List.iter
    (fun item ->
      List.iter
        (fun c ->
          match c with
          | V id -> refs_v.(id) <- refs_v.(id) + 1
          | G g -> refs_g.(g) <- refs_g.(g) + 1)
        (children_of arena item))
    order;
  let label = Array.make (Stdlib.max 1 arena.Arena.n_count) None in
  let glabel = Array.make (Stdlib.max 1 arena.Arena.g_count) None in
  let next = ref 0 in
  let fresh () =
    let n = !next in
    incr next;
    Printf.sprintf "l%d" n
  in
  let rec render_v id fmt =
    match label.(id) with
    | Some name -> Fmt.string fmt name
    | None -> render_v_body id fmt
  and render_v_body id fmt =
    match arena.Arena.nodes.(id) with
    | NBinary (op, a, b) ->
        Fmt.pf fmt "(%t %s %t)" (render_v a) (Expr.Value.binary_sym op)
          (render_v b)
    | NCell c -> Cell.pp fmt c
    | NConst x -> Fmt.pf fmt "%h" x
    | NMax (op, a, b) ->
        Fmt.pf fmt "%s(%t, %t)" (max_name op) (render_v a) (render_v b)
    | NRound a -> Fmt.pf fmt "f32(%t)" (render_v a)
    | NSelect (g, a, b) ->
        Fmt.pf fmt "select(%t, %t, %t)" (render_g g) (render_v a) (render_v b)
    | NUnary (op, a) ->
        Fmt.pf fmt "%s(%t)" (Expr.Value.unary_name op) (render_v a)
  and render_g g fmt =
    match glabel.(g) with
    | Some name -> Fmt.string fmt name
    | None -> render_g_body g fmt
  and render_g_body g fmt =
    match arena.Arena.gnodes.(g) with
    | GLt (a, b) -> Fmt.pf fmt "(%t < %t)" (render_v a) (render_v b)
    | GPoolBetter (a, b) ->
        Fmt.pf fmt "better(%t, %t)" (render_v a) (render_v b)
  in
  let defs = ref [] in
  List.iter
    (fun item ->
      match item with
      | V id -> (
          if refs_v.(id) >= 2 then
            match arena.Arena.nodes.(id) with
            | NCell _ | NConst _ -> ()
            | _ ->
                let name = fresh () in
                label.(id) <- Some name;
                defs := (name, render_v_body id) :: !defs)
      | G g ->
          if refs_g.(g) >= 2 then begin
            let name = fresh () in
            glabel.(g) <- Some name;
            defs := (name, render_g_body g) :: !defs
          end)
    order;
  match List.rev !defs with
  | [] -> render_v root.id fmt
  | defs ->
      List.iter
        (fun (name, render) -> Fmt.pf fmt "let %s = %t in@ " name render)
        defs;
      render_v root.id fmt

(* ---- structural rewrites: project, erase_rounds ----------------------------

   Both are unconditional postorder folds (every reachable node is rebuilt,
   unlike [eval]/[normalise]'s data-dependent [Select]), so the single
   postorder pass above is exactly the right shape: build each node once its
   children's rebuilt forms exist, memoised per original id via [out_v]/[out_g],
   into the target arena's own hash-consing. *)

let project ~into ~boundary (root : t) : t =
  let arena = root.arena in
  let order = postorder root in
  let out_v : t option array =
    Array.make (Stdlib.max 1 arena.Arena.n_count) None
  in
  let out_g : guard option array =
    Array.make (Stdlib.max 1 arena.Arena.g_count) None
  in
  let gv id = Option.get out_v.(id) in
  let gg g = Option.get out_g.(g) in
  List.iter
    (fun item ->
      match item with
      | V id ->
          let v =
            match arena.Arena.nodes.(id) with
            | NBinary (op, a, b) -> binary into op (gv a) (gv b)
            | NCell c -> (
                match boundary c.Cell.origin with
                | Some var ->
                    cell into { c with Cell.origin = Origin.Boundary var }
                | None -> cell into c)
            | NConst x -> const into x
            | NMax (op, a, b) -> max into op (gv a) (gv b)
            | NRound a -> round into (gv a)
            | NSelect (g, a, b) -> select into (gg g) (gv a) (gv b)
            | NUnary (op, a) -> unary into op (gv a)
          in
          out_v.(id) <- Some v
      | G g ->
          let v =
            match arena.Arena.gnodes.(g) with
            | GLt (a, b) -> lt into (gv a) (gv b)
            | GPoolBetter (a, b) -> pool_better into ~best:(gv a) ~value:(gv b)
          in
          out_g.(g) <- Some v)
    order;
  gv root.id

let erase_rounds ~into (root : t) : t =
  let arena = root.arena in
  let order = postorder root in
  let out_v : t option array =
    Array.make (Stdlib.max 1 arena.Arena.n_count) None
  in
  let out_g : guard option array =
    Array.make (Stdlib.max 1 arena.Arena.g_count) None
  in
  let gv id = Option.get out_v.(id) in
  let gg g = Option.get out_g.(g) in
  List.iter
    (fun item ->
      match item with
      | V id ->
          let v =
            match arena.Arena.nodes.(id) with
            | NBinary (op, a, b) -> binary into op (gv a) (gv b)
            | NCell c -> cell into c
            | NConst x -> const into x
            | NMax (op, a, b) -> max into op (gv a) (gv b)
            | NRound a -> gv a (* drop the wrapper *)
            | NSelect (g, a, b) -> select into (gg g) (gv a) (gv b)
            | NUnary (op, a) -> unary into op (gv a)
          in
          out_v.(id) <- Some v
      | G g ->
          let v =
            match arena.Arena.gnodes.(g) with
            | GLt (a, b) -> lt into (gv a) (gv b)
            | GPoolBetter (a, b) -> pool_better into ~best:(gv a) ~value:(gv b)
          in
          out_g.(g) <- Some v)
    order;
  gv root.id

(* ---- normalisation -----------------------------------------------------------

   Memoised recursion, for the same reason as [eval]: a closed [Select] guard
   must normalise only the taken branch (normalising both would collect
   [blocked] cells from a branch that cannot run), so this cannot be a uniform
   forward fold over every reachable node. [blocked] membership is intrinsic to
   a node (a function of it and [stored_f32] alone, never of which caller
   demanded it), so accumulating it in a plain mutable set alongside a memoised
   rebuild is sound regardless of how many different paths reach that node. *)

type normalised = { blocked : Cell.Set.t; expr : t }

let normalise ~into ~stored_f32 (root : t) : normalised =
  let arena = root.arena in
  let blocked = ref Cell.Set.empty in
  let memo : t option array =
    Array.make (Stdlib.max 1 arena.Arena.n_count) None
  in
  let gmemo : guard option array =
    Array.make (Stdlib.max 1 arena.Arena.g_count) None
  in
  let closed (x : t) = match out x with Const v -> Some v | _ -> None in
  let fold2 build op (a : t) (b : t) =
    match (closed a, closed b) with
    | Some x, Some y -> const into (op x y)
    | _ -> build a b
  in
  let rec go id : t =
    match memo.(id) with
    | Some r -> r
    | None ->
        let r =
          match arena.Arena.nodes.(id) with
          | NBinary (op, a, b) ->
              let a = go a and b = go b in
              fold2
                (fun a b -> binary into op a b)
                (Expr.Value.apply_binary op)
                a b
          | NCell c -> cell into c
          | NConst x -> const into x
          | NMax (op, a, b) ->
              let a = go a and b = go b in
              fold2 (fun a b -> max into op a b) (Expr.Max_op.apply op) a b
          | NRound a -> (
              let inner = go a in
              match out inner with
              | Cell c ->
                  if stored_f32 c then inner
                  else begin
                    blocked := Cell.Set.add c !blocked;
                    round into inner
                  end
              | Const v -> const into (to_f32 v)
              | Round _ -> inner
              | Binary _ | Max _ | Select _ | Unary _ -> round into inner)
          | NSelect (g, a, b) -> (
              let g' = go_guard g in
              match guard_value g' with
              | Some true -> go a
              | Some false -> go b
              | None -> select into g' (go a) (go b))
          | NUnary (op, a) -> (
              let a = go a in
              match closed a with
              | Some v -> const into (Expr.Value.apply_unary op v)
              | None -> unary into op a)
        in
        memo.(id) <- Some r;
        r
  and go_guard g : guard =
    match gmemo.(g) with
    | Some r -> r
    | None ->
        let r =
          match arena.Arena.gnodes.(g) with
          | GLt (a, b) -> lt into (go a) (go b)
          | GPoolBetter (a, b) -> pool_better into ~best:(go a) ~value:(go b)
        in
        gmemo.(g) <- Some r;
        r
  and guard_value (g : guard) : bool option =
    match guard_out g with
    | Lt (a, b) -> (
        match (closed a, closed b) with
        | Some x, Some y -> Some (x < y)
        | _ -> None)
    | Pool_better { best; value } -> (
        match (closed best, closed value) with
        | Some best, Some value -> Some (Expr.Max_op.pool_better ~best ~value)
        | _ -> None)
  in
  let expr = go root.id in
  { blocked = !blocked; expr }

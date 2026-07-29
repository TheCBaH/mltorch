(* See ground_expr.mli. *)

module Origin = struct
  type t =
    | Dst of Tensor_id.t
    | Input of Input_var.t
    | Shared of Tensor_id.t
    | Src of Tensor_id.t

  let rank = function Dst _ -> 0 | Input _ -> 1 | Shared _ -> 2 | Src _ -> 3

  let compare a b =
    match (a, b) with
    | Dst x, Dst y | Shared x, Shared y | Src x, Src y -> Tensor_id.compare x y
    | Input x, Input y -> Input_var.compare x y
    | _ -> Int.compare (rank a) (rank b)

  let equal a b = compare a b = 0

  (* The raw edge this stands for, where there is one. An input variable names
     no edge in either graph, which is why it has no stage and why this is an
     option rather than a total projection. *)
  let edge = function Dst id | Shared id | Src id -> Some id | Input _ -> None

  let pp fmt = function
    | Dst id -> Fmt.pf fmt "dst.%a" Tensor_id.pp id
    | Input v -> Input_var.pp fmt v
    | Shared id -> Tensor_id.pp fmt id
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
  (* The ORIGIN, not a raw id: an input variable and an edge that happen to
     share a number are different cells, and keying off the number alone would
     have them draw the same value. *)
  let origin_key (o : Origin.t) =
    match o with
    | Origin.Dst id -> (0, Tensor_id.to_int id)
    | Origin.Input v -> (1, Input_var.to_int v)
    | Origin.Shared id -> (2, Tensor_id.to_int id)
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

type guard = Lt of t * t | Pool_better of { best : t; value : t }

and t =
  | Binary of Expr.binary_op * t * t
  | Cell of Cell.t
  | Const of float
  | Max of Max_op.t * t * t
  | Round of t
  | Select of guard * t * t
  | Unary of Expr.unary_op * t

(* ---- f32 storage ---------------------------------------------------------- *)

(* The round-trip [Tensor.materialize] performs at every stage boundary. *)
let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* ---- comparison ----------------------------------------------------------- *)

(* Bits, not [Float.compare]: [Identical] is a claim about bits, and the
   ordinary comparisons equate -0. with +0. and NaN with NaN. *)
let compare_const a b =
  Int64.compare (Int64.bits_of_float a) (Int64.bits_of_float b)

(* Tags keep the constructor ordering explicit rather than depending on the
   declaration order of the variant. *)
let tag = function
  | Binary _ -> 0
  | Cell _ -> 1
  | Const _ -> 2
  | Max _ -> 3
  | Round _ -> 4
  | Select _ -> 5
  | Unary _ -> 6

let rec compare a b =
  match (a, b) with
  | Binary (o1, l1, r1), Binary (o2, l2, r2) ->
      lex3 (Stdlib.compare o1 o2)
        (fun () -> compare l1 l2)
        (fun () -> compare r1 r2)
  | Cell c1, Cell c2 -> Cell.compare c1 c2
  | Const x, Const y -> compare_const x y
  | Max (o1, l1, r1), Max (o2, l2, r2) ->
      lex3 (Stdlib.compare o1 o2)
        (fun () -> compare l1 l2)
        (fun () -> compare r1 r2)
  | Round x, Round y -> compare x y
  | Select (g1, l1, r1), Select (g2, l2, r2) ->
      lex3 (compare_guard g1 g2)
        (fun () -> compare l1 l2)
        (fun () -> compare r1 r2)
  | Unary (o1, x), Unary (o2, y) ->
      lex3 (Stdlib.compare o1 o2) (fun () -> compare x y) (fun () -> 0)
  | _ -> Stdlib.compare (tag a) (tag b)

and compare_guard g1 g2 =
  match (g1, g2) with
  | Lt (a1, b1), Lt (a2, b2) ->
      lex3 (compare a1 a2) (fun () -> compare b1 b2) (fun () -> 0)
  | Pool_better p1, Pool_better p2 ->
      lex3 (compare p1.best p2.best)
        (fun () -> compare p1.value p2.value)
        (fun () -> 0)
  | Lt _, Pool_better _ -> -1
  | Pool_better _, Lt _ -> 1

and lex3 first second third =
  if first <> 0 then first
  else
    let s = second () in
    if s <> 0 then s else third ()

let equal a b = compare a b = 0

(* ---- traversal ------------------------------------------------------------ *)

let rec cells = function
  | Binary (_, a, b) -> Cell.Set.union (cells a) (cells b)
  | Cell c -> Cell.Set.singleton c
  | Const _ -> Cell.Set.empty
  | Max (_, a, b) -> Cell.Set.union (cells a) (cells b)
  | Round x -> cells x
  | Select (g, a, b) ->
      Cell.Set.union (cells_guard g) (Cell.Set.union (cells a) (cells b))
  | Unary (_, x) -> cells x

and cells_guard = function
  | Lt (a, b) -> Cell.Set.union (cells a) (cells b)
  | Pool_better { best; value } -> Cell.Set.union (cells best) (cells value)

let rec size = function
  | Binary (_, a, b) -> 1 + size a + size b
  | Cell _ | Const _ -> 1
  | Max (_, a, b) -> 1 + size a + size b
  | Round x -> 1 + size x
  | Select (g, a, b) -> 1 + size_guard g + size a + size b
  | Unary (_, x) -> 1 + size x

and size_guard = function
  | Lt (a, b) -> size a + size b
  | Pool_better { best; value } -> size best + size value

let rec erase_rounds = function
  | Binary (op, a, b) -> Binary (op, erase_rounds a, erase_rounds b)
  | (Cell _ | Const _) as leaf -> leaf
  | Max (op, a, b) -> Max (op, erase_rounds a, erase_rounds b)
  | Round x -> erase_rounds x
  | Select (g, a, b) ->
      Select (erase_rounds_guard g, erase_rounds a, erase_rounds b)
  | Unary (op, x) -> Unary (op, erase_rounds x)

and erase_rounds_guard = function
  | Lt (a, b) -> Lt (erase_rounds a, erase_rounds b)
  | Pool_better { best; value } ->
      Pool_better { best = erase_rounds best; value = erase_rounds value }

(* ---- evaluation ----------------------------------------------------------- *)

let rec eval e v =
  match e with
  | Binary (op, a, b) -> Expr.apply_binary_op op (eval a v) (eval b v)
  | Cell c -> Valuation.find v c
  | Const x -> x
  | Max (op, a, b) -> Max_op.apply op (eval a v) (eval b v)
  | Round x -> to_f32 (eval x v)
  | Select (g, a, b) -> if eval_guard g v then eval a v else eval b v
  | Unary (op, x) -> Expr.apply_unary_op op (eval x v)

and eval_guard g v =
  match g with
  | Lt (a, b) -> Expr.apply_compare_op Expr.Lt (eval a v) (eval b v)
  | Pool_better { best; value } ->
      Max_op.pool_better ~best:(eval best v) ~value:(eval value v)

(* ---- printing ------------------------------------------------------------- *)

let rec pp fmt = function
  | Binary (op, a, b) ->
      Fmt.pf fmt "(%a %s %a)" pp a (Expr.binary_op_sym op) pp b
  | Cell c -> Cell.pp fmt c
  | Const x -> Fmt.pf fmt "%h" x
  | Max (op, a, b) ->
      Fmt.pf fmt "%s(%a, %a)"
        (match op with Max_op.Float_max -> "fmax" | Max_op.Pool_max -> "pmax")
        pp a pp b
  | Round x -> Fmt.pf fmt "f32(%a)" pp x
  | Select (g, a, b) -> Fmt.pf fmt "select(%a, %a, %a)" pp_guard g pp a pp b
  | Unary (op, x) -> Fmt.pf fmt "%s(%a)" (Expr.unary_op_name op) pp x

and pp_guard fmt = function
  | Lt (a, b) -> Fmt.pf fmt "(%a < %a)" pp a pp b
  | Pool_better { best; value } -> Fmt.pf fmt "better(%a, %a)" pp best pp value

(* ---- normalisation -------------------------------------------------------- *)

type normalised = { blocked : Cell.Set.t; expr : t }

(* Collapses the three [Round] rules AND folds closed arithmetic, threading the
   set of cells whose [Round] the first rule's side condition refused.

   Folding is what a bound constant is for: substituting payloads turns a
   constant sub-DAG's leaves into [Const], but only folding turns the arithmetic
   over them into the single number the destination edge carries. Because every
   subtree is folded bottom-up, "this subtree is closed" is exactly "it came
   back a [Const]", so no separate closedness analysis is needed.

   Folding evaluates through [eval], so it reproduces the engine's arithmetic
   including [Round]'s f32 step rather than an idealised real-arithmetic
   value. *)
let normalise ~stored_f32 e =
  let closed = function Const v -> Some v | _ -> None in
  let fold2 build op a b =
    match (closed a, closed b) with
    | Some x, Some y -> Const (op x y)
    | _ -> build a b
  in
  let rec go blocked = function
    | Binary (op, a, b) ->
        let blocked, a = go blocked a in
        let blocked, b = go blocked b in
        ( blocked,
          fold2 (fun a b -> Binary (op, a, b)) (Expr.apply_binary_op op) a b )
    | (Cell _ | Const _) as leaf -> (blocked, leaf)
    | Max (op, a, b) ->
        let blocked, a = go blocked a in
        let blocked, b = go blocked b in
        (blocked, fold2 (fun a b -> Max (op, a, b)) (Max_op.apply op) a b)
    | Round x -> (
        let blocked, inner = go blocked x in
        match inner with
        | Cell c ->
            if stored_f32 c then (blocked, inner)
            else (Cell.Set.add c blocked, Round inner)
        | Const v -> (blocked, Const (to_f32 v))
        | Round _ -> (blocked, inner)
        | _ -> (blocked, Round inner))
    | Select (g, a, b) -> (
        let blocked, g = go_guard blocked g in
        (* A closed guard decides the branch outright, even when the branches
           themselves are open — which is what collapses a max-pool index chain
           over known data. Only the SELECTED branch is normalised: normalising
           both would collect blocked cells from code that cannot run, and the
           driver reports a blocked collapse before it probes, so a spurious one
           would mask a real counterexample in the branch actually taken. *)
        match guard_value g with
        | Some true -> go blocked a
        | Some false -> go blocked b
        | None ->
            let blocked, a = go blocked a in
            let blocked, b = go blocked b in
            (blocked, Select (g, a, b)))
    | Unary (op, x) -> (
        let blocked, x = go blocked x in
        ( blocked,
          match closed x with
          | Some v -> Const (Expr.apply_unary_op op v)
          | None -> Unary (op, x) ))
  and go_guard blocked = function
    | Lt (a, b) ->
        let blocked, a = go blocked a in
        let blocked, b = go blocked b in
        (blocked, Lt (a, b))
    | Pool_better { best; value } ->
        let blocked, best = go blocked best in
        let blocked, value = go blocked value in
        (blocked, Pool_better { best; value })
  and guard_value = function
    | Lt (a, b) -> (
        match (closed a, closed b) with
        | Some x, Some y -> Some (Expr.apply_compare_op Expr.Lt x y)
        | _ -> None)
    | Pool_better { best; value } -> (
        match (closed best, closed value) with
        | Some best, Some value -> Some (Max_op.pool_better ~best ~value)
        | _ -> None)
  in
  let blocked, expr = go Cell.Set.empty e in
  { blocked; expr }

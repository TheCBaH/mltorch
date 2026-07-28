(* See ground_expr.mli. *)

module Cell = struct
  type t = { coord : Vec6.coord; id : Tensor_id.t }

  let compare a b =
    match Tensor_id.compare a.id b.id with
    | 0 -> Stdlib.compare a.coord b.coord
    | n -> n

  let pp fmt c = Fmt.pf fmt "%a%a" Tensor_id.pp c.id Vec6.pp_coord c.coord

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

  (* A ramp over the flattened coordinate, offset per edge, so no two cells of
     one tensor share a value and no two tensors line up. *)
  let ramp (c : Cell.t) =
    let axis a = Dim.to_int (Vec6.get c.coord a) in
    float_of_int
      ((Tensor_id.to_int c.id * 101)
      + (axis Axis.N * 10007)
      + (axis Axis.T * 1009)
      + (axis Axis.D * 503)
      + (axis Axis.H * 53)
      + (axis Axis.W * 7)
      + axis Axis.C + 1)

  (* [ramp] already distinguishes every cell, so it is the whole key here. *)
  let pseudo_random n (c : Cell.t) =
    let h = Hashtbl.hash (n, ramp c) in
    let v = float_of_int ((h mod 1999) - 999) /. 250. in
    if Float.equal v 0. then 1. else v

  let draw n cells =
    Cell.Set.fold
      (fun c acc ->
        Cell.Map.add c (if n = 0 then ramp c else pseudo_random n c) acc)
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

(* Collapses the three [Round] rules, threading the set of cells whose [Round]
   the first rule's side condition refused. *)
let normalise ~stored_f32 e =
  let rec go blocked = function
    | Binary (op, a, b) ->
        let blocked, a = go blocked a in
        let blocked, b = go blocked b in
        (blocked, Binary (op, a, b))
    | (Cell _ | Const _) as leaf -> (blocked, leaf)
    | Max (op, a, b) ->
        let blocked, a = go blocked a in
        let blocked, b = go blocked b in
        (blocked, Max (op, a, b))
    | Round x -> (
        let blocked, inner = go blocked x in
        match inner with
        | Cell c ->
            if stored_f32 c then (blocked, inner)
            else (Cell.Set.add c blocked, Round inner)
        | Const v -> (blocked, Const (to_f32 v))
        | Round _ -> (blocked, inner)
        | _ -> (blocked, Round inner))
    | Select (g, a, b) ->
        let blocked, g = go_guard blocked g in
        let blocked, a = go blocked a in
        let blocked, b = go blocked b in
        (blocked, Select (g, a, b))
    | Unary (op, x) ->
        let blocked, x = go blocked x in
        (blocked, Unary (op, x))
  and go_guard blocked = function
    | Lt (a, b) ->
        let blocked, a = go blocked a in
        let blocked, b = go blocked b in
        (blocked, Lt (a, b))
    | Pool_better { best; value } ->
        let blocked, best = go blocked best in
        let blocked, value = go blocked value in
        (blocked, Pool_better { best; value })
  in
  let blocked, expr = go Cell.Set.empty e in
  { blocked; expr }

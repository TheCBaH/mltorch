(* See coeff_form.mli. *)

module Generator = struct
  type t = Cell of Ground_expr.Cell.t | Opaque of Ground_expr.t

  let compare a b =
    match (a, b) with
    | Cell x, Cell y -> Ground_expr.Cell.compare x y
    | Opaque x, Opaque y -> Ground_expr.compare x y
    | Cell _, Opaque _ -> -1
    | Opaque _, Cell _ -> 1

  let pp fmt = function
    | Cell c -> Ground_expr.Cell.pp fmt c
    | Opaque e -> Fmt.pf fmt "<%a>" Ground_expr.pp e
end

(* A monomial is a sorted multiset of generators; the empty list is the
   constant term. *)
module Monomial = Map.Make (struct
  type t = Generator.t list

  let compare = List.compare Generator.compare
end)

type t = float Monomial.t

let zero : t = Monomial.empty
let constant v : t = if Float.equal v 0. then zero else Monomial.singleton [] v

let add_term m coeff (p : t) : t =
  if Float.equal coeff 0. then p
  else
    Monomial.update m
      (function None -> Some coeff | Some c -> Some (c +. coeff))
      p

let add (a : t) (b : t) : t = Monomial.fold add_term b a
let scale k (a : t) : t = Monomial.map (fun c -> c *. k) a
let sub a b = add a (scale (-1.) b)

let mul (a : t) (b : t) : t =
  Monomial.fold
    (fun ma ca acc ->
      Monomial.fold
        (fun mb cb acc ->
          add_term (List.merge Generator.compare ma mb) (ca *. cb) acc)
        b acc)
    a zero

let atom g : t = Monomial.singleton [ g ] 1.

(* [Monomial.bindings] is sorted, so this is deterministic. *)
let pp fmt (p : t) =
  let term fmt (m, c) =
    match m with
    | [] -> Fmt.pf fmt "%h" c
    | _ -> Fmt.pf fmt "%h*%a" c (Fmt.list ~sep:(Fmt.any "*") Generator.pp) m
  in
  Fmt.pf fmt "@[<h>%a@]"
    (Fmt.list ~sep:(Fmt.any " + ") term)
    (Monomial.bindings p)

(* The constant term, when the polynomial has nothing else — used to decide
   whether a division can be turned into a scale. *)
let as_constant (p : t) =
  match Monomial.bindings p with
  | [] -> Some 0.
  | [ ([], c) ] -> Some c
  | _ -> None

let rec of_ground (e : Ground_expr.t) : t =
  match e with
  | Ground_expr.Const v -> constant v
  | Ground_expr.Round x -> of_ground x (* the [Equivalent] reading *)
  | Ground_expr.Binary (Expr.Value.Add, a, b) -> add (of_ground a) (of_ground b)
  | Ground_expr.Binary (Expr.Value.Sub, a, b) -> sub (of_ground a) (of_ground b)
  | Ground_expr.Binary (Expr.Value.Mul, a, b) -> mul (of_ground a) (of_ground b)
  | Ground_expr.Binary (Expr.Value.Div, a, b) -> (
      let bp = of_ground b in
      match as_constant bp with
      | Some c when not (Float.equal c 0.) -> scale (1. /. c) (of_ground a)
      | _ -> atom (Generator.Opaque e))
  | Ground_expr.Cell c -> atom (Generator.Cell c)
  | Ground_expr.Max _ | Ground_expr.Select _ | Ground_expr.Unary _ ->
      atom (Generator.Opaque e)

let close ~tolerance a b =
  Float.abs (a -. b)
  <= tolerance *. Float.max 1. (Float.max (Float.abs a) (Float.abs b))

(* Drop terms that are already within tolerance of absent, so a coefficient that
   cancelled to a rounding residue on one side only does not count as a
   structural difference. *)
let significant ~tolerance (p : t) =
  Monomial.filter (fun _ c -> not (close ~tolerance c 0.)) p

let agree_within ~tolerance a b =
  let a = significant ~tolerance a and b = significant ~tolerance b in
  Monomial.equal (fun x y -> close ~tolerance x y) a b

(* Maximal arithmetic regions go through the polynomial view; matching
   non-arithmetic heads recurse, so a relu wrapping the fold compares its
   operands rather than two unequal opaque generators. *)
let rec agree ~tolerance (a : Ground_expr.t) (b : Ground_expr.t) =
  match (a, b) with
  | Ground_expr.Round x, _ -> agree ~tolerance x b
  | _, Ground_expr.Round y -> agree ~tolerance a y
  | Ground_expr.Select (g1, x1, y1), Ground_expr.Select (g2, x2, y2) ->
      agree_guard ~tolerance g1 g2
      && agree ~tolerance x1 x2 && agree ~tolerance y1 y2
  | Ground_expr.Max (o1, x1, y1), Ground_expr.Max (o2, x2, y2) ->
      o1 = o2 && agree ~tolerance x1 x2 && agree ~tolerance y1 y2
  | Ground_expr.Unary (o1, x1), Ground_expr.Unary (o2, x2) ->
      o1 = o2 && agree ~tolerance x1 x2
  | _ -> agree_within ~tolerance (of_ground a) (of_ground b)

and agree_guard ~tolerance g1 g2 =
  match (g1, g2) with
  | Ground_expr.Lt (a1, b1), Ground_expr.Lt (a2, b2) ->
      agree ~tolerance a1 a2 && agree ~tolerance b1 b2
  | Ground_expr.Pool_better p1, Ground_expr.Pool_better p2 ->
      agree ~tolerance p1.best p2.best && agree ~tolerance p1.value p2.value
  | Ground_expr.Lt _, Ground_expr.Pool_better _
  | Ground_expr.Pool_better _, Ground_expr.Lt _ ->
      false

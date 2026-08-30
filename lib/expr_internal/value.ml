type binary_op = Expr_repr.binary_op = Add | Div | Mul | Sub
type unary_op = Expr_repr.unary_op = Erf | Exp | Log | Sqrt

type t = Expr_repr.value =
  | Binary of binary_op * t * t
  | Const of float
  | Intrinsic of Intrinsic.t
  | Load of Source.t * Role.Position.t Index.t Coord.t
  | Reduce of Expr_repr.reduction
  | Round_f32 of t
  | Select of Expr_repr.bool_expr * t * t
  | Unary of unary_op * t
  | Value_of_index of Role.Delta.t Index.t

let const x = Const x
let add a b = Binary (Add, a, b)
let sub a b = Binary (Sub, a, b)
let mul a b = Binary (Mul, a, b)
let div a b = Binary (Div, a, b)
let exp a = Unary (Exp, a)
let sqrt a = Unary (Sqrt, a)
let erf a = Unary (Erf, a)
let log a = Unary (Log, a)
let select c a b = Select (c, a, b)
let value_of_index i = Value_of_index i
let load s c = Load (s, c)
let round_f32 a = Round_f32 a
let intrinsic i = Intrinsic i
let reduce r = Reduce r

let apply_binary = function
  | Add -> ( +. )
  | Div -> ( /. )
  | Mul -> ( *. )
  | Sub -> ( -. )

let erf_approx x =
  let p = 0.3275911 in
  let a1 = 0.254829592 in
  let a2 = -0.284496736 in
  let a3 = 1.421413741 in
  let a4 = -1.453152027 in
  let a5 = 1.061405429 in
  let sign = if x < 0. then -1. else 1. in
  let ax = Stdlib.abs_float x in
  let t = 1. /. (1. +. (p *. ax)) in
  let poly =
    t *. (a1 +. (t *. (a2 +. (t *. (a3 +. (t *. (a4 +. (t *. a5))))))))
  in
  sign *. (1. -. (poly *. Stdlib.exp (-.ax *. ax)))

let apply_unary = function
  | Erf -> erf_approx
  | Exp -> Stdlib.exp
  | Log -> Stdlib.log
  | Sqrt -> Stdlib.sqrt

let binary_sym = function Add -> "+" | Div -> "/" | Mul -> "*" | Sub -> "-"

let unary_name = function
  | Erf -> "erf"
  | Exp -> "exp"
  | Log -> "log"
  | Sqrt -> "sqrt"

let index_tag : type r. r Index.t -> int = function
  | Index.Add _ -> 5
  | Index.Assume_position _ -> 12
  | Index.Ceil_div_pos _ -> 8
  | Index.Clamp_low _ -> 11
  | Index.Const _ -> 3
  | Index.Data _ -> 13
  | Index.Floor_div_pos _ -> 7
  | Index.Max _ -> 10
  | Index.Min _ -> 9
  | Index.Of_position _ -> 4
  | Index.Output _ -> 0
  | Index.Reduce _ -> 1
  | Index.Scale _ -> 6
  | Index.Zero -> 2

let level env v =
  match Reduce_var.Map.find_opt v env with
  | Some l -> l
  | None -> -1 - Reduce_var.hash v

let ( <?> ) c f = if c <> 0 then c else f ()

let rec cmp_index : type r s.
    int Reduce_var.Map.t ->
    int Reduce_var.Map.t ->
    r Index.t ->
    s Index.t ->
    int =
 fun ea eb a b ->
  Int.compare (index_tag a) (index_tag b) <?> fun () ->
  match (a, b) with
  | Index.Add (x1, x2), Index.Add (y1, y2) ->
      cmp_index ea eb x1 y1 <?> fun () -> cmp_index ea eb x2 y2
  | Index.Assume_position x, Index.Assume_position y -> cmp_index ea eb x y
  | Index.Ceil_div_pos (x, k), Index.Ceil_div_pos (y, l) ->
      Int.compare k l <?> fun () -> cmp_index ea eb x y
  | Index.Clamp_low x, Index.Clamp_low y -> cmp_index ea eb x y
  | Index.Const x, Index.Const y -> Int.compare x y
  | Index.Data (s1, c1, e1), Index.Data (s2, c2, e2) ->
      Source.compare s1 s2 <?> fun () ->
      List.fold_left2
        (fun acc x y -> acc <?> fun () -> cmp_index ea eb x y)
        0 (Coord.to_list c1) (Coord.to_list c2)
      <?> fun () -> Int.compare e1 e2
  | Index.Floor_div_pos (x, k), Index.Floor_div_pos (y, l) ->
      Int.compare k l <?> fun () -> cmp_index ea eb x y
  | Index.Max (x1, x2), Index.Max (y1, y2) ->
      cmp_index ea eb x1 y1 <?> fun () -> cmp_index ea eb x2 y2
  | Index.Min (x1, x2), Index.Min (y1, y2) ->
      cmp_index ea eb x1 y1 <?> fun () -> cmp_index ea eb x2 y2
  | Index.Of_position x, Index.Of_position y -> cmp_index ea eb x y
  | Index.Output x, Index.Output y -> Axis.compare x y
  | Index.Reduce x, Index.Reduce y -> Int.compare (level ea x) (level eb y)
  | Index.Scale (k, x), Index.Scale (l, y) ->
      Int.compare k l <?> fun () -> cmp_index ea eb x y
  | Index.Zero, Index.Zero -> 0
  | _ -> 0

(* This is the stable structural-comparison encoding, not a display table: the
   assigned integers determine [compare]'s ordering and therefore must remain
   unchanged. Its branch order deliberately follows that encoding rather than
   the alphabetized [t] declaration. *)
let tag = function
  | Const _ -> 0
  | Binary _ -> 1
  | Unary _ -> 2
  | Select _ -> 3
  | Value_of_index _ -> 4
  | Load _ -> 5
  | Round_f32 _ -> 6
  | Reduce _ -> 7
  | Intrinsic _ -> 8

let cmp_intrinsic ea eb (Intrinsic.Max_pool x) (Intrinsic.Max_pool y) =
  let open Intrinsic.Max_pool in
  let fld f = Int.compare (f x) (f y) in
  Source.compare x.source y.source <?> fun () ->
  fld (fun d -> d.in_h) <?> fun () ->
  fld (fun d -> d.in_w) <?> fun () ->
  fld (fun d -> d.kernel_h) <?> fun () ->
  fld (fun d -> d.kernel_w) <?> fun () ->
  fld (fun d -> d.stride_h) <?> fun () ->
  fld (fun d -> d.stride_w) <?> fun () ->
  fld (fun d -> d.pad_h) <?> fun () ->
  fld (fun d -> d.pad_w) <?> fun () ->
  Stdlib.compare x.result y.result <?> fun () ->
  List.fold_left2
    (fun acc a b -> acc <?> fun () -> cmp_index ea eb a b)
    0 (Coord.to_list x.out) (Coord.to_list y.out)

let compare a b =
  let rec go ea eb n a b =
    Int.compare (tag a) (tag b) <?> fun () ->
    match (a, b) with
    | Const x, Const y -> Core.Float_bits.compare_portable x y
    | Binary (o, x1, x2), Binary (p, y1, y2) ->
        Stdlib.compare o p <?> fun () ->
        go ea eb n x1 y1 <?> fun () -> go ea eb n x2 y2
    | Unary (o, x), Unary (p, y) ->
        Stdlib.compare o p <?> fun () -> go ea eb n x y
    | Round_f32 x, Round_f32 y -> go ea eb n x y
    | Select (c, x1, x2), Select (d, y1, y2) ->
        (match (c, d) with
          | Expr_repr.Value_lt (p, q), Expr_repr.Value_lt (r, s) ->
              go ea eb n p r <?> fun () -> go ea eb n q s
          | Expr_repr.Index_eq (p, q), Expr_repr.Index_eq (r, s) ->
              cmp_index ea eb p r <?> fun () -> cmp_index ea eb q s
          | Expr_repr.Value_lt _, Expr_repr.Index_eq _ -> -1
          | Expr_repr.Index_eq _, Expr_repr.Value_lt _ -> 1)
        <?> fun () ->
        go ea eb n x1 y1 <?> fun () -> go ea eb n x2 y2
    | Value_of_index x, Value_of_index y -> cmp_index ea eb x y
    | Load (s, x), Load (t, y) ->
        Source.compare s t <?> fun () ->
        List.fold_left2
          (fun acc a b -> acc <?> fun () -> cmp_index ea eb a b)
          0 (Coord.to_list x) (Coord.to_list y)
    | Reduce r, Reduce s ->
        Stdlib.compare r.kind s.kind <?> fun () ->
        cmp_index ea eb r.lo s.lo <?> fun () ->
        cmp_index ea eb r.hi s.hi <?> fun () ->
        go
          (Reduce_var.Map.add r.var n ea)
          (Reduce_var.Map.add s.var n eb)
          (n + 1) r.body s.body
    | Intrinsic x, Intrinsic y -> cmp_intrinsic ea eb x y
    | _ -> 0
  in
  go Reduce_var.Map.empty Reduce_var.Map.empty 0 a b

let equal a b = compare a b = 0

let hash e =
  let mix h x = (h * 31) + x in
  let rec idx : type r. int Reduce_var.Map.t -> int -> r Index.t -> int =
   fun env h i ->
    let h = mix h (index_tag i) in
    match i with
    | Index.Add (a, b) -> idx env (idx env h a) b
    | Index.Assume_position a -> idx env h a
    | Index.Ceil_div_pos (a, d) -> idx env (mix h d) a
    | Index.Clamp_low a -> idx env h a
    | Index.Const n -> mix h n
    | Index.Data (s, c, e) ->
        let h = mix h (Source.hash s) in
        let h = Coord.fold (fun h i -> idx env h i) h c in
        mix h e
    | Index.Floor_div_pos (a, d) -> idx env (mix h d) a
    | Index.Max (a, b) -> idx env (idx env h a) b
    | Index.Min (a, b) -> idx env (idx env h a) b
    | Index.Of_position a -> idx env h a
    | Index.Output a -> mix h (Axis.to_int a)
    | Index.Reduce v -> mix h (level env v)
    | Index.Scale (k, a) -> idx env (mix h k) a
    | Index.Zero -> h
  in
  let rec go env n h (e : t) =
    let h = mix h (tag e) in
    match e with
    | Const x ->
        let b = Core.Float_bits.portable x in
        mix
          (mix h (Int64.to_int (Int64.logand b 0xFFFFFFFFL)))
          (Int64.to_int (Int64.shift_right_logical b 32))
    | Binary (o, a, b) -> go env n (go env n (mix h (Hashtbl.hash o)) a) b
    | Unary (o, a) -> go env n (mix h (Hashtbl.hash o)) a
    | Round_f32 a -> go env n h a
    | Select (c, a, b) ->
        let h =
          match c with
          | Expr_repr.Value_lt (x, y) -> go env n (go env n h x) y
          | Expr_repr.Index_eq (x, y) -> idx env (idx env h x) y
        in
        go env n (go env n h a) b
    | Value_of_index i -> idx env h i
    | Load (s, c) ->
        Coord.fold (fun h i -> idx env h i) (mix h (Source.hash s)) c
    | Reduce r ->
        let h = mix h (Hashtbl.hash r.kind) in
        let h = idx env (idx env h r.lo) r.hi in
        go (Reduce_var.Map.add r.var n env) (n + 1) h r.body
    | Intrinsic (Intrinsic.Max_pool d) ->
        let h = mix h (Source.hash d.source) in
        let h =
          List.fold_left mix h
            [
              d.in_h;
              d.in_w;
              d.kernel_h;
              d.kernel_w;
              d.stride_h;
              d.stride_w;
              d.pad_h;
              d.pad_w;
              Hashtbl.hash d.result;
            ]
        in
        Coord.fold (fun h i -> idx env h i) h d.out
  in
  go Reduce_var.Map.empty 0 17 e

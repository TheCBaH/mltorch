type binary_op = Expr_repr.binary_op = Add | Div | Mul | Sub
type unary_op = Expr_repr.unary_op = Erf | Exp | Log | Sqrt | Trunc

type t = Expr_repr.value =
  | Binary of binary_op * t * t
  | Const of float
  | Intrinsic of Intrinsic.t
  | Local of Local_var.t
  | Local_at of Local_var.t * Role.Position.t Index.t
  | Local_scan_at of
      Local_var.t * Role.Position.t Index.t * Role.Position.t Index.t
  | Load of Source.t * Role.Position.t Index.t Coord.t
  | Reduce of Expr_repr.reduction
  | Round_f32 of t
  | Scan_at of
      Expr_repr.scan * Role.Position.t Index.t * Role.Position.t Index.t
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
let trunc a = Unary (Trunc, a)
let select c a b = Select (c, a, b)
let value_of_index i = Value_of_index i
let load s c = Load (s, c)
let round_f32 a = Round_f32 a
let intrinsic i = Intrinsic i
let local v = Local v
let local_at v i = Local_at (v, i)
let local_scan_at v ~row ~lane = Local_scan_at (v, row, lane)
let scan_at s ~row ~lane = Scan_at (s, row, lane)
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
  | Trunc -> Float.trunc

let binary_sym = function Add -> "+" | Div -> "/" | Mul -> "*" | Sub -> "-"

let unary_name = function
  | Erf -> "erf"
  | Exp -> "exp"
  | Log -> "log"
  | Sqrt -> "sqrt"
  | Trunc -> "trunc"

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
  | Local _ -> 9
  | Local_at _ -> 10
  | Local_scan_at _ -> 11
  | Scan_at _ -> 12

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
  let rec go ea eb la lb n a b =
    Int.compare (tag a) (tag b) <?> fun () ->
    match (a, b) with
    | Const x, Const y -> Core.Float_bits.compare_portable x y
    | Binary (o, x1, x2), Binary (p, y1, y2) ->
        Stdlib.compare o p <?> fun () ->
        go ea eb la lb n x1 y1 <?> fun () -> go ea eb la lb n x2 y2
    | Unary (o, x), Unary (p, y) ->
        Stdlib.compare o p <?> fun () -> go ea eb la lb n x y
    | Round_f32 x, Round_f32 y -> go ea eb la lb n x y
    | Select (c, x1, x2), Select (d, y1, y2) ->
        (match (c, d) with
          | Expr_repr.Value_lt (p, q), Expr_repr.Value_lt (r, s) ->
              go ea eb la lb n p r <?> fun () -> go ea eb la lb n q s
          | Expr_repr.Index_eq (p, q), Expr_repr.Index_eq (r, s) ->
              cmp_index ea eb p r <?> fun () -> cmp_index ea eb q s
          | Expr_repr.Value_lt _, Expr_repr.Index_eq _ -> -1
          | Expr_repr.Index_eq _, Expr_repr.Value_lt _ -> 1)
        <?> fun () ->
        go ea eb la lb n x1 y1 <?> fun () -> go ea eb la lb n x2 y2
    | Value_of_index x, Value_of_index y -> cmp_index ea eb x y
    | Load (s, x), Load (t, y) ->
        Source.compare s t <?> fun () ->
        List.fold_left2
          (fun acc a b -> acc <?> fun () -> cmp_index ea eb a b)
          0 (Coord.to_list x) (Coord.to_list y)
    | Local x, Local y -> Local_var.compare x y
    (* [la]/[lb] hold only [prev] identities, scoped to their own scan's
       [update] -- everywhere else they are empty, so the [None, None] arm
       is what every non-scan comparison already took: raw identity compare,
       unchanged. Only within a matching scan scope do two structurally
       alpha-equivalent [prev] readers compare by binder level instead. *)
    | Local_at (x, i), Local_at (y, j) ->
        (match (Local_var.Map.find_opt x la, Local_var.Map.find_opt y lb) with
          | None, None -> Local_var.compare x y
          | Some _, None -> -1
          | None, Some _ -> 1
          | Some lx, Some ly -> Int.compare lx ly)
        <?> fun () -> cmp_index ea eb i j
    | Local_scan_at (x, ri, li), Local_scan_at (y, rj, lj) ->
        Local_var.compare x y <?> fun () ->
        cmp_index ea eb ri rj <?> fun () -> cmp_index ea eb li lj
    | Reduce r, Reduce s ->
        Stdlib.compare r.kind s.kind <?> fun () ->
        cmp_index ea eb r.lo s.lo <?> fun () ->
        cmp_index ea eb r.hi s.hi <?> fun () ->
        go
          (Reduce_var.Map.add r.var n ea)
          (Reduce_var.Map.add s.var n eb)
          la lb (n + 1) r.body s.body
    | Scan_at (p, ri, li), Scan_at (q, rj, lj) ->
        Int.compare p.Expr_repr.width q.Expr_repr.width <?> fun () ->
        Int.compare p.Expr_repr.steps q.Expr_repr.steps <?> fun () ->
        cmp_index ea eb ri rj <?> fun () ->
        cmp_index ea eb li lj <?> fun () ->
        go
          (Reduce_var.Map.add p.Expr_repr.lane n ea)
          (Reduce_var.Map.add q.Expr_repr.lane n eb)
          la lb (n + 1) p.Expr_repr.init q.Expr_repr.init
        <?> fun () ->
        go
          (Reduce_var.Map.add p.Expr_repr.lane n
             (Reduce_var.Map.add p.Expr_repr.step (n + 1) ea))
          (Reduce_var.Map.add q.Expr_repr.lane n
             (Reduce_var.Map.add q.Expr_repr.step (n + 1) eb))
          (Local_var.Map.add p.Expr_repr.prev (n + 2) la)
          (Local_var.Map.add q.Expr_repr.prev (n + 2) lb)
          (n + 3) p.Expr_repr.update q.Expr_repr.update
    | Intrinsic x, Intrinsic y -> cmp_intrinsic ea eb x y
    | _ -> 0
  in
  go Reduce_var.Map.empty Reduce_var.Map.empty Local_var.Map.empty
    Local_var.Map.empty 0 a b

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
  (* [lenv] holds only [prev] identities, scoped to their own scan's
     [update], mirroring [env] for reducers. Elsewhere it is empty, so
     [Local_at]'s hash is unchanged from before scan existed. *)
  let local_hash lenv v =
    match Local_var.Map.find_opt v lenv with
    | Some l -> l
    | None -> Local_var.hash v
  in
  let rec go env lenv n h (e : t) =
    let h = mix h (tag e) in
    match e with
    | Const x ->
        let b = Core.Float_bits.portable x in
        mix
          (mix h (Int64.to_int (Int64.logand b 0xFFFFFFFFL)))
          (Int64.to_int (Int64.shift_right_logical b 32))
    | Binary (o, a, b) ->
        go env lenv n (go env lenv n (mix h (Hashtbl.hash o)) a) b
    | Unary (o, a) -> go env lenv n (mix h (Hashtbl.hash o)) a
    | Round_f32 a -> go env lenv n h a
    | Select (c, a, b) ->
        let h =
          match c with
          | Expr_repr.Value_lt (x, y) -> go env lenv n (go env lenv n h x) y
          | Expr_repr.Index_eq (x, y) -> idx env (idx env h x) y
        in
        go env lenv n (go env lenv n h a) b
    | Value_of_index i -> idx env h i
    | Load (s, c) ->
        Coord.fold (fun h i -> idx env h i) (mix h (Source.hash s)) c
    | Local v -> mix h (Local_var.hash v)
    | Local_at (v, i) -> idx env (mix h (local_hash lenv v)) i
    | Local_scan_at (v, row, lane) ->
        idx env (idx env (mix h (Local_var.hash v)) row) lane
    | Reduce r ->
        let h = mix h (Hashtbl.hash r.kind) in
        let h = idx env (idx env h r.lo) r.hi in
        go (Reduce_var.Map.add r.var n env) lenv (n + 1) h r.body
    | Scan_at (s, row, lane) ->
        let h = mix h s.Expr_repr.width in
        let h = mix h s.Expr_repr.steps in
        let h = idx env (idx env h row) lane in
        let h =
          go
            (Reduce_var.Map.add s.Expr_repr.lane n env)
            lenv (n + 1) h s.Expr_repr.init
        in
        go
          (Reduce_var.Map.add s.Expr_repr.lane n
             (Reduce_var.Map.add s.Expr_repr.step (n + 1) env))
          (Local_var.Map.add s.Expr_repr.prev (n + 2) lenv)
          (n + 3) h s.Expr_repr.update
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
  go Reduce_var.Map.empty Local_var.Map.empty 0 17 e

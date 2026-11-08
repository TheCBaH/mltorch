open Ssa_ir

module Region = struct
  type t = Dtcm | Ram | Sram

  let name = function Dtcm -> "dtcm" | Ram -> "ram" | Sram -> "sram"
end

module Class = struct
  type t =
    | Branch
    | F32_div
    | F32_fma
    | F32_math of Expr.Value.unary_op
    | F32_op
    | F32_sqrt
    | F64_op
    | Int_op
    | Iteration
    | Load of Region.t
    | Local
    | Store of Region.t
    | Vec_f32_div
    | Vec_f32_fma
    | Vec_f32_op
    | Vec_load of Region.t
    | Vec_load_strided of Region.t
    | Vec_math of Expr.Value.unary_op
    | Vec_other
    | Vec_store of Region.t
    | Vec_store_strided of Region.t

  let compare = Stdlib.compare

  let unary = function
    | Expr.Value.Cos -> "cos"
    | Expr.Value.Erf -> "erf"
    | Expr.Value.Exp -> "exp"
    | Expr.Value.Log -> "log"
    | Expr.Value.Round_bf16 -> "round_bf16"
    | Expr.Value.Sin -> "sin"
    | Expr.Value.Sqrt -> "sqrt"
    | Expr.Value.Trunc -> "trunc"

  let name = function
    | Branch -> "branch"
    | F32_div -> "f32_div"
    | F32_fma -> "f32_fma"
    | F32_math u -> "f32_" ^ unary u
    | F32_op -> "f32_op"
    | F32_sqrt -> "f32_sqrt"
    | F64_op -> "f64_op"
    | Int_op -> "int_op"
    | Iteration -> "iteration"
    | Load r -> "load_" ^ Region.name r
    | Local -> "local"
    | Store r -> "store_" ^ Region.name r
    | Vec_f32_div -> "vec_f32_div"
    | Vec_f32_fma -> "vec_f32_fma"
    | Vec_f32_op -> "vec_f32_op"
    | Vec_load r -> "vec_load_" ^ Region.name r
    | Vec_load_strided r -> "vec_load_strided_" ^ Region.name r
    | Vec_math u -> "vec_" ^ unary u
    | Vec_other -> "vec_other"
    | Vec_store r -> "vec_store_" ^ Region.name r
    | Vec_store_strided r -> "vec_store_strided_" ^ Region.name r
end

module M = Map.Make (Class)

module Contract_call = struct
  type t = {
    desc : Contraction.t;
    engine : Contraction.Select.Engine.t option;
    a : Region.t;
    b : Region.t;
    c : Region.t;
  }
end

module Attend_call = struct
  type t = {
    desc : Attention_block.t;
    q : Region.t;
    kv : Region.t;
    o : Region.t;
  }
end

type t = {
  counts : (Class.t * float) list;
  contracts : (Contract_call.t * float) list;
  attends : (Attend_call.t * float) list;
  bodies : (Class.t * float) list list;
  unknown : int;
}

(* The mutable accumulator of one walk. *)
type acc = {
  mutable classes : float M.t;
  mutable bodies : (Ssa_region.t * float M.t) list;
  mutable contracts : (Contract_call.t * float) list;
  mutable attends : (Attend_call.t * float) list;
  mutable unknown : int;
}

let bump acc c w =
  if w <> 0. then
    acc.classes <-
      M.update c (fun v -> Some (Option.value v ~default:0. +. w)) acc.classes

let bump_assoc l x w =
  if List.mem_assoc x l then
    List.map (fun (y, v) -> if y = x then (y, v +. w) else (y, v)) l
  else l @ [ (x, w) ]

(* 128-bit register groups a vector of [lanes] elements of [bits] fills *)
let groups ty =
  match ty with
  | Ssa_type.Vec (s, l) ->
      let bits = Ssa_type.bits s * Ssa_type.Lanes.to_int l in
      float_of_int ((bits + 127) / 128)
  | Ssa_type.Mask l -> float_of_int ((Ssa_type.Lanes.to_int l + 127) / 128)
  | _ -> 1.

let result_ty (i : Ssa_instr.t) =
  match i.Ssa_instr.results with v :: _ -> Some v.Ssa_value.ty | [] -> None

let is_f64 = function
  | Some (Ssa_type.Scalar Ssa_type.F64) | Some (Ssa_type.Vec (Ssa_type.F64, _))
    ->
      true
  | _ -> false

let operand_f64 (vs : Ssa_value.t list) =
  List.exists (fun (v : Ssa_value.t) -> is_f64 (Some v.Ssa_value.ty)) vs

(* the class of a scalar float operation, or of its lanewise lift *)
let float_class ~vector ~f64 (op : Ssa_op.t) =
  if f64 then Some Class.F64_op
  else
    match op with
    | Ssa_op.Float_binary (Expr.Value.Div, _, _) ->
        Some (if vector then Class.Vec_f32_div else Class.F32_div)
    | Ssa_op.Float_binary _ | Ssa_op.Float_compare _ | Ssa_op.Float_max _
    | Ssa_op.Select _ | Ssa_op.Pred_not _ | Ssa_op.Pred_or _
    | Ssa_op.Pool_better _ ->
        Some (if vector then Class.Vec_f32_op else Class.F32_op)
    | Ssa_op.Float_fma _ ->
        Some (if vector then Class.Vec_f32_fma else Class.F32_fma)
    | Ssa_op.Float_unary (Expr.Value.Sqrt, _) when not vector ->
        Some Class.F32_sqrt
    | Ssa_op.Float_unary (u, _) ->
        Some (if vector then Class.Vec_math u else Class.F32_math u)
    | _ -> None

let unit_steps (steps : int64 Expr.Coord.t) =
  (* contiguous when only the innermost axis steps, by one *)
  let others =
    Expr.Coord.fold (fun n s -> if s <> 0L then n + 1 else n) 0 steps
  in
  others <= 1 && (steps.Expr.Coord.c = 1L || others = 0)

let instr ~region acc w (i : Ssa_instr.t) =
  let ty = result_ty i in
  let op = i.Ssa_instr.op in
  match op with
  | Ssa_op.Attend { desc; q; k; o; _ } ->
      acc.attends <-
        bump_assoc acc.attends
          { Attend_call.desc; q = region q; kv = region k; o = region o }
          w
  | Ssa_op.Contract { desc; engine; a; b; c } ->
      acc.contracts <-
        bump_assoc acc.contracts
          {
            Contract_call.desc;
            engine;
            a = region a;
            b = region b;
            c = region c;
          }
          w
  | Ssa_op.Const _ | Ssa_op.Mark _ | Ssa_op.Mark_lanes _ | Ssa_op.Local_alloc _
    ->
      ()
  | Ssa_op.Load { buffer; _ } | Ssa_op.Load_in_bounds { buffer; _ } ->
      bump acc (Class.Load (region buffer)) w
  | Ssa_op.Store { buffer; _ } -> bump acc (Class.Store (region buffer)) w
  | Ssa_op.Local_read _ | Ssa_op.Local_write _ -> bump acc Class.Local w
  | Ssa_op.Vec_load { buffer; steps; _ } ->
      let r = region buffer in
      bump acc
        (if unit_steps steps then Class.Vec_load r else Class.Vec_load_strided r)
        (w *. Option.fold ~none:1. ~some:groups ty)
  | Ssa_op.Vec_store { buffer; steps; value; _ } ->
      let r = region buffer in
      bump acc
        (if unit_steps steps then Class.Vec_store r
         else Class.Vec_store_strided r)
        (w *. groups value.Ssa_value.ty)
  | Ssa_op.Vec_extract _ | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _
  | Ssa_op.Vec_splat _ ->
      bump acc Class.Vec_other (w *. Option.fold ~none:1. ~some:groups ty)
  | Ssa_op.Lanewise inner -> (
      let f64 = is_f64 ty || operand_f64 (Ssa_op.operands inner) in
      match float_class ~vector:true ~f64 inner with
      | Some c -> bump acc c (w *. Option.fold ~none:1. ~some:groups ty)
      | None -> bump acc Class.Vec_other w)
  | _ -> (
      let f64 = is_f64 ty || operand_f64 (Ssa_op.operands op) in
      match float_class ~vector:false ~f64 op with
      | Some c -> bump acc c w
      | None -> bump acc Class.Int_op w)

(* Index arithmetic over known values: what loop bounds and conditions are
   made of. *)
let eval env (i : Ssa_instr.t) =
  let get (v : Ssa_value.t) = Ssa_id.Value.Map.find_opt v.Ssa_value.id env in
  let ( let* ) = Option.bind in
  let r =
    match i.Ssa_instr.op with
    | Ssa_op.Const (Ssa_const.Index n) | Ssa_op.Const (Ssa_const.I64 n) ->
        Some n
    | Ssa_op.Const (Ssa_const.Pred b) -> Some (if b then 1L else 0L)
    | Ssa_op.Index_add (a, b) | Ssa_op.Index_add_in_domain (a, b) ->
        let* a = get a in
        let* b = get b in
        Some (Int64.add a b)
    | Ssa_op.Index_scale (k, a) | Ssa_op.Index_scale_in_domain (k, a) ->
        let* a = get a in
        Some (Int64.mul k a)
    | Ssa_op.Index_min (a, b) ->
        let* a = get a in
        let* b = get b in
        Some (min a b)
    | Ssa_op.Index_max (a, b) ->
        let* a = get a in
        let* b = get b in
        Some (max a b)
    | Ssa_op.Index_clamp_low a ->
        let* a = get a in
        Some (max 0L a)
    | Ssa_op.Index_floor_div (k, a) ->
        let* a = get a in
        Some
          (if Int64.rem a k <> 0L && a < 0L then Int64.pred (Int64.div a k)
           else Int64.div a k)
    | Ssa_op.Index_ceil_div (k, a) ->
        let* a = get a in
        Some (Int64.neg (Int64.div (Int64.neg a) k))
    | Ssa_op.Index_of_i64 a -> get a
    | Ssa_op.Index_compare (c, a, b) | Ssa_op.I64_compare (c, a, b) ->
        let* a = get a in
        let* b = get b in
        Some
          (if
             match c with
             | Ssa_op.Compare.Eq -> a = b
             | Ssa_op.Compare.Lt -> a < b
           then 1L
           else 0L)
    | Ssa_op.I64_arith (op, a, b) ->
        let* a = get a in
        let* b = get b in
        Some
          (match op with
          | Ssa_op.I64_op.Add -> Int64.add a b
          | Ssa_op.I64_op.Mul -> Int64.mul a b
          | Ssa_op.I64_op.Sub -> Int64.sub a b)
    | _ -> None
  in
  match (r, i.Ssa_instr.results) with
  | Some n, v :: _ -> Ssa_id.Value.Map.add v.Ssa_value.id n env
  | _ -> env

(* a body of instructions alone costs the same every trip *)
let flat (r : Ssa_region.t) =
  List.for_all
    (function Ssa_stmt.Instr _ -> true | _ -> false)
    r.Ssa_region.body

let rec walk ~region:reg acc w env (r : Ssa_region.t) =
  List.fold_left
    (fun env s -> stmt ~region:reg acc w env s)
    env r.Ssa_region.body

and loop ~region:reg acc w env ~lo ~hi ~step (body : Ssa_region.t) =
  let get (v : Ssa_value.t) = Ssa_id.Value.Map.find_opt v.Ssa_value.id env in
  let induction =
    match body.Ssa_region.params with v :: _ -> Some v | [] -> None
  in
  match (get lo, get hi) with
  | Some lo, Some hi ->
      let trips =
        if hi <= lo then 0L
        else Int64.div (Int64.add (Int64.sub hi lo) (Int64.pred step)) step
      in
      if flat body then (
        let w' = w *. Int64.to_float trips in
        let outer = acc.classes in
        acc.classes <- M.empty;
        bump acc Class.Iteration w';
        ignore (walk ~region:reg acc w' env body);
        let sub = acc.classes in
        acc.classes <- M.union (fun _ a b -> Some (a +. b)) outer sub;
        (* the same body run again (an outer loop's trips) is one body *)
        acc.bodies <-
          (if List.exists (fun (r, _) -> r == body) acc.bodies then
             List.map
               (fun (r, m) ->
                 if r == body then
                   (r, M.union (fun _ a b -> Some (a +. b)) m sub)
                 else (r, m))
               acc.bodies
           else acc.bodies @ [ (body, sub) ]))
      else
        let rec go k =
          if k < hi then (
            bump acc Class.Iteration w;
            let env =
              match induction with
              | Some v -> Ssa_id.Value.Map.add v.Ssa_value.id k env
              | None -> env
            in
            ignore (walk ~region:reg acc w env body);
            go (Int64.add k step))
        in
        go lo
  | _ ->
      acc.unknown <- acc.unknown + 1;
      bump acc Class.Iteration w;
      ignore (walk ~region:reg acc w env body)

and stmt ~region:reg acc w env = function
  | Ssa_stmt.Instr i ->
      instr ~region:reg acc w i;
      eval env i
  | Ssa_stmt.For { lo; hi; step; body; _ } ->
      loop ~region:reg acc w env ~lo ~hi ~step body;
      env
  | Ssa_stmt.Ordered_sum { lo; hi; body; _ } ->
      loop ~region:reg acc w env ~lo ~hi ~step:1L body;
      env
  | Ssa_stmt.If { cond; then_; else_; _ } -> (
      bump acc Class.Branch w;
      match Ssa_id.Value.Map.find_opt cond.Ssa_value.id env with
      | Some c ->
          ignore (walk ~region:reg acc w env (if c <> 0L then then_ else else_));
          env
      | None ->
          acc.unknown <- acc.unknown + 1;
          ignore (walk ~region:reg acc (w /. 2.) env then_);
          ignore (walk ~region:reg acc (w /. 2.) env else_);
          env)

let program ?(region = fun _ -> Region.Dtcm) (p : Ssa_program.t) =
  let acc =
    {
      classes = M.empty;
      bodies = [];
      contracts = [];
      attends = [];
      unknown = 0;
    }
  in
  ignore (walk ~region acc 1. Ssa_id.Value.Map.empty p.Ssa_program.entry);
  {
    counts = M.bindings acc.classes;
    contracts = acc.contracts;
    attends = acc.attends;
    bodies = List.map (fun (_, m) -> M.bindings m) acc.bodies;
    unknown = acc.unknown;
  }

let count (t : t) c = Option.value (List.assoc_opt c t.counts) ~default:0.

let add (a : t) (b : t) =
  let counts =
    List.fold_left
      (fun m (c, v) ->
        M.update c (fun x -> Some (Option.value x ~default:0. +. v)) m)
      (M.of_seq (List.to_seq a.counts))
      b.counts
  in
  {
    counts = M.bindings counts;
    contracts =
      List.fold_left (fun l (d, w) -> bump_assoc l d w) a.contracts b.contracts;
    attends =
      List.fold_left (fun l (d, w) -> bump_assoc l d w) a.attends b.attends;
    bodies = a.bodies @ b.bodies;
    unknown = a.unknown + b.unknown;
  }

let pp ppf (t : t) =
  Fmt.pf ppf "@[<v>%a"
    Fmt.(
      list ~sep:cut (fun ppf (c, v) -> Fmt.pf ppf "%s %.0f" (Class.name c) v))
    t.counts;
  List.iter
    (fun ((c : Contract_call.t), w) ->
      Fmt.pf ppf "@,contract %s%s (%s %s %s) x%.0f" (Contraction.name c.desc)
        (Option.fold ~none:""
           ~some:(fun e -> " engine=" ^ Contraction.Select.Engine.name e)
           c.engine)
        (Region.name c.a) (Region.name c.b) (Region.name c.c) w)
    t.contracts;
  List.iter
    (fun ((c : Attend_call.t), w) ->
      Fmt.pf ppf "@,attend %s (%s %s %s) x%.0f"
        (Attention_block.name c.desc)
        (Region.name c.q) (Region.name c.kv) (Region.name c.o) w)
    t.attends;
  if t.unknown > 0 then Fmt.pf ppf "@,unknown %d" t.unknown;
  Fmt.pf ppf "@]"

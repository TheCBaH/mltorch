(* Integer ranges of the values a program computes, and the claims they back.

   A range is a fact about every execution that reaches the value's definition:
   [Empty] says none does (the definition is unreachable), and a missing fact is
   the whole domain, never an assumption. Sums and products are formed in [int64]
   from operands that are in the 32-bit index domain, so they cannot wrap, and a
   checked operation's own range is its wide range clipped to the domain, because
   an execution that left it failed before producing the value.

   The analysis is derived from the program and never stored on it. A pass asks
   for it, and {!Ssa_verify} asks for it again to check a claim the program makes
   (an operation marked proven): a claim is accepted only where this analysis
   re-derives it. *)

type range = Empty | Range of { lo : int64; hi : int64 }

let domain = Range { lo = Ssa_const.index_min; hi = Ssa_const.index_max }
let top = Range { lo = Int64.min_int; hi = Int64.max_int }
let point n = Range { lo = n; hi = n }

let join a b =
  match (a, b) with
  | Empty, x | x, Empty -> x
  | Range a, Range b ->
      Range { lo = Stdlib.min a.lo b.lo; hi = Stdlib.max a.hi b.hi }

let meet a b =
  match (a, b) with
  | Empty, _ | _, Empty -> Empty
  | Range a, Range b ->
      let lo = Stdlib.max a.lo b.lo and hi = Stdlib.min a.hi b.hi in
      if Int64.compare lo hi > 0 then Empty else Range { lo; hi }

let subset r ~lo ~hi =
  match r with
  | Empty -> true
  | Range r -> Int64.compare r.lo lo >= 0 && Int64.compare r.hi hi <= 0

let within_domain r = subset r ~lo:Ssa_const.index_min ~hi:Ssa_const.index_max

(* Monotone nondecreasing maps act on the endpoints. *)
let map_monotone f = function
  | Empty -> Empty
  | Range r -> Range { lo = f r.lo; hi = f r.hi }

let add a b =
  match (a, b) with
  | Empty, _ | _, Empty -> Empty
  | Range a, Range b ->
      Range { lo = Int64.add a.lo b.lo; hi = Int64.add a.hi b.hi }

(* A scale by a literal swaps the endpoints when the literal is negative. *)
let scale k = function
  | Empty -> Empty
  | Range r ->
      let a = Int64.mul k r.lo and b = Int64.mul k r.hi in
      Range { lo = Stdlib.min a b; hi = Stdlib.max a b }

let floor_div n d =
  let q = Int64.div n d and r = Int64.rem n d in
  if Int64.compare r 0L < 0 then Int64.pred q else q

let ceil_div n d = Int64.neg (floor_div (Int64.neg n) d)

type t = { ranges : range Ssa_id.Value.Map.t }

let of_type (v : Ssa_value.t) =
  match v.Ssa_value.ty with
  | Ssa_type.Scalar Ssa_type.Index -> domain
  | _ -> top

let range t (v : Ssa_value.t) =
  match Ssa_id.Value.Map.find_opt v.Ssa_value.id t.ranges with
  | Some r -> r
  | None -> of_type v

let analyze (p : Ssa_program.t) =
  let ranges = ref Ssa_id.Value.Map.empty in
  let set (v : Ssa_value.t) r =
    ranges := Ssa_id.Value.Map.add v.Ssa_value.id r !ranges
  in
  let get (v : Ssa_value.t) =
    match Ssa_id.Value.Map.find_opt v.Ssa_value.id !ranges with
    | Some r -> r
    | None -> of_type v
  in
  (* The checked operations keep only what survives their own failure. *)
  let checked r = meet r domain in
  let op (i : Ssa_instr.t) =
    let result = List.hd i.Ssa_instr.results in
    let set_result r = set result r in
    match i.Ssa_instr.op with
    | Ssa_op.Const (Ssa_const.Index n) | Ssa_op.Const (Ssa_const.I64 n) ->
        set_result (point n)
    | Ssa_op.Convert (Ssa_op.Convert.Index_to_i64, a) -> set_result (get a)
    | Ssa_op.Index_add (a, b) | Ssa_op.Index_add_in_domain (a, b) ->
        set_result (checked (add (get a) (get b)))
    | Ssa_op.Index_scale (k, a) | Ssa_op.Index_scale_in_domain (k, a) ->
        set_result (checked (scale k (get a)))
    | Ssa_op.Index_clamp_low a ->
        set_result (map_monotone (fun x -> Stdlib.max 0L x) (get a))
    | Ssa_op.Index_floor_div (k, a) ->
        set_result (map_monotone (fun x -> floor_div x k) (get a))
    | Ssa_op.Index_ceil_div (k, a) ->
        set_result (map_monotone (fun x -> ceil_div x k) (get a))
    | Ssa_op.Index_max (a, b) -> (
        match (get a, get b) with
        | Empty, _ | _, Empty -> set_result Empty
        | Range a, Range b ->
            set_result
              (Range { lo = Stdlib.max a.lo b.lo; hi = Stdlib.max a.hi b.hi }))
    | Ssa_op.Index_min (a, b) -> (
        match (get a, get b) with
        | Empty, _ | _, Empty -> set_result Empty
        | Range a, Range b ->
            set_result
              (Range { lo = Stdlib.min a.lo b.lo; hi = Stdlib.min a.hi b.hi }))
    | Ssa_op.Index_of_i64 a -> set_result (meet (get a) domain)
    | Ssa_op.Select (_, a, b) -> (
        match result.Ssa_value.ty with
        | Ssa_type.Scalar (Ssa_type.Index | Ssa_type.I64) ->
            set_result (join (get a) (get b))
        | _ -> ())
    | Ssa_op.Check_access _ | Ssa_op.Check_gather _ | Ssa_op.Check_local _
    | Ssa_op.Check_scan _ | Ssa_op.Const _ | Ssa_op.Convert _
    | Ssa_op.Float_binary _ | Ssa_op.Float_compare _ | Ssa_op.Float_fma _
    | Ssa_op.Float_max _ | Ssa_op.Float_to_i64 _ | Ssa_op.Float_unary _
    | Ssa_op.I64_arith _ | Ssa_op.I64_compare _ | Ssa_op.I64_div _
    | Ssa_op.Index_compare _ | Ssa_op.Load _ | Ssa_op.Load_in_bounds _
    | Ssa_op.Local_alloc _ | Ssa_op.Lanewise _ | Ssa_op.Local_read _
    | Ssa_op.Local_write _ | Ssa_op.Mark _ | Ssa_op.Mark_lanes _
    | Ssa_op.Meter_charge | Ssa_op.Meter_release _ | Ssa_op.Meter_reserve _
    | Ssa_op.Meter_reset | Ssa_op.Pool_better _ | Ssa_op.Pred_not _
    | Ssa_op.Pred_or _ | Ssa_op.Store _ | Ssa_op.Vec_extract _
    | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _ | Ssa_op.Vec_load _
    | Ssa_op.Vec_splat _ | Ssa_op.Vec_store _ ->
        ()
  in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and induction ~lo ~hi ~step (iv : Ssa_value.t) =
    (* a loop that cannot run has no induction value: its body is unreachable *)
    match (get lo, get hi) with
    | Empty, _ | _, Empty -> set iv Empty
    | Range l, Range h ->
        if Int64.compare h.hi l.lo <= 0 then set iv Empty
        else if Int64.equal l.lo l.hi && Int64.equal h.lo h.hi then
          (* constant bounds: the last value the stride reaches, exactly *)
          if Int64.compare h.lo l.lo <= 0 then set iv Empty
          else
            let trips =
              Int64.div (Int64.add (Int64.sub h.lo l.lo) (Int64.pred step)) step
            in
            set iv
              (Range
                 {
                   lo = l.lo;
                   hi = Int64.add l.lo (Int64.mul (Int64.pred trips) step);
                 })
        else
          set iv (Range { lo = l.lo; hi = Stdlib.max l.lo (Int64.pred h.hi) })
  and stmt : Ssa_region.t Ssa_stmt.t -> unit = function
    | Ssa_stmt.Instr i -> op i
    | Ssa_stmt.For { lo; hi; step; body; _ } ->
        (match body.Ssa_region.params with
        | iv :: _ -> induction ~lo ~hi ~step iv
        | [] -> ());
        region body
    | Ssa_stmt.If { results; then_; else_; _ } ->
        region then_;
        region else_;
        List.iteri
          (fun k (res : Ssa_value.t) ->
            match res.Ssa_value.ty with
            | Ssa_type.Scalar (Ssa_type.Index | Ssa_type.I64) ->
                set res
                  (join
                     (get (List.nth then_.Ssa_region.yields k))
                     (get (List.nth else_.Ssa_region.yields k)))
            | _ -> ())
          results
    | Ssa_stmt.Ordered_sum { lo; hi; body; _ } ->
        (match body.Ssa_region.params with
        | iv :: _ -> induction ~lo ~hi ~step:1L iv
        | [] -> ());
        region body
  in
  region p.Ssa_program.entry;
  { ranges = !ranges }

(* ---- the claims the ranges back ---------------------------------------------- *)

let add_stays_in_domain t a b = within_domain (add (range t a) (range t b))
let scale_stays_in_domain t k a = within_domain (scale k (range t a))

(* Every coordinate component inside the buffer's extent on its axis, or the flat
   offset inside its element count. *)
let in_bounds t (b : Ssa_buffer.t) (at : Ssa_access.t) =
  match at with
  | Ssa_access.Coord c ->
      Expr.Coord.foldi
        (fun axis ok v ->
          ok
          && subset (range t v) ~lo:0L
               ~hi:(Int64.pred (Expr.Coord.get b.Ssa_buffer.extents axis)))
        true c
  | Ssa_access.Flat o -> (
      match Ssa_buffer.elements b.Ssa_buffer.extents with
      | Some n -> subset (range t o) ~lo:0L ~hi:(Int64.pred n)
      | None -> false)

(* Every lane of a vector access inside the buffer: along each axis the first
   lane's coordinate range, moved by the span the lanes' steps add, stays inside
   the extent. The steps are literals, so the span is exact. *)
let lanes_in_bounds t (b : Ssa_buffer.t) ~(at : Ssa_value.t Expr.Coord.t)
    ~(steps : int64 Expr.Coord.t) ~lanes =
  let reach = Int64.of_int (Ssa_type.Lanes.to_int lanes - 1) in
  Expr.Coord.foldi
    (fun axis ok v ->
      ok
      &&
      let span = Int64.mul reach (Expr.Coord.get steps axis) in
      match range t v with
      | Empty -> true
      | Range r ->
          let extent = Expr.Coord.get b.Ssa_buffer.extents axis in
          Int64.compare (Int64.add r.lo (Stdlib.min 0L span)) 0L >= 0
          && Int64.compare (Int64.add r.hi (Stdlib.max 0L span)) extent < 0)
    true at

(* What a loop's bounds say about how often its body runs. *)
type trips = At_least_one | Exactly of int64 | Unknown | Zero

let trips t ~lo ~hi ~step =
  match (range t lo, range t hi) with
  | Empty, _ | _, Empty -> Zero
  | Range l, Range h ->
      if Int64.equal l.lo l.hi && Int64.equal h.lo h.hi then
        if Int64.compare h.lo l.lo <= 0 then Zero
        else
          Exactly
            (Int64.div (Int64.add (Int64.sub h.lo l.lo) (Int64.pred step)) step)
      else if Int64.compare h.hi l.lo <= 0 then Zero
      else if Int64.compare h.lo l.hi > 0 then At_least_one
      else Unknown

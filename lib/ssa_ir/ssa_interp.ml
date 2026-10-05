module Counters = struct
  type t = { mutable loads : int; mutable stores : int; marks : int array }

  let create () = { loads = 0; stores = 0; marks = Array.make 6 0 }

  let slot = function
    | Ssa_mark.Emitter -> 0
    | Ssa_mark.Key -> 1
    | Ssa_mark.Local -> 2
    | Ssa_mark.Reduction -> 3
    | Ssa_mark.Scan -> 4
    | Ssa_mark.Scan_update -> 5

  let mark t m = t.marks.(slot m)
  let loads t = t.loads
  let stores t = t.stores
  let count t m = t.marks.(slot m) <- t.marks.(slot m) + 1
end

type failure =
  [ `Coord_out_of_range of Expr.Source.t * Expr.Axis.t * int * int Expr.Coord.t
  | `Gather_index_out_of_range of Expr.Eval.Gather_index_out_of_range.t
  | `I64_division_by_zero
  | `I64_division_overflow
  | `I64_from_float_infinite
  | `I64_from_float_nan
  | `I64_from_float_out_of_range of float
  | `Index_overflow of Expr.Index_overflow.t
  | `Scan_meter of Expr.Scan_meter.error
  | `Scan_projection of Expr.Eval.scan_error
  | `Unbound_local of Expr.Local_var.t ]

type error = [ failure | `Invalid_program of Ssa_verify.diagnostic ]

let pp_error fmt : [< error ] -> unit = function
  | `Coord_out_of_range (src, axis, v, c) ->
      Fmt.pf fmt "%a: coordinate %a = %d is outside the buffer, at (%a)"
        Expr.Source.pp src Expr.Axis.pp axis v (Expr.Coord.pp Fmt.int) c
  | `Gather_index_out_of_range
      { Expr.Eval.Gather_index_out_of_range.raw; extent } ->
      Fmt.pf fmt "gather index %Ld is outside [-%d, %d)" raw extent extent
  | (`I64_division_by_zero | `I64_division_overflow) as e ->
      Expr.Value.pp_i64_division_error fmt e
  | ( `I64_from_float_infinite | `I64_from_float_nan
    | `I64_from_float_out_of_range _ ) as e ->
      Expr.Value.pp_i64_from_float_error fmt e
  | `Scan_meter e -> Expr.Scan_meter.pp_error fmt e
  | `Scan_projection e -> Expr.Eval.pp_scan_error fmt e
  | `Unbound_local v ->
      Fmt.pf fmt "local %a is unbound here" Expr.Local_var.pp v
  | `Index_overflow { Expr.Index_overflow.op; lhs; rhs } ->
      Fmt.pf fmt "index %s overflows on %d and %d"
        (match op with `Add -> "add" | `Mul -> "mul" | `Sub -> "sub")
        lhs rhs
  | `Invalid_program d -> Ssa_verify.pp_diagnostic fmt d

(* A runtime value. An index and an i64 are both [I], a binary32 and a binary64
   both [F]: the verifier has already fixed which a slot holds, and a binary32
   value is kept rounded. *)
type local = {
  cells : float array;
  written : bool array;
  var : Expr.Local_var.t option;
}

type v =
  | E
  | F of float
  | I of int64
  | L of local
  | M of bool array
  | P of bool
  | V of float array

(* The scan meter, as [Expr.Scan_meter] keeps it: an update budget that fails
   BEFORE the charged body runs, and live state counted against the nesting
   peak. *)
type meter = { mutable live_state : int; mutable updates_remaining : int64 }

type st = {
  esc : failure Err.Escape.t;
  buffers : Ssa_buffer.t list;
  scan_limits : Expr.Scan_limits.t;
  memory : Ssa_memory.t;
  counters : Counters.t;
  fused : bool;
  env : v array;
  mutable meter : meter;
}

let get st (x : Ssa_value.t) = st.env.((x.Ssa_value.id :> int))
let set st (x : Ssa_value.t) v = st.env.((x.Ssa_value.id :> int)) <- v

let fresh_meter (limits : Expr.Scan_limits.t) =
  { live_state = 0; updates_remaining = Expr.Scan_limits.max_updates limits }

let local_of st x =
  match get st x with
  | L l -> l
  | E | F _ | I _ | M _ | P _ | V _ -> invalid_arg "Ssa_interp: local"

let float_of st x =
  match get st x with
  | F f -> f
  | E | I _ | L _ | M _ | P _ | V _ -> invalid_arg "Ssa_interp: float"

let int_of st x =
  match get st x with
  | I i -> i
  | E | F _ | L _ | M _ | P _ | V _ -> invalid_arg "Ssa_interp: int"

let bool_of st x =
  match get st x with
  | P b -> b
  | E | F _ | I _ | L _ | M _ | V _ -> invalid_arg "Ssa_interp: pred"

let is_f32 (x : Ssa_value.t) =
  match x.Ssa_value.ty with
  | Ssa_type.Scalar Ssa_type.F32 | Ssa_type.Vec (Ssa_type.F32, _) -> true
  | _ -> false

let buffer st id =
  match
    List.find_opt
      (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.equal b.Ssa_buffer.id id)
      st.buffers
  with
  | Some b -> b
  | None -> invalid_arg "Ssa_interp: undeclared buffer"

let cells st id =
  match Ssa_memory.find st.memory id with
  | Some c -> c
  | None -> invalid_arg "Ssa_interp: buffer is not in memory"

let coord_of st c = Expr.Coord.map (int_of st) c

(* The element offset an access names, after the checks the access owes. A
   coordinate load reports the first axis outside; a flat offset or any store
   outside is a defect, not a row. *)
let element_of_coord st (b : Ssa_buffer.t) c ~checked =
  match Ssa_memory.first_outside b.Ssa_buffer.extents c with
  | None -> Int64.to_int (Ssa_memory.offset b.Ssa_buffer.extents c)
  | Some axis ->
      if checked then
        Err.Escape.throw st.esc
          (`Coord_out_of_range
             ( Ssa_buffer.source b,
               axis,
               Int64.to_int (Expr.Coord.get c axis),
               Expr.Coord.map Int64.to_int c )
            : failure)
      else invalid_arg "Ssa_interp: store outside its buffer"

let element st (b : Ssa_buffer.t) (at : Ssa_access.t) ~checked =
  match at with
  | Ssa_access.Coord c -> element_of_coord st b (coord_of st c) ~checked
  | Ssa_access.Flat o -> (
      let o = int_of st o in
      match Ssa_buffer.elements b.Ssa_buffer.extents with
      | Some n when Int64.compare o 0L >= 0 && Int64.compare o n < 0 ->
          Int64.to_int o
      | Some _ | None ->
          invalid_arg "Ssa_interp: flat access outside its buffer")

let overflow st op lhs rhs =
  Err.Escape.throw st.esc
    (`Index_overflow
       {
         Expr.Index_overflow.op;
         lhs = Int64.to_int lhs;
         rhs = Int64.to_int rhs;
       }
      : failure)

let i64_of_float st x =
  match
    Err.payload
      (Expr.Value.i64_of_float x
        : (int64, Expr.Value.i64_from_float_error) Err.t)
  with
  | Ok v -> v
  | Error e -> Err.Escape.throw st.esc (e :> failure)

let i64_div st x y =
  match
    Err.payload
      (Expr.Value.apply_i64_binary Expr.Value.I64_div x y
        : (int64, Expr.Value.i64_division_error) Err.t)
  with
  | Ok v -> v
  | Error e -> Err.Escape.throw st.esc (e :> failure)

let checked_index st op lhs rhs r =
  if Ssa_const.in_index_domain r then I r else overflow st op lhs rhs

(* The channel a dequantizing load reads: its C coordinate, or 0 for a flat
   access, which the verifier allows only where the decode is per tensor. *)
let channel_of st = function
  | Ssa_op.Load { at = Ssa_access.Coord c; _ }
  | Ssa_op.Load_in_bounds { at = Ssa_access.Coord c; _ } ->
      Int64.to_int (int_of st c.Expr.Coord.c)
  | _ -> 0

(* One cell decoded to the working value it stands for. [channel] is the C
   coordinate a per-channel dequantization reads. *)
let read_cell st id (b : Ssa_buffer.t) decode ~at ~channel =
  match (cells st id, decode) with
  | ( Ssa_memory.Floats a,
      ( Ssa_op.Decode.Bool_to_f64 | Ssa_op.Decode.F32_to_f64
      | Ssa_op.Decode.F64_to_f64 ) ) ->
      F a.(at)
  | Ssa_memory.Int64s a, Ssa_op.Decode.I64 -> I a.(at)
  | Ssa_memory.Int64s a, Ssa_op.Decode.I64_to_f64 -> F (Int64.to_float a.(at))
  | Ssa_memory.Ints a, Ssa_op.Decode.F16_to_f64 ->
      F (Ssa_half.f16_to_float a.(at))
  | Ssa_memory.Ints a, Ssa_op.Decode.Bf16_to_f64 ->
      F (Ssa_half.bf16_to_float a.(at))
  | Ssa_memory.Ints a, Ssa_op.Decode.I32_to_f64 -> F (float_of_int a.(at))
  | Ssa_memory.Ints a, (Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I8_dequant)
    -> (
      match Ssa_format.quant b.Ssa_buffer.format with
      | Some q -> F (Ssa_format.dequantize q ~channel ~cell:a.(at))
      | None -> invalid_arg "Ssa_interp: dequantizing an unquantized buffer")
  | (Ssa_memory.Floats _ | Ssa_memory.Int64s _ | Ssa_memory.Ints _), _ ->
      invalid_arg "Ssa_interp: decode does not match the cells"

let write_cell st id encode ~at (value : v) =
  match (cells st id, encode, value) with
  | Ssa_memory.Floats a, Ssa_op.Encode.Bool_nonzero, F x ->
      a.(at) <- (if x <> 0. then 1. else 0.)
  | Ssa_memory.Floats a, Ssa_op.Encode.F32_round, F x ->
      a.(at) <- Ssa_const.round_f32 x
  | Ssa_memory.Int64s a, Ssa_op.Encode.I64, I x -> a.(at) <- x
  | _ -> invalid_arg "Ssa_interp: encode does not match the cells"

(* The coordinate lane [k] of a vector access names. *)
let lane_coord (c : int64 Expr.Coord.t) (steps : int64 Expr.Coord.t) k =
  Expr.Coord.mapi
    (fun axis x ->
      Int64.add x (Int64.mul (Int64.of_int k) (Expr.Coord.get steps axis)))
    c

(* A lane of a vector or mask operand, as the scalar value the operation lifted
   to lanes reads. *)
let lane_value st k x =
  match get st x with
  | V a -> Ssa_scalar.F a.(k)
  | M a -> Ssa_scalar.P a.(k)
  | E | F _ | I _ | L _ | P _ -> invalid_arg "Ssa_interp: a vector operand"

let lanes_of (ty : Ssa_type.t) =
  match ty with
  | Ssa_type.Vec (_, l) | Ssa_type.Mask l -> Ssa_type.Lanes.to_int l
  | Ssa_type.Effect | Ssa_type.Local | Ssa_type.Scalar _ ->
      invalid_arg "Ssa_interp: not a vector type"

let exec_op st (i : Ssa_instr.t) =
  let result =
    match i.Ssa_instr.results with
    | r :: _ -> r
    | [] -> invalid_arg "Ssa_interp: no result"
  in
  let put v = set st result v in
  let scalar x =
    match get st x with
    | F f -> Ssa_scalar.F f
    | I n -> Ssa_scalar.I n
    | P b -> Ssa_scalar.P b
    | E | L _ | M _ | V _ -> invalid_arg "Ssa_interp: a scalar operand"
  in
  match
    Ssa_scalar.eval ~fused:st.fused i.Ssa_instr.op ~result:result.Ssa_value.ty
      ~get:scalar
  with
  | Some (Ssa_scalar.F f) -> put (F f)
  | Some (Ssa_scalar.I n) -> put (I n)
  | Some (Ssa_scalar.P b) -> put (P b)
  | None -> (
      match i.Ssa_instr.op with
      | Ssa_op.Check_access { buffer = id; at } ->
          ignore (element st (buffer st id) at ~checked:true)
      | Ssa_op.Check_local { var; at; extent } ->
          let at = int_of st at in
          if Int64.compare at 0L < 0 || Int64.compare at extent >= 0 then
            Err.Escape.throw st.esc (`Unbound_local var : failure)
      | Ssa_op.Check_scan { var; row; lane; row_extent; lane_extent } ->
          let row = Int64.to_int (int_of st row) in
          let lane = Int64.to_int (int_of st lane) in
          let projection =
            { Expr.Eval.Scan_projection.local = var; row; lane }
          in
          let bounds extent = { Expr.Eval.Scan_bounds.projection; extent } in
          if row < 0 || Int64.compare (Int64.of_int row) row_extent >= 0 then
            Err.Escape.throw st.esc
              (`Scan_projection
                 (Expr.Eval.Row_out_of_range (bounds (Int64.to_int row_extent)))
                : failure)
          else if lane < 0 || Int64.compare (Int64.of_int lane) lane_extent >= 0
          then
            Err.Escape.throw st.esc
              (`Scan_projection
                 (Expr.Eval.Lane_out_of_range
                    (bounds (Int64.to_int lane_extent)))
                : failure)
      | Ssa_op.Local_alloc { slots; var } ->
          let n = Int64.to_int slots in
          put (L { cells = Array.make n 0.; written = Array.make n false; var })
      | Ssa_op.Local_read { local; at } ->
          let l = local_of st local in
          let at = int_of st at in
          let n = Array.length l.cells in
          if Int64.compare at 0L < 0 || Int64.compare at (Int64.of_int n) >= 0
          then
            match l.var with
            | Some v -> Err.Escape.throw st.esc (`Unbound_local v : failure)
            | None -> invalid_arg "Ssa_interp: a local read outside its object"
          else
            let at = Int64.to_int at in
            if not l.written.(at) then
              invalid_arg
                "Ssa_interp: a read of a local cell that was never written"
            else put (F l.cells.(at))
      | Ssa_op.Local_write { local; at; value } ->
          let l = local_of st local in
          let at = int_of st at in
          let n = Array.length l.cells in
          if Int64.compare at 0L < 0 || Int64.compare at (Int64.of_int n) >= 0
          then invalid_arg "Ssa_interp: a local write outside its object"
          else
            let at = Int64.to_int at in
            l.cells.(at) <- float_of st value;
            l.written.(at) <- true
      | Ssa_op.Meter_charge ->
          if Int64.compare st.meter.updates_remaining 0L <= 0 then
            Err.Escape.throw st.esc
              (`Scan_meter
                 (Expr.Scan_meter.Updates_exhausted
                    { limit = Expr.Scan_limits.max_updates st.scan_limits })
                : failure)
          else
            st.meter.updates_remaining <-
              Int64.sub st.meter.updates_remaining 1L
      | Ssa_op.Meter_release width ->
          st.meter.live_state <- st.meter.live_state - (2 * Int64.to_int width)
      | Ssa_op.Meter_reserve width ->
          let need = 2 * Int64.to_int width in
          let limits = st.scan_limits in
          if st.meter.live_state + need > Expr.Scan_limits.max_state limits then
            Err.Escape.throw st.esc
              (`Scan_meter
                 (Expr.Scan_meter.State_over_limit
                    { limit = Expr.Scan_limits.max_state limits })
                : failure)
          else st.meter.live_state <- st.meter.live_state + need
      | Ssa_op.Meter_reset -> st.meter <- fresh_meter st.scan_limits
      | Ssa_op.Check_gather { raw; extent } ->
          let raw = int_of st raw in
          let bound = extent in
          if
            Int64.compare raw (Int64.neg bound) < 0
            || not (Int64.compare raw bound < 0)
          then
            Err.Escape.throw st.esc
              (`Gather_index_out_of_range
                 {
                   Expr.Eval.Gather_index_out_of_range.raw;
                   extent = Int64.to_int extent;
                 }
                : failure)
      | Ssa_op.Float_to_i64 a -> put (I (i64_of_float st (float_of st a)))
      | Ssa_op.I64_div (a, b) ->
          let x = int_of st a in
          let y = int_of st b in
          put (I (i64_div st x y))
      | Ssa_op.Index_add (a, b) ->
          let x = int_of st a in
          let y = int_of st b in
          put (checked_index st `Add x y (Int64.add x y))
      | Ssa_op.Index_add_in_domain (a, b) ->
          let r = Int64.add (int_of st a) (int_of st b) in
          if Ssa_const.in_index_domain r then put (I r)
          else invalid_arg "Ssa_interp: an unchecked sum left the index domain"
      | Ssa_op.Index_scale_in_domain (k, a) ->
          let r = Int64.mul k (int_of st a) in
          if Ssa_const.in_index_domain r then put (I r)
          else
            invalid_arg "Ssa_interp: an unchecked product left the index domain"
      | Ssa_op.Index_of_i64 a ->
          let x = int_of st a in
          if Ssa_const.in_index_domain x then put (I x)
          else
            invalid_arg "Ssa_interp: an int64 outside the index domain narrowed"
      | Ssa_op.Index_scale (k, a) ->
          let y = int_of st a in
          put (checked_index st `Mul k y (Int64.mul k y))
      | Ssa_op.Load { buffer = id; at; decode }
      | Ssa_op.Load_in_bounds { buffer = id; at; decode } ->
          let b = buffer st id in
          (* a load proved in bounds checks nothing: outside is a defect of the
         proof, as for a store *)
          let checked =
            match i.Ssa_instr.op with
            | Ssa_op.Load_in_bounds _ -> false
            | _ -> true
          in
          let at = element st b at ~checked in
          st.counters.Counters.loads <- st.counters.Counters.loads + 1;
          put
            (read_cell st id b decode ~at
               ~channel:(channel_of st i.Ssa_instr.op))
      | Ssa_op.Lanewise inner -> (
          let n = lanes_of result.Ssa_value.ty in
          let lane_ty =
            match result.Ssa_value.ty with
            | Ssa_type.Vec (s, _) -> Ssa_type.Scalar s
            | _ -> Ssa_type.Scalar Ssa_type.Pred
          in
          let lane k =
            match
              Ssa_scalar.eval ~fused:st.fused inner ~result:lane_ty
                ~get:(lane_value st k)
            with
            | Some v -> v
            | None ->
                invalid_arg "Ssa_interp: a lane-wise operation is not pure"
          in
          match result.Ssa_value.ty with
          | Ssa_type.Vec _ ->
              put
                (V
                   (Array.init n (fun k ->
                        match lane k with
                        | Ssa_scalar.F x -> x
                        | Ssa_scalar.I _ | Ssa_scalar.P _ ->
                            invalid_arg "Ssa_interp: a lane is not a float")))
          | _ ->
              put
                (M
                   (Array.init n (fun k ->
                        match lane k with
                        | Ssa_scalar.P b -> b
                        | Ssa_scalar.F _ | Ssa_scalar.I _ ->
                            invalid_arg "Ssa_interp: a lane is not a predicate")))
          )
      | Ssa_op.Mark m -> Counters.count st.counters m
      | Ssa_op.Mark_lanes { mark; lanes } ->
          for _ = 1 to Ssa_type.Lanes.to_int lanes do
            Counters.count st.counters mark
          done
      | Ssa_op.Vec_extract { lane; vector } -> (
          let k = Ssa_type.Lane.to_int lane in
          match get st vector with
          | V a -> put (F a.(k))
          | M a -> put (P a.(k))
          | E | F _ | I _ | L _ | P _ -> invalid_arg "Ssa_interp: a vector")
      | Ssa_op.Vec_insert { lane; vector; element } -> (
          let k = Ssa_type.Lane.to_int lane in
          match (get st vector, get st element) with
          | V a, F x ->
              let a = Array.copy a in
              a.(k) <- x;
              put (V a)
          | M a, P x ->
              let a = Array.copy a in
              a.(k) <- x;
              put (M a)
          | _ -> invalid_arg "Ssa_interp: a vector insert")
      | Ssa_op.Vec_iota { base; step; lanes } ->
          let base = int_of st base in
          put
            (V
               (Array.init (Ssa_type.Lanes.to_int lanes) (fun k ->
                    Int64.to_float
                      (Int64.add base (Int64.mul (Int64.of_int k) step)))))
      | Ssa_op.Vec_splat { element; lanes } -> (
          let n = Ssa_type.Lanes.to_int lanes in
          match get st element with
          | F x -> put (V (Array.make n x))
          | P b -> put (M (Array.make n b))
          | E | I _ | L _ | M _ | V _ -> invalid_arg "Ssa_interp: a splat")
      | Ssa_op.Vec_load { buffer = id; at; steps; decode; lanes } ->
          let b = buffer st id in
          let n = Ssa_type.Lanes.to_int lanes in
          let first = coord_of st at in
          let cells =
            Array.init n (fun k ->
                let c = lane_coord first steps k in
                let at = element_of_coord st b c ~checked:false in
                st.counters.Counters.loads <- st.counters.Counters.loads + 1;
                match
                  read_cell st id b decode ~at
                    ~channel:(Int64.to_int c.Expr.Coord.c)
                with
                | F x -> x
                | E | I _ | L _ | M _ | P _ | V _ ->
                    invalid_arg "Ssa_interp: a vector load of a non-float")
          in
          put (V cells)
      | Ssa_op.Vec_store { buffer = id; at; steps; encode; value; lanes } -> (
          let b = buffer st id in
          let first = coord_of st at in
          match get st value with
          | V values ->
              for k = 0 to Ssa_type.Lanes.to_int lanes - 1 do
                let at =
                  element_of_coord st b (lane_coord first steps k)
                    ~checked:false
                in
                st.counters.Counters.stores <- st.counters.Counters.stores + 1;
                write_cell st id encode ~at (F values.(k))
              done
          | E | F _ | I _ | L _ | M _ | P _ ->
              invalid_arg "Ssa_interp: a vector store")
      | Ssa_op.Store { buffer = id; at; encode; value } ->
          let b = buffer st id in
          let at = element st b at ~checked:false in
          st.counters.Counters.stores <- st.counters.Counters.stores + 1;
          write_cell st id encode ~at (get st value)
      | Ssa_op.Const _ | Ssa_op.Convert _ | Ssa_op.Float_binary _
      | Ssa_op.Float_compare _ | Ssa_op.Float_fma _ | Ssa_op.Float_max _
      | Ssa_op.Float_unary _ | Ssa_op.I64_arith _ | Ssa_op.I64_compare _
      | Ssa_op.Index_ceil_div _ | Ssa_op.Index_clamp_low _
      | Ssa_op.Index_compare _ | Ssa_op.Index_floor_div _ | Ssa_op.Index_max _
      | Ssa_op.Index_min _ | Ssa_op.Pool_better _ | Ssa_op.Pred_not _
      | Ssa_op.Pred_or _ | Ssa_op.Select _ ->
          (* evaluated by [Ssa_scalar] above, which is total on them *)
          invalid_arg "Ssa_interp: a pure operation reached the effectful cases"
      )

(* Run a region's statements; the caller reads the yields. *)
let rec region st (r : Ssa_region.t) = List.iter (stmt st) r.Ssa_region.body
and yields st (r : Ssa_region.t) = List.map (get st) r.Ssa_region.yields

and stmt st : Ssa_region.t Ssa_stmt.t -> unit = function
  | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
      let lo = int_of st lo in
      let hi = int_of st hi in
      let iv, carried =
        match body.Ssa_region.params with
        | iv :: carried -> (iv, carried)
        | [] -> invalid_arg "Ssa_interp: loop without an induction value"
      in
      let state = ref (List.map (get st) inits) in
      let i = ref lo in
      while Int64.compare !i hi < 0 do
        set st iv (I !i);
        List.iter2 (set st) carried !state;
        region st body;
        (* every yield is read before any parameter is rebound *)
        state := yields st body;
        i := Int64.add !i step
      done;
      List.iter2 (set st) results !state
  | Ssa_stmt.If { cond; results; then_; else_ } ->
      let r = if bool_of st cond then then_ else else_ in
      region st r;
      List.iter2 (set st) results (yields st r)
  | Ssa_stmt.Instr i -> exec_op st i
  | Ssa_stmt.Ordered_sum { lo; hi; seed; token = _; results; body } ->
      let lo = int_of st lo in
      let hi = int_of st hi in
      let iv =
        match body.Ssa_region.params with
        | iv :: _ -> iv
        | [] -> invalid_arg "Ssa_interp: sum without an induction value"
      in
      let narrow = if is_f32 seed then Ssa_const.round_f32 else Fun.id in
      let acc = ref (get st seed) in
      let i = ref lo in
      while Int64.compare !i hi < 0 do
        set st iv (I !i);
        List.iter
          (fun p ->
            if Ssa_type.equal p.Ssa_value.ty Ssa_type.Effect then set st p E)
          body.Ssa_region.params;
        region st body;
        (match (yields st body, !acc) with
        | F term :: _, F sum -> acc := F (narrow (sum +. term))
        | V term :: _, V sum ->
            (* each lane is its own left fold *)
            acc := V (Array.mapi (fun k s -> narrow (s +. term.(k))) sum)
        | _ -> invalid_arg "Ssa_interp: a sum term is not a float");
        i := Int64.add !i 1L
      done;
      List.iter2 (set st) results [ !acc; E ]

let run ?(counters = Counters.create ()) ?(fused = true) (p : Ssa_program.t)
    ~memory =
  match Err.payload (Ssa_verify.check p) with
  | Error (`Invalid_program d) -> Err.fail (`Invalid_program d : error)
  | Ok () ->
      Err.map_error
        (fun (f : failure) -> (f :> error))
        ( Err.Escape.with_escape @@ fun esc ->
          let st =
            {
              esc;
              buffers = p.Ssa_program.buffers;
              scan_limits = p.Ssa_program.scan_limits;
              memory;
              counters;
              fused;
              env = Array.make (p.Ssa_program.next_value :> int) E;
              meter = fresh_meter p.Ssa_program.scan_limits;
            }
          in
          region st p.Ssa_program.entry )

(* The machine without its control flow: a consumer that sequences the
   operations itself (the CFG interpreter) runs each through [exec], so every
   operation means exactly what it means here. *)
module Machine = struct
  type nonrec t = st
  type value = v

  let run ?(counters = Counters.create ()) ?(fused = true) ~buffers ~scan_limits
      ~next_value ~memory body =
    Err.map_error
      (fun (f : failure) -> (f :> error))
      ( Err.Escape.with_escape @@ fun esc ->
        body
          {
            esc;
            buffers;
            scan_limits;
            memory;
            counters;
            fused;
            env = Array.make (next_value : Ssa_id.Value.Next.t :> int) E;
            meter = fresh_meter scan_limits;
          } )

  let exec = exec_op
  let read = get
  let write = set
  let index = int_of
  let predicate = bool_of
  let effect_value = E
end

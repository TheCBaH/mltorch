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

type v = E | F of float | I of int64 | L of local | P of bool

(* The scan meter, as [Expr.Scan_meter] keeps it: an update budget that fails
   BEFORE the charged body runs, and live state counted against the nesting
   peak. *)
type meter = { mutable live_state : int; mutable updates_remaining : int64 }

type st = {
  esc : failure Err.Escape.t;
  program : Ssa_program.t;
  memory : Ssa_memory.t;
  counters : Counters.t;
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
  | E | F _ | I _ | P _ -> invalid_arg "Ssa_interp: local"

let float_of st x =
  match get st x with
  | F f -> f
  | E | I _ | L _ | P _ -> invalid_arg "Ssa_interp: float"

let int_of st x =
  match get st x with
  | I i -> i
  | E | F _ | L _ | P _ -> invalid_arg "Ssa_interp: int"

let bool_of st x =
  match get st x with
  | P b -> b
  | E | F _ | I _ | L _ -> invalid_arg "Ssa_interp: pred"

let is_f32 (x : Ssa_value.t) =
  Ssa_type.equal x.Ssa_value.ty (Ssa_type.Scalar Ssa_type.F32)

let const : Ssa_const.t -> v = function
  | Ssa_const.F32 x | Ssa_const.F64 x -> F x
  | Ssa_const.I64 x | Ssa_const.Index x -> I x
  | Ssa_const.Pred b -> P b

let buffer st id =
  match Ssa_program.find_buffer st.program id with
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
let element st (b : Ssa_buffer.t) (at : Ssa_access.t) ~checked =
  match at with
  | Ssa_access.Coord c -> (
      let c = coord_of st c in
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
          else invalid_arg "Ssa_interp: store outside its buffer")
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

let floor_div n d =
  let q = Int64.div n d and r = Int64.rem n d in
  if Int64.compare r 0L < 0 then Int64.pred q else q

let ceil_div n d = Int64.neg (floor_div (Int64.neg n) d)

(* The channel a dequantizing load reads: its C coordinate, or 0 for a flat
   access, which the verifier allows only where the decode is per tensor. *)
let channel_of st = function
  | Ssa_op.Load { at = Ssa_access.Coord c; _ } ->
      Int64.to_int (int_of st c.Expr.Coord.c)
  | _ -> 0

let exec_op st (i : Ssa_instr.t) =
  let result =
    match i.Ssa_instr.results with
    | r :: _ -> r
    | [] -> invalid_arg "Ssa_interp: no result"
  in
  let put v = set st result v in
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
      let projection = { Expr.Eval.Scan_projection.local = var; row; lane } in
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
             (Expr.Eval.Lane_out_of_range (bounds (Int64.to_int lane_extent)))
            : failure)
  | Ssa_op.Local_alloc { slots; var } ->
      let n = Int64.to_int slots in
      put (L { cells = Array.make n 0.; written = Array.make n false; var })
  | Ssa_op.Local_read { local; at } ->
      let l = local_of st local in
      let at = int_of st at in
      let n = Array.length l.cells in
      if Int64.compare at 0L < 0 || Int64.compare at (Int64.of_int n) >= 0 then
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
      if Int64.compare at 0L < 0 || Int64.compare at (Int64.of_int n) >= 0 then
        invalid_arg "Ssa_interp: a local write outside its object"
      else
        let at = Int64.to_int at in
        l.cells.(at) <- float_of st value;
        l.written.(at) <- true
  | Ssa_op.Meter_charge ->
      if Int64.compare st.meter.updates_remaining 0L <= 0 then
        Err.Escape.throw st.esc
          (`Scan_meter
             (Expr.Scan_meter.Updates_exhausted
                {
                  limit =
                    Expr.Scan_limits.max_updates
                      st.program.Ssa_program.scan_limits;
                })
            : failure)
      else st.meter.updates_remaining <- Int64.sub st.meter.updates_remaining 1L
  | Ssa_op.Meter_release width ->
      st.meter.live_state <- st.meter.live_state - (2 * Int64.to_int width)
  | Ssa_op.Meter_reserve width ->
      let need = 2 * Int64.to_int width in
      let limits = st.program.Ssa_program.scan_limits in
      if st.meter.live_state + need > Expr.Scan_limits.max_state limits then
        Err.Escape.throw st.esc
          (`Scan_meter
             (Expr.Scan_meter.State_over_limit
                { limit = Expr.Scan_limits.max_state limits })
            : failure)
      else st.meter.live_state <- st.meter.live_state + need
  | Ssa_op.Meter_reset ->
      st.meter <- fresh_meter st.program.Ssa_program.scan_limits
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
  | Ssa_op.Const c -> put (const c)
  | Ssa_op.Convert (c, a) -> (
      match c with
      | Ssa_op.Convert.F32_to_f64 -> put (F (float_of st a))
      | Ssa_op.Convert.F64_to_f32 ->
          put (F (Ssa_const.round_f32 (float_of st a)))
      | Ssa_op.Convert.I64_to_f32 ->
          put (F (Ssa_const.round32_of_i64 (int_of st a)))
      | Ssa_op.Convert.I64_to_f64 -> put (F (Int64.to_float (int_of st a)))
      | Ssa_op.Convert.Index_to_f64 -> put (F (Int64.to_float (int_of st a)))
      | Ssa_op.Convert.Index_to_i64 -> put (I (int_of st a)))
  | Ssa_op.Float_binary (op, a, b) ->
      let x = float_of st a in
      let y = float_of st b in
      let r = Expr.Value.apply_binary op x y in
      put (F (if is_f32 result then Ssa_const.round_f32 r else r))
  | Ssa_op.Float_compare (c, a, b) ->
      let x = float_of st a in
      let y = float_of st b in
      put
        (P
           (match c with
           | Ssa_op.Compare.Eq -> x = y
           | Ssa_op.Compare.Lt -> x < y))
  | Ssa_op.Float_max (a, b) ->
      let x = float_of st a in
      let y = float_of st b in
      put (F (Expr.Max_op.apply Expr.Max_op.Float_max x y))
  | Ssa_op.Float_to_i64 a -> put (I (i64_of_float st (float_of st a)))
  | Ssa_op.Float_unary (op, a) ->
      put (F (Expr.Value.apply_unary op (float_of st a)))
  | Ssa_op.I64_arith (op, a, b) ->
      let x = int_of st a in
      let y = int_of st b in
      put
        (I
           (match op with
           | Ssa_op.I64_op.Add -> Int64.add x y
           | Ssa_op.I64_op.Mul -> Int64.mul x y
           | Ssa_op.I64_op.Sub -> Int64.sub x y))
  | Ssa_op.I64_compare (c, a, b) ->
      let x = int_of st a in
      let y = int_of st b in
      put
        (P
           (match c with
           | Ssa_op.Compare.Eq -> Int64.equal x y
           | Ssa_op.Compare.Lt -> Int64.compare x y < 0))
  | Ssa_op.I64_div (a, b) ->
      let x = int_of st a in
      let y = int_of st b in
      put (I (i64_div st x y))
  | Ssa_op.Index_add (a, b) ->
      let x = int_of st a in
      let y = int_of st b in
      put (checked_index st `Add x y (Int64.add x y))
  | Ssa_op.Index_ceil_div (k, a) -> put (I (ceil_div (int_of st a) k))
  | Ssa_op.Index_clamp_low a -> put (I (Stdlib.max 0L (int_of st a)))
  | Ssa_op.Index_compare (c, a, b) ->
      let x = int_of st a in
      let y = int_of st b in
      put
        (P
           (match c with
           | Ssa_op.Compare.Eq -> Int64.equal x y
           | Ssa_op.Compare.Lt -> Int64.compare x y < 0))
  | Ssa_op.Index_floor_div (k, a) -> put (I (floor_div (int_of st a) k))
  | Ssa_op.Index_max (a, b) ->
      let x = int_of st a in
      let y = int_of st b in
      put (I (Stdlib.max x y))
  | Ssa_op.Index_min (a, b) ->
      let x = int_of st a in
      let y = int_of st b in
      put (I (Stdlib.min x y))
  | Ssa_op.Index_of_i64 a ->
      let x = int_of st a in
      if Ssa_const.in_index_domain x then put (I x)
      else invalid_arg "Ssa_interp: an int64 outside the index domain narrowed"
  | Ssa_op.Index_scale (k, a) ->
      let y = int_of st a in
      put (checked_index st `Mul k y (Int64.mul k y))
  | Ssa_op.Load { buffer = id; at; decode } -> (
      let b = buffer st id in
      let at = element st b at ~checked:true in
      st.counters.Counters.loads <- st.counters.Counters.loads + 1;
      match (cells st id, decode) with
      | ( Ssa_memory.Floats a,
          ( Ssa_op.Decode.Bool_to_f64 | Ssa_op.Decode.F32_to_f64
          | Ssa_op.Decode.F64_to_f64 ) ) ->
          put (F a.(at))
      | Ssa_memory.Int64s a, Ssa_op.Decode.I64 -> put (I a.(at))
      | Ssa_memory.Int64s a, Ssa_op.Decode.I64_to_f64 ->
          put (F (Int64.to_float a.(at)))
      | Ssa_memory.Ints a, Ssa_op.Decode.F16_to_f64 ->
          put (F (Ssa_half.f16_to_float a.(at)))
      | Ssa_memory.Ints a, Ssa_op.Decode.Bf16_to_f64 ->
          put (F (Ssa_half.bf16_to_float a.(at)))
      | Ssa_memory.Ints a, Ssa_op.Decode.I32_to_f64 ->
          put (F (float_of_int a.(at)))
      | Ssa_memory.Ints a, (Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I8_dequant)
        -> (
          match Ssa_format.quant b.Ssa_buffer.format with
          | Some q ->
              put
                (F
                   (Ssa_format.dequantize q
                      ~channel:(channel_of st i.Ssa_instr.op)
                      ~cell:a.(at)))
          | None -> invalid_arg "Ssa_interp: dequantizing an unquantized buffer"
          )
      | (Ssa_memory.Floats _ | Ssa_memory.Int64s _ | Ssa_memory.Ints _), _ ->
          invalid_arg "Ssa_interp: decode does not match the cells")
  | Ssa_op.Mark m -> Counters.count st.counters m
  | Ssa_op.Pool_better (a, b) ->
      let best = float_of st a in
      let value = float_of st b in
      put (P (Expr.Max_op.pool_better ~best ~value))
  | Ssa_op.Pred_not a -> put (P (not (bool_of st a)))
  | Ssa_op.Pred_or (a, b) ->
      let x = bool_of st a in
      let y = bool_of st b in
      put (P (x || y))
  | Ssa_op.Select (p, a, b) -> put (get st (if bool_of st p then a else b))
  | Ssa_op.Store { buffer = id; at; encode; value } -> (
      let b = buffer st id in
      let at = element st b at ~checked:false in
      st.counters.Counters.stores <- st.counters.Counters.stores + 1;
      match (cells st id, encode) with
      | Ssa_memory.Floats a, Ssa_op.Encode.Bool_nonzero ->
          a.(at) <- (if float_of st value <> 0. then 1. else 0.)
      | Ssa_memory.Floats a, Ssa_op.Encode.F32_round ->
          a.(at) <- Ssa_const.round_f32 (float_of st value)
      | Ssa_memory.Int64s a, Ssa_op.Encode.I64 -> a.(at) <- int_of st value
      | (Ssa_memory.Floats _ | Ssa_memory.Int64s _ | Ssa_memory.Ints _), _ ->
          invalid_arg "Ssa_interp: encode does not match the cells")

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
      let acc = ref (float_of st seed) in
      let i = ref lo in
      while Int64.compare !i hi < 0 do
        set st iv (I !i);
        List.iter
          (fun p ->
            if Ssa_type.equal p.Ssa_value.ty Ssa_type.Effect then set st p E)
          body.Ssa_region.params;
        region st body;
        (match yields st body with
        | F term :: _ -> acc := narrow (!acc +. term)
        | _ -> invalid_arg "Ssa_interp: a sum term is not a float");
        i := Int64.add !i 1L
      done;
      List.iter2 (set st) results [ F !acc; E ]

let run ?(counters = Counters.create ()) (p : Ssa_program.t) ~memory =
  match Err.payload (Ssa_verify.check p) with
  | Error (`Invalid_program d) -> Err.fail (`Invalid_program d : error)
  | Ok () ->
      Err.map_error
        (fun (f : failure) -> (f :> error))
        ( Err.Escape.with_escape @@ fun esc ->
          let st =
            {
              esc;
              program = p;
              memory;
              counters;
              env = Array.make (p.Ssa_program.next_value :> int) E;
              meter = fresh_meter p.Ssa_program.scan_limits;
            }
          in
          region st p.Ssa_program.entry )

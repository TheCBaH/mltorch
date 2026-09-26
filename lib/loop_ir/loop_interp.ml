type error =
  [ Expr.Eval.error
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Unbound_input of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | #Expr.Eval.error as e -> Expr.Eval.pp_error fmt e
  | `Binding_mismatch m -> Kernel_eval.Binding_mismatch.pp fmt m
  | `Unbound_input id -> Fmt.pf fmt "no binding for input %a" Tensor_id.pp id

type counters = {
  mutable emitters : int;
  mutable keys : int;
  mutable loads : int;
  mutable locals : int;
  mutable reductions : int;
  mutable scan_updates : int;
  mutable scans : int;
}

let counters () =
  {
    emitters = 0;
    keys = 0;
    loads = 0;
    locals = 0;
    reductions = 0;
    scan_updates = 0;
    scans = 0;
  }

(* The scan meter, as [Expr.Scan_meter] keeps it: an update budget that fails
   BEFORE the charged body runs, and live state counted against the nesting
   peak. [Expr.Scan_meter]'s own reservation is not exported, so the state half
   is transcribed here from the same rule and the same limit. *)
type meter = { mutable live_state : int; mutable updates_remaining : int64 }

type state = {
  esc : error Err.Escape.t;
  limits : Expr.Scan_limits.t;
  mutable meter : meter;
  counters : counters;
  buffers : Tensor.packed Tensor_id.Map.t;
  vars : (int, int) Hashtbl.t;
  floats : (int, float) Hashtbl.t;
  int64s : (int, int64) Hashtbl.t;
  indices : (int, int) Hashtbl.t;
  arrays : (int, float array) Hashtbl.t;
}

let fresh_meter limits =
  { live_state = 0; updates_remaining = Expr.Scan_limits.max_updates limits }

let lookup table key what =
  match Hashtbl.find_opt table key with
  | Some v -> v
  | None -> invalid_arg ("Loop_interp: read of unassigned " ^ what)

(* ---- index arithmetic -------------------------------------------------------

   Exact, in [int64] with saturation: an index never wraps, so a value that left
   the domain is still visibly outside it whatever the host's [int] is (32 bits
   under js_of_ocaml). The domain itself is [Loop_range.domain], and it is what
   an explicit [Fail_if] guards: the interpreter reports an out-of-domain
   [Add]/[Scale] only when a program asks, through [Index_overflows] or a
   [Loop_failure.Index_overflow] site. An unguarded out-of-domain index that
   reaches a use is a defect in the program ([narrow] below), which is what makes
   the guard a check that can be seen to matter.

   The reported row is [Expr.Eval.index]'s own, [{op; lhs; rhs}] of the first
   operation that left the domain, which under js_of_ocaml is exactly where the
   language's checked [int] arithmetic reports it. *)

let in_domain x = Loop_range.(within ~inner:{ lo = x; hi = x } ~outer:domain)

let floor_div n d =
  let q = Int64.div n d and r = Int64.rem n d in
  if Int64.compare r 0L < 0 then Int64.pred q else q

let ceil_div n d = Int64.neg (floor_div (Int64.neg n) d)

let positive_divisor d =
  if d <= 0 then invalid_arg "Loop_interp: non-positive divisor"
  else Int64.of_int d

let rec index_with ~overflow st : Loop_index.t -> int64 = function
  | Loop_index.Add (a, b) ->
      let x = index_with ~overflow st a in
      let y = index_with ~overflow st b in
      let r = Loop_range.saturating_add x y in
      if not (in_domain r) then overflow `Add x y;
      r
  | Loop_index.Ceil_div_pos (a, d) ->
      ceil_div (index_with ~overflow st a) (positive_divisor d)
  | Loop_index.Clamp_low a -> Stdlib.max 0L (index_with ~overflow st a)
  | Loop_index.Const n -> Int64.of_int n
  | Loop_index.Floor_div_pos (a, d) ->
      floor_div (index_with ~overflow st a) (positive_divisor d)
  | Loop_index.Max (a, b) ->
      let x = index_with ~overflow st a in
      let y = index_with ~overflow st b in
      Stdlib.max x y
  | Loop_index.Min (a, b) ->
      let x = index_with ~overflow st a in
      let y = index_with ~overflow st b in
      Stdlib.min x y
  | Loop_index.Scale (k, a) ->
      let y = index_with ~overflow st a in
      let r = Loop_range.saturating_mul k y in
      if not (in_domain r) then overflow `Mul (Int64.of_int k) y;
      r
  | Loop_index.Temp t ->
      Int64.of_int (lookup st.indices (Loop_temp.to_int t) "index temporary")
  | Loop_index.Var v ->
      Int64.of_int (lookup st.vars (Loop_var.to_int v) "loop variable")

let unguarded _ _ _ = ()
let index st i = index_with ~overflow:unguarded st i

(* Narrowing to [int] happens only after the domain check, never before. *)
let narrow v =
  if in_domain v then Int64.to_int v
  else invalid_arg "Loop_interp: unguarded index outside the index domain"

let idx st i = narrow (index st i)

let overflows st i =
  match
    Err.payload
      (Err.Escape.with_escape (fun stop ->
           ignore
             (index_with ~overflow:(fun _ _ _ -> Err.Escape.throw stop ()) st i)))
  with
  | Ok () -> false
  | Error () -> true

let coord st (c : Loop_index.coord) : int Expr.Coord.t =
  (* Axis order, as [Expr.Eval] evaluates a load's components. *)
  let n = idx st c.Expr.Coord.n in
  let t = idx st c.Expr.Coord.t in
  let d = idx st c.Expr.Coord.d in
  let h = idx st c.Expr.Coord.h in
  let w = idx st c.Expr.Coord.w in
  let c = idx st c.Expr.Coord.c in
  Expr.Coord.make ~n ~t ~d ~h ~w ~c

(* The first axis (in [Expr.Axis.all] order) whose component lies outside the
   buffer's shape, as [Expr_bridge.bound_in_range] finds it. *)
let first_out_of_range (sg : Tensor_sig.t) (c : int Expr.Coord.t) =
  List.find_opt
    (fun a ->
      let i = Expr.Coord.get c a in
      i < 0 || i >= Dim.to_int (Vec6.get sg.Tensor_sig.shape a))
    Expr.Axis.all

let buffer_tensor st (b : Loop_buffer.t) =
  match Tensor_id.Map.find_opt b.Loop_buffer.id st.buffers with
  | Some t -> t
  | None -> invalid_arg "Loop_interp: buffer is not bound"

let checked_coord st (b : Loop_buffer.t) c =
  let c = coord st c in
  (match first_out_of_range b.Loop_buffer.sg c with
  | Some _ -> invalid_arg "Loop_interp: unchecked access out of range"
  | None -> ());
  c

let round_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let rec eval : type a. state -> a Loop_expr.t -> a =
 fun st -> function
  | Loop_expr.Array_get (a, i) ->
      let arr = lookup st.arrays (Loop_array.to_int a) "array" in
      let i = idx st i in
      if i < 0 || i >= Array.length arr then
        invalid_arg "Loop_interp: unchecked array read out of range"
      else arr.(i)
  | Loop_expr.Binary (op, a, b) ->
      let x = eval st a in
      let y = eval st b in
      Expr.Value.apply_binary op x y
  | Loop_expr.Const x -> x
  | Loop_expr.Float_max (a, b) ->
      let x = eval st a in
      let y = eval st b in
      Expr.Max_op.apply Expr.Max_op.Float_max x y
  | Loop_expr.Float_to_i64 a ->
      Err.Escape.or_throw st.esc
        (Err.map_error
           (fun (e : Expr.Value.i64_from_float_error) -> (e :> error))
           (Expr.Value.i64_of_float (eval st a)))
  | Loop_expr.I64_binary (op, a, b) ->
      let x = eval st a in
      let y = eval st b in
      Err.Escape.or_throw st.esc
        (Err.map_error
           (fun (e : Expr.Value.i64_division_error) -> (e :> error))
           (Expr.Value.apply_i64_binary op x y))
  | Loop_expr.I64_const n -> n
  | Loop_expr.I64_of_index i -> index st i
  | Loop_expr.I64_to_float a -> Int64.to_float (eval st a)
  | Loop_expr.Load (b, c) ->
      let c = checked_coord st b c in
      st.counters.loads <- st.counters.loads + 1;
      Tensor.read_at_raw (buffer_tensor st b) (fun a -> Expr.Coord.get c a)
  | Loop_expr.Load_i64 (b, c) -> (
      let c = checked_coord st b c in
      st.counters.loads <- st.counters.loads + 1;
      match
        Tensor.read_i64_at6 (buffer_tensor st b) (fun a -> Expr.Coord.get c a)
      with
      | Ok v -> v
      | Error _ -> invalid_arg "Loop_interp: load_i64 of a non-I64 buffer")
  | Loop_expr.Round_f32 a -> round_f32 (eval st a)
  | Loop_expr.Select (p, a, b) -> if pred st p then eval st a else eval st b
  | Loop_expr.Temp (Loop_carrier.Float, t) ->
      lookup st.floats (Loop_temp.to_int t) "float temporary"
  | Loop_expr.Temp (Loop_carrier.Int64, t) ->
      lookup st.int64s (Loop_temp.to_int t) "int64 temporary"
  | Loop_expr.Unary (op, a) -> Expr.Value.apply_unary op (eval st a)
  | Loop_expr.Value_of_index i ->
      let i = idx st i in
      Err.Escape.or_throw st.esc
        (Err.map_error
           (fun (e : Expr.Eval.index_error) -> (e :> error))
           (Expr.Eval.float_of_index i))

and pred st : Loop_expr.pred -> bool = function
  | Loop_bool.I64_eq (a, b) ->
      let x = eval st a in
      let y = eval st b in
      Int64.equal x y
  | Loop_bool.I64_lt (a, b) ->
      let x = eval st a in
      let y = eval st b in
      Int64.compare x y < 0
  | Loop_bool.Index_eq (a, b) ->
      let x = index st a in
      let y = index st b in
      Int64.equal x y
  | Loop_bool.Index_lt (a, b) ->
      let x = index st a in
      let y = index st b in
      Int64.compare x y < 0
  | Loop_bool.Index_overflows i -> overflows st i
  | Loop_bool.Not p -> not (pred st p)
  | Loop_bool.Or (p, q) -> pred st p || pred st q
  | Loop_bool.Out_of_range (i, n) ->
      let i = index st i in
      Int64.compare i 0L < 0 || Int64.compare i (Int64.of_int n) >= 0
  | Loop_bool.Pool_better (best, value) ->
      let best = eval st best in
      let value = eval st value in
      Expr.Max_op.pool_better ~best ~value
  | Loop_bool.Value_eq (a, b) ->
      let x = eval st a in
      let y = eval st b in
      x = y
  | Loop_bool.Value_lt (a, b) ->
      let x = eval st a in
      let y = eval st b in
      x < y

let projection st ~local ~row ~lane =
  { Expr.Eval.Scan_projection.local; row = idx st row; lane = idx st lane }

(* A [Fail_if] that fired, as the row [Kernel_eval] would report. The site
   carries the expressions; only here, once the check has fired, are they
   evaluated. *)
let raise_failure st : Loop_failure.t -> 'a = function
  | Loop_failure.Gather_out_of_range { raw; extent } ->
      Err.Escape.throw st.esc
        (`Gather_index_out_of_range
           { Expr.Eval.Gather_index_out_of_range.raw = eval st raw; extent }
          : error)
  | Loop_failure.I64_division_by_zero ->
      Err.Escape.throw st.esc (`I64_division_by_zero : error)
  | Loop_failure.I64_division_overflow ->
      Err.Escape.throw st.esc (`I64_division_overflow : error)
  | Loop_failure.I64_from_float { value } ->
      let (_ : int64) =
        Err.Escape.or_throw st.esc
          (Err.map_error
             (fun (e : Expr.Value.i64_from_float_error) -> (e :> error))
             (Expr.Value.i64_of_float (eval st value)))
      in
      invalid_arg "Loop_interp: i64_from_float fired on a valid value"
  | Loop_failure.Index_overflow { index = i } ->
      ignore
        (index_with
           ~overflow:(fun op lhs rhs ->
             Err.Escape.throw st.esc
               (`Index_overflow
                  {
                    Expr.Index_overflow.op;
                    lhs = Int64.to_int lhs;
                    rhs = Int64.to_int rhs;
                  }
                 : error))
           st i);
      invalid_arg
        "Loop_interp: index_overflow fired on an index inside the domain"
  | Loop_failure.Load_out_of_range { buffer; coord = c } -> (
      let c = coord st c in
      match first_out_of_range buffer.Loop_buffer.sg c with
      | Some a ->
          Err.Escape.throw st.esc
            (`Coord_out_of_range
               ( Expr_bridge.source_of_id buffer.Loop_buffer.id,
                 a,
                 Expr.Coord.get c a,
                 c )
              : error)
      | None -> invalid_arg "Loop_interp: load_out_of_range fired in range")
  | Loop_failure.Scan_lane_out_of_range { local; row; lane; extent } ->
      Err.Escape.throw st.esc
        (`Scan_projection
           (Expr.Eval.Lane_out_of_range
              {
                Expr.Eval.Scan_bounds.projection =
                  projection st ~local ~row ~lane;
                extent;
              })
          : error)
  | Loop_failure.Scan_row_out_of_range { local; row; lane; extent } ->
      Err.Escape.throw st.esc
        (`Scan_projection
           (Expr.Eval.Row_out_of_range
              {
                Expr.Eval.Scan_bounds.projection =
                  projection st ~local ~row ~lane;
                extent;
              })
          : error)
  | Loop_failure.Local_out_of_range { local; _ } ->
      Err.Escape.throw st.esc (`Unbound_local local : error)

let store st (b : Loop_buffer.t) c (v : Loop_stored.t) =
  let (Tensor.Tensor t as packed) = buffer_tensor st b in
  match (v, t.Tensor.payload.Payload.fmt) with
  | Loop_stored.F32 e, Payload.F32 | Loop_stored.Bool e, Payload.Bool ->
      let x = eval st e in
      let c = checked_coord st b c in
      Tensor.set_float packed
        (Vec6.coord ~n:c.Expr.Coord.n ~t:c.Expr.Coord.t ~d:c.Expr.Coord.d
           ~h:c.Expr.Coord.h ~w:c.Expr.Coord.w ~c:c.Expr.Coord.c)
        x
  | Loop_stored.I64 e, Payload.I64 ->
      let x = eval st e in
      let c = checked_coord st b c in
      let i =
        (Vec6.offset t.Tensor.shape
           (Vec6.coord ~n:c.Expr.Coord.n ~t:c.Expr.Coord.t ~d:c.Expr.Coord.d
              ~h:c.Expr.Coord.h ~w:c.Expr.Coord.w ~c:c.Expr.Coord.c)
          :> int)
      in
      t.Tensor.payload.Payload.data.{i} <- x
  | _ ->
      invalid_arg "Loop_interp: stored value does not match the buffer format"

let rec exec st : Loop_stmt.t -> unit = function
  | Loop_stmt.Alloc (a, n) ->
      Hashtbl.replace st.arrays (Loop_array.to_int a) (Array.make (n :> int) 0.)
  | Loop_stmt.Array_set (a, i, e) ->
      let arr = lookup st.arrays (Loop_array.to_int a) "array" in
      let v = eval st e in
      let i = idx st i in
      if i < 0 || i >= Array.length arr then
        invalid_arg "Loop_interp: unchecked array write out of range"
      else arr.(i) <- v
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      Hashtbl.replace st.floats (Loop_temp.to_int t) (eval st e)
  | Loop_stmt.Assign (Loop_carrier.Int64, t, e) ->
      Hashtbl.replace st.int64s (Loop_temp.to_int t) (eval st e)
  | Loop_stmt.Assign_index (t, i) ->
      Hashtbl.replace st.indices (Loop_temp.to_int t) (idx st i)
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      Hashtbl.replace st.indices (Loop_temp.to_int t) (narrow (eval st e))
  | Loop_stmt.Charge_scan_update ->
      if Int64.compare st.meter.updates_remaining 0L <= 0 then
        Err.Escape.throw st.esc
          (`Scan_meter
             (Expr.Scan_meter.Updates_exhausted
                { limit = Expr.Scan_limits.max_updates st.limits })
            : error)
      else st.meter.updates_remaining <- Int64.sub st.meter.updates_remaining 1L
  | Loop_stmt.Release_scan_state width ->
      st.meter.live_state <- st.meter.live_state - (2 * width)
  | Loop_stmt.Reserve_scan_state width ->
      if
        st.meter.live_state + (2 * width) > Expr.Scan_limits.max_state st.limits
      then
        Err.Escape.throw st.esc
          (`Scan_meter
             (Expr.Scan_meter.State_over_limit
                { limit = Expr.Scan_limits.max_state st.limits })
            : error)
      else st.meter.live_state <- st.meter.live_state + (2 * width)
  | Loop_stmt.Reset_meter -> st.meter <- fresh_meter st.limits
  | Loop_stmt.Fail_if (p, f) -> if pred st p then raise_failure st f
  | Loop_stmt.For { var; lo; hi; body } ->
      let lo = idx st lo in
      let hi = idx st hi in
      let key = Loop_var.to_int var in
      for v = lo to hi - 1 do
        Hashtbl.replace st.vars key v;
        List.iter (exec st) body
      done
  | Loop_stmt.If (p, yes, no) ->
      List.iter (exec st) (if pred st p then yes else no)
  | Loop_stmt.Mark m -> (
      let c = st.counters in
      match m with
      | Loop_mark.Emitter -> c.emitters <- c.emitters + 1
      | Loop_mark.Key -> c.keys <- c.keys + 1
      | Loop_mark.Local -> c.locals <- c.locals + 1
      | Loop_mark.Reduction -> c.reductions <- c.reductions + 1
      | Loop_mark.Scan -> c.scans <- c.scans + 1
      | Loop_mark.Scan_update -> c.scan_updates <- c.scan_updates + 1)
  | Loop_stmt.Store { buffer; coord; value } -> store st buffer coord value

(* An Output buffer starts zeroed. Only the two float-path formats and the
   int64 track can be produced ([Kernel.create]'s [Format_rule]), so any other
   signature is a malformed program rather than a typed failure. *)
let allocate (b : Loop_buffer.t) =
  let shape = b.Loop_buffer.sg.Tensor_sig.shape in
  match b.Loop_buffer.sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.F32 -> Tensor.create shape
  | Payload.Fmt Payload.Bool -> Tensor.materialize_bool shape (fun _ -> false)
  | Payload.Fmt Payload.I64 -> Tensor.materialize_i64 shape (fun _ -> 0L)
  | Payload.Fmt _ -> invalid_arg "Loop_interp: output buffer format"

let bind_buffers esc (p : Loop_program.t) ~bind =
  List.fold_left
    (fun acc (b : Loop_buffer.t) ->
      match b.Loop_buffer.role with
      | Loop_buffer.Input -> (
          match bind b.Loop_buffer.id with
          | None -> Err.Escape.throw esc (`Unbound_input b.Loop_buffer.id)
          | Some tensor ->
              Err.Escape.or_throw esc
                (Err.map_error
                   (fun (`Binding_mismatch m) -> `Binding_mismatch m)
                   (Kernel_eval.check_binding b.Loop_buffer.id b.Loop_buffer.sg
                      tensor));
              Tensor_id.Map.add b.Loop_buffer.id tensor acc)
      | Loop_buffer.Output | Loop_buffer.Scratch ->
          Tensor_id.Map.add b.Loop_buffer.id (allocate b) acc)
    Tensor_id.Map.empty p.Loop_program.buffers

let run ?(counters = counters ()) (p : Loop_program.t) ~bind =
  Err.Escape.with_escape @@ fun esc ->
  let st =
    {
      esc;
      limits = p.Loop_program.scan_limits;
      meter = fresh_meter p.Loop_program.scan_limits;
      counters;
      buffers = bind_buffers esc p ~bind;
      vars = Hashtbl.create 16;
      floats = Hashtbl.create 16;
      int64s = Hashtbl.create 16;
      indices = Hashtbl.create 16;
      arrays = Hashtbl.create 4;
    }
  in
  List.iter (exec st) p.Loop_program.body;
  List.fold_left
    (fun acc (b : Loop_buffer.t) ->
      match b.Loop_buffer.role with
      | Loop_buffer.Output ->
          Tensor_id.Map.add b.Loop_buffer.id
            (Tensor_id.Map.find b.Loop_buffer.id st.buffers)
            acc
      | Loop_buffer.Input | Loop_buffer.Scratch -> acc)
    Tensor_id.Map.empty p.Loop_program.buffers

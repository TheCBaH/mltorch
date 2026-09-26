module B = Js_build
module Failure = Loop_js_failure

let function_name = "loop_kernel"

(* Names by first appearance, per emission, as [Loop_pp] does. Every multi-operand
   node names its operands left to right with explicit [let]s: an OCaml call
   evaluates its arguments in an unspecified order, and a name must not depend on
   which one the compiler picked. *)
type names = {
  vars : (int, int) Hashtbl.t;
  temps : (int, int) Hashtbl.t;
  arrays : (int, int) Hashtbl.t;
  buffers : (int, int) Hashtbl.t;
  sites : Loop_failure.t array;
  mutable next_site : int;
}

let ordinal table key =
  match Hashtbl.find_opt table key with
  | Some n -> n
  | None ->
      let n = Hashtbl.length table in
      Hashtbl.add table key n;
      n

let name prefix n = Js_ident.v (prefix ^ string_of_int n)
let var nm v = name "i" (ordinal nm.vars (Loop_var.to_int v))
let temp nm t = name "x" (ordinal nm.temps (Loop_temp.to_int t))
let array nm a = name "a" (ordinal nm.arrays (Loop_array.to_int a))

let buffer_prefix nm (b : Loop_buffer.t) =
  "b" ^ string_of_int (ordinal nm.buffers (Tensor_id.to_int b.Loop_buffer.id))

let buffer nm b = Js_ident.v (buffer_prefix nm b)

(* [Ident.v] is the one place a name is checked, so a per-channel table's name is
   built from the buffer's and checked like any other. *)
let quant_scales nm b = Js_ident.v (buffer_prefix nm b ^ "_scale")
let quant_zeros nm b = Js_ident.v (buffer_prefix nm b ^ "_zero")

let rec index nm : Loop_index.t -> B.idx B.t = function
  | Loop_index.Add (a, b) ->
      let a = index nm a in
      let b = index nm b in
      B.Idx.add a b
  | Loop_index.Ceil_div_pos (a, d) -> B.Idx.ceil_div_pos (index nm a) d
  | Loop_index.Clamp_low a -> B.Idx.clamp_low (index nm a)
  | Loop_index.Const n -> B.Idx.const n
  | Loop_index.Floor_div_pos (a, d) -> B.Idx.floor_div_pos (index nm a) d
  | Loop_index.Max (a, b) ->
      let a = index nm a in
      let b = index nm b in
      B.Idx.max a b
  | Loop_index.Min (a, b) ->
      let a = index nm a in
      let b = index nm b in
      B.Idx.min a b
  | Loop_index.Scale (k, a) -> B.Idx.scale k (index nm a)
  | Loop_index.Temp t -> B.Idx.var (temp nm t)
  | Loop_index.Var v -> B.Idx.var (var nm v)

(* The dense row-major offset of a coordinate in a buffer's shape, the same
   [Vec6.offset] linearisation the tensors use. *)
let offset nm (b : Loop_buffer.t) (c : Loop_index.coord) =
  let shape = b.Loop_buffer.sg.Tensor_sig.shape in
  List.fold_left
    (fun acc a ->
      let extent = Dim.to_int (Vec6.get shape a) in
      let i = index nm (Expr.Coord.get c a) in
      match acc with
      | None -> Some i
      | Some acc -> Some (B.Idx.add (B.Idx.scale extent acc) i))
    None Expr.Axis.all
  |> Option.get

let fmt_of (b : Loop_buffer.t) =
  let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
  Payload.fmt_name f

let typed_array b =
  match fmt_of b with
  | "bf16" | "f16" -> "Uint16Array"
  | "bool" -> "Uint8Array"
  | "f32" -> "Float32Array"
  | "f64" -> "Float64Array"
  | "i16" -> "Int16Array"
  | "i32" -> "Int32Array"
  | "i64" -> "BigInt64Array"
  | "i8" -> "Int8Array"
  | f -> invalid_arg ("Loop_js.typed_array: no typed array for format " ^ f)

let unary_js : Expr.Value.unary_op -> B.num B.t -> B.num B.t = function
  | Expr.Value.Cos -> B.Num.cos
  | Expr.Value.Erf -> Loop_js_runtime.erf
  | Expr.Value.Exp -> B.Num.exp
  | Expr.Value.Log -> B.Num.log
  | Expr.Value.Sin -> B.Num.sin
  | Expr.Value.Sqrt -> B.Num.sqrt
  | Expr.Value.Trunc -> B.Num.trunc

let binary_js : Expr.Value.binary_op -> B.num B.t -> B.num B.t -> B.num B.t =
  function
  | Expr.Value.Add -> B.Num.add
  | Expr.Value.Div -> B.Num.div
  | Expr.Value.Mul -> B.Num.mul
  | Expr.Value.Sub -> B.Num.sub

(* One node per checked operation: every [Add] and [Scale] node's value must stay
   in the domain [Loop_range.domain]. Post-order, so the first node to leave it is
   the one the interpreter reports, with its operands: an [Add]'s two, a
   [Scale]'s factor and operand. Each node's own JavaScript value is exact in a
   [Number] until it leaves the domain, and the first node to leave is one of
   these, so a later cancellation cannot hide it. *)
type overflow_node = {
  op : Failure.Overflow_op.t;
  value : B.idx B.t;
  lhs : B.idx B.t;
  rhs : B.idx B.t;
}

let overflow_nodes nm i =
  let rec go acc (i : Loop_index.t) =
    match i with
    | Loop_index.Add (a, b) ->
        let acc = go (go acc a) b in
        let lhs = index nm a in
        let rhs = index nm b in
        { op = Failure.Overflow_op.Add; value = index nm i; lhs; rhs } :: acc
    | Loop_index.Scale (k, a) ->
        let acc = go acc a in
        let rhs = index nm a in
        {
          op = Failure.Overflow_op.Mul;
          value = index nm i;
          lhs = B.Idx.const k;
          rhs;
        }
        :: acc
    | Loop_index.Ceil_div_pos (a, _)
    | Loop_index.Clamp_low a
    | Loop_index.Floor_div_pos (a, _) ->
        go acc a
    | Loop_index.Max (a, b) | Loop_index.Min (a, b) -> go (go acc a) b
    | Loop_index.Const _ | Loop_index.Temp _ | Loop_index.Var _ -> acc
  in
  List.rev (go [] i)

(* The decode a load applies, by the buffer's declared format: the JavaScript
   counterpart of [Payload.get_float]. A quantized cell is [scale * (q - zero)],
   the integer difference exact and small, then one binary64 multiply, in the
   order [Quant.dequantize] does it. A per-channel buffer names its two constant
   arrays ([quant_scales]) and indexes them by the C component of the load. *)
let quant_of (b : Loop_buffer.t) =
  match b.Loop_buffer.sg.Tensor_sig.quant with
  | Some q -> q
  | None -> invalid_arg "Loop_js: a quantized buffer without parameters"

let bool_cell cell =
  B.select (B.Num.ne cell (B.Num.const 0.)) (B.Num.const 1.) (B.Num.const 0.)

let load_cell nm (b : Loop_buffer.t) c =
  let cells () = B.load (B.Arr.num (buffer nm b)) (offset nm b c) in
  match fmt_of b with
  | "bf16" ->
      Loop_js_runtime.bf16_to_float
        (B.load (B.Arr.bits (buffer nm b)) (offset nm b c))
  | "bool" -> bool_cell (cells ())
  | "f16" ->
      Loop_js_runtime.f16_to_float
        (B.load (B.Arr.bits (buffer nm b)) (offset nm b c))
  | "f32" | "f64" | "i32" -> cells ()
  | "i64" -> B.Num.of_big (B.load (B.Arr.big (buffer nm b)) (offset nm b c))
  | "i16" | "i8" -> (
      let q = quant_of b in
      match Quant.channel_count q with
      | None ->
          let scale, zero = Quant.params q ~c:(Dim.index 0) in
          B.Num.mul (B.Num.const scale)
            (B.Num.sub (cells ()) (B.Num.const (float_of_int zero)))
      | Some _ ->
          let ch = index nm (Expr.Coord.get c Expr.Axis.C) in
          let cell = cells () in
          B.Num.mul
            (B.load (B.Arr.num (quant_scales nm b)) ch)
            (B.Num.sub cell (B.load (B.Arr.num (quant_zeros nm b)) ch)))
  | f -> invalid_arg ("Loop_js: no decode for format " ^ f)

let rec num nm : float Loop_expr.t -> B.num B.t = function
  | Loop_expr.Array_get (a, i) -> B.load (B.Arr.num (array nm a)) (index nm i)
  | Loop_expr.Binary (op, a, b) ->
      let a = num nm a in
      let b = num nm b in
      binary_js op a b
  | Loop_expr.Const x -> B.Num.const x
  | Loop_expr.Float_max (a, b) ->
      let a = num nm a in
      let b = num nm b in
      Loop_js_runtime.float_max a b
  | Loop_expr.I64_to_float a -> B.Num.of_big (big nm a)
  | Loop_expr.Load (b, c) -> load_cell nm b c
  | Loop_expr.Round_f32 a -> B.Num.fround (num nm a)
  | Loop_expr.Select (p, a, b) ->
      let p = pred nm p in
      let a = num nm a in
      let b = num nm b in
      B.select p a b
  | Loop_expr.Temp (Loop_carrier.Float, t) -> B.Num.var (temp nm t)
  | Loop_expr.Unary (op, a) -> unary_js op (num nm a)
  | Loop_expr.Value_of_index i -> B.Num.of_idx (index nm i)

and big nm : int64 Loop_expr.t -> B.big B.t = function
  | Loop_expr.Float_to_i64 a -> B.Big.of_num_trunc (num nm a)
  | Loop_expr.I64_binary (op, a, b) -> (
      let a = big nm a in
      let b = big nm b in
      match op with
      | Expr.Value.I64_add -> B.Big.add_wrap a b
      | Expr.Value.I64_div -> B.Big.div_unchecked a b
      | Expr.Value.I64_mul -> B.Big.mul_wrap a b
      | Expr.Value.I64_sub -> B.Big.sub_wrap a b)
  | Loop_expr.I64_const n -> B.Big.const n
  | Loop_expr.I64_of_index i -> B.Big.of_idx (index nm i)
  | Loop_expr.Load_i64 (b, c) ->
      B.load (B.Arr.big (buffer nm b)) (offset nm b c)
  | Loop_expr.Select (p, a, b) ->
      let p = pred nm p in
      let a = big nm a in
      let b = big nm b in
      B.select p a b
  | Loop_expr.Temp (Loop_carrier.Int64, t) -> B.Big.var (temp nm t)

and pred nm : Loop_expr.pred -> B.bool_ B.t = function
  | Loop_bool.I64_eq (a, b) ->
      let a = big nm a in
      let b = big nm b in
      B.Big.eq a b
  | Loop_bool.I64_lt (a, b) ->
      let a = big nm a in
      let b = big nm b in
      B.Big.lt a b
  | Loop_bool.Index_eq (a, b) ->
      let a = index nm a in
      let b = index nm b in
      B.Idx.eq a b
  | Loop_bool.Index_lt (a, b) ->
      let a = index nm a in
      let b = index nm b in
      B.Idx.lt a b
  | Loop_bool.Index_overflows i -> (
      match overflow_nodes nm i with
      | [] -> B.Pred.false_
      | first :: rest ->
          List.fold_left
            (fun acc n -> B.Pred.or_ acc (B.Idx.outside_int32 n.value))
            (B.Idx.outside_int32 first.value)
            rest)
  | Loop_bool.Not p -> B.Pred.not_ (pred nm p)
  | Loop_bool.Or (p, q) ->
      let p = pred nm p in
      let q = pred nm q in
      B.Pred.or_ p q
  | Loop_bool.Out_of_range (i, n) -> B.Idx.out_of_range (index nm i) n
  | Loop_bool.Pool_better (best, value) ->
      let best = num nm best in
      let value = num nm value in
      Loop_js_runtime.pool_better best value
  | Loop_bool.Value_eq (a, b) ->
      let a = num nm a in
      let b = num nm b in
      B.Num.eq a b
  | Loop_bool.Value_lt (a, b) ->
      let a = num nm a in
      let b = num nm b in
      B.Num.lt a b

(* A scan-meter limit is an int64 the meter counts down in a [Number]: exact
   only up to 2^53, and a limit beyond it would silently round. *)
let exact_number n =
  let x = Int64.to_float n in
  if Float.abs x <= 9007199254740992. && Int64.equal (Int64.of_float x) n then
    B.Num.const x
  else invalid_arg "Loop_js: a scan limit is not exact in a Number"

(* A scan projection's failure record. [cached] says whether a stored trace
   local or an inline scan was read; which trace is decoded from [site]. *)
let scan_failure nm which ~site ~local ~row ~lane ~extent =
  let row = index nm row in
  let lane = index nm lane in
  Failure.record Failure.Kind.Scan_projection
    [
      (Failure.Field.Which, B.string (Failure.Projection.to_string which));
      (Failure.Field.Cached, B.bool (Option.is_some local));
      (Failure.Field.Row, B.expr row);
      (Failure.Field.Lane, B.expr lane);
      (Failure.Field.Extent, B.expr (B.Idx.const extent));
      (Failure.Field.Site, B.expr (B.Idx.const site));
    ]

(* A failure is a returned record, never a host exception. The fields are the
   ones the interpreter's typed row carries. *)
let failure nm ~site : Loop_failure.t -> Js_ast.expr = function
  | Loop_failure.Load_out_of_range { buffer = b; coord = c } ->
      let shape = b.Loop_buffer.sg.Tensor_sig.shape in
      let extents =
        List.map
          (fun a -> B.Idx.const (Dim.to_int (Vec6.get shape a)))
          Expr.Axis.all
      in
      let coord =
        List.fold_left
          (fun acc a -> index nm (Expr.Coord.get c a) :: acc)
          [] Expr.Axis.all
        |> List.rev
      in
      Loop_js_runtime.coord_failure
        ~buffer:(B.Idx.const (Tensor_id.to_int b.Loop_buffer.id))
        ~extents ~coord
  | Loop_failure.Gather_out_of_range { raw; extent } ->
      Failure.record Failure.Kind.Gather_index_out_of_range
        [
          (Failure.Field.Raw, B.to_string (big nm raw));
          (Failure.Field.Extent, B.expr (B.Idx.const extent));
        ]
  | Loop_failure.I64_division_by_zero ->
      Failure.record Failure.Kind.I64_division_by_zero []
  | Loop_failure.I64_division_overflow ->
      Failure.record Failure.Kind.I64_division_overflow []
  | Loop_failure.I64_from_float { value } ->
      Loop_js_runtime.i64_from_float_failure (num nm value)
  | Loop_failure.Index_overflow _ ->
      invalid_arg "Loop_js.failure: an index overflow is written per node"
  | Loop_failure.Local_out_of_range _ ->
      Failure.record Failure.Kind.Unbound_local
        [ (Failure.Field.Site, B.expr (B.Idx.const site)) ]
  | Loop_failure.Scan_lane_out_of_range { local; row; lane; extent } ->
      scan_failure nm Failure.Projection.Lane ~site ~local ~row ~lane ~extent
  | Loop_failure.Scan_row_out_of_range { local; row; lane; extent } ->
      scan_failure nm Failure.Projection.Row ~site ~local ~row ~lane ~extent

(* The [Fail_if] sites are numbered in the walk [Loop_js_failure.sites] makes,
   and each is checked against that array by physical equality. *)
let next_site nm f =
  let k = nm.next_site in
  if k >= Array.length nm.sites || nm.sites.(k) != f then
    invalid_arg "Loop_js: a failure site drifted from Loop_js_failure.sites";
  nm.next_site <- k + 1;
  k

let meter_failure which limit =
  B.Stmt.return_
    (Failure.record Failure.Kind.Scan_meter
       [
         (Failure.Field.Which, B.string (Failure.Meter.to_string which));
         (Failure.Field.Limit, limit);
       ])

let scan_remaining = Js_ident.v "scan_remaining"
let scan_live = Js_ident.v "scan_live"

let rec stmt nm ~limits (s : Loop_stmt.t) : Js_ast.stmt list =
  match s with
  | Loop_stmt.Alloc (a, n) ->
      [ B.Stmt.const_arr (array nm a) (B.Arr.new_float64 (n :> int)) ]
  | Loop_stmt.Array_set (a, i, e) ->
      let i = index nm i in
      let e = num nm e in
      [ B.store (B.Arr.num (array nm a)) i e ]
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      [ B.Stmt.assign_num (temp nm t) (num nm e) ]
  | Loop_stmt.Assign (Loop_carrier.Int64, t, e) ->
      [ B.Stmt.assign_big (temp nm t) (big nm e) ]
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      [ B.Stmt.assign_idx (temp nm t) (B.Idx.of_big_bounded (big nm e)) ]
  | Loop_stmt.Assign_index (t, i) ->
      [ B.Stmt.assign_idx (temp nm t) (index nm i) ]
  | Loop_stmt.Fail_if (p, f) -> (
      let site = next_site nm f in
      match (p, f) with
      | Loop_bool.Index_overflows i, Loop_failure.Index_overflow { index = j }
        when i = j ->
          (* One [if] per node, post-order: the first to fire is the node the
             interpreter reports, with its operands. *)
          List.map
            (fun n ->
              B.Stmt.if_
                (B.Idx.outside_int32 n.value)
                [
                  B.Stmt.return_
                    (Failure.record Failure.Kind.Index_overflow
                       [
                         ( Failure.Field.Op,
                           B.string (Failure.Overflow_op.to_string n.op) );
                         (Failure.Field.Lhs, B.expr n.lhs);
                         (Failure.Field.Rhs, B.expr n.rhs);
                       ]);
                ]
                [])
            (overflow_nodes nm i)
      | _, Loop_failure.Index_overflow _ ->
          invalid_arg
            "Loop_js: an index overflow failure under a foreign predicate"
      | _ ->
          let p = pred nm p in
          [ B.Stmt.if_ p [ B.Stmt.return_ (failure nm ~site f) ] [] ])
  | Loop_stmt.For { var = v; lo; hi; body } ->
      let name = var nm v in
      let lo = index nm lo in
      let hi = index nm hi in
      [ B.Stmt.for_ name ~lo ~hi (block nm ~limits body) ]
  | Loop_stmt.If (p, yes, no) ->
      let p = pred nm p in
      let yes = block nm ~limits yes in
      let no = block nm ~limits no in
      [ B.Stmt.if_ p yes no ]
  | Loop_stmt.Charge_scan_update ->
      [
        B.Stmt.if_
          (B.Num.le (B.Num.var scan_remaining) (B.Num.const 0.))
          [
            meter_failure Failure.Meter.Updates_exhausted
              (B.expr (exact_number (Expr.Scan_limits.max_updates limits)));
          ]
          [];
        B.Stmt.decr_num scan_remaining (B.Num.const 1.);
      ]
  | Loop_stmt.Mark _ -> []
  | Loop_stmt.Release_scan_state width ->
      [ B.Stmt.decr_num scan_live (B.Num.const (float_of_int (2 * width))) ]
  | Loop_stmt.Reserve_scan_state width ->
      let live = float_of_int (2 * width) in
      let max_state = Expr.Scan_limits.max_state limits in
      [
        B.Stmt.if_
          (B.Num.gt
             (B.Num.add (B.Num.var scan_live) (B.Num.const live))
             (B.Num.const (float_of_int max_state)))
          [
            meter_failure Failure.Meter.State_over_limit
              (B.expr (B.Num.const (float_of_int max_state)));
          ]
          [];
        B.Stmt.incr_num scan_live (B.Num.const live);
      ]
  | Loop_stmt.Reset_meter ->
      [
        B.Stmt.assign_num scan_remaining
          (exact_number (Expr.Scan_limits.max_updates limits));
        B.Stmt.assign_num scan_live (B.Num.const 0.);
      ]
  | Loop_stmt.Store { buffer = b; coord = c; value } -> (
      let off = offset nm b c in
      match value with
      | Loop_stored.Bool e ->
          [ B.store (B.Arr.num (buffer nm b)) off (bool_cell (num nm e)) ]
      | Loop_stored.F32 e ->
          [ B.store (B.Arr.num (buffer nm b)) off (num nm e) ]
      | Loop_stored.I64 e ->
          [ B.store (B.Arr.big (buffer nm b)) off (big nm e) ])

and block nm ~limits body = List.concat_map (stmt nm ~limits) body

(* Temporaries are declared once at function scope: an accumulator is assigned
   across iterations, so a per-iteration [let] would drop it. *)
(* What function scope must declare: the temporaries, in first-assigned order,
   and whether the program touches the scan meter. *)
let declarations (p : Loop_program.t) =
  let floats = ref [] and int64s = ref [] and indices = ref [] in
  let meter = ref false in
  let seen = ref Loop_temp.Set.empty in
  let add r t =
    if not (Loop_temp.Set.mem t !seen) then (
      seen := Loop_temp.Set.add t !seen;
      r := t :: !r)
  in
  let rec go (s : Loop_stmt.t) =
    match s with
    | Loop_stmt.Assign (Loop_carrier.Float, t, _) -> add floats t
    | Loop_stmt.Assign (Loop_carrier.Int64, t, _) -> add int64s t
    | Loop_stmt.Assign_index (t, _) | Loop_stmt.Assign_index_of_i64 (t, _) ->
        add indices t
    | Loop_stmt.For { body; _ } -> List.iter go body
    | Loop_stmt.If (_, yes, no) ->
        List.iter go yes;
        List.iter go no
    | Loop_stmt.Charge_scan_update | Loop_stmt.Release_scan_state _
    | Loop_stmt.Reserve_scan_state _ | Loop_stmt.Reset_meter ->
        meter := true
    | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ | Loop_stmt.Fail_if _
    | Loop_stmt.Mark _ | Loop_stmt.Store _ ->
        ()
  in
  List.iter go p.Loop_program.body;
  (List.rev !floats, List.rev !int64s, List.rev !indices, !meter)

(* The helpers the entry function needs: start from its free names, add every
   helper that defines one of them, and repeat with the added helpers' own free
   names. A helper is in the prelude because a name is used, never because some
   text happened to match. Emitted in [Loop_js_runtime.Name] order. *)
let prelude entry =
  let needed_by (h : Loop_js_runtime.Helper.t) needed =
    List.exists
      (fun d -> Js_ident.Set.mem d needed)
      h.Loop_js_runtime.Helper.defines
  in
  let rec fix chosen needed =
    let added =
      List.filter
        (fun h -> (not (List.memq h chosen)) && needed_by h needed)
        Loop_js_runtime.helpers
    in
    if added = [] then chosen
    else
      let needed =
        List.fold_left
          (fun acc (h : Loop_js_runtime.Helper.t) ->
            Js_ident.Set.union acc (Js_check.free h.Loop_js_runtime.Helper.body))
          needed added
      in
      fix (chosen @ added) needed
  in
  let chosen = fix [] (Js_check.free [ Js_ast.Stmt.Function entry ]) in
  List.concat_map
    (fun h -> if List.memq h chosen then h.Loop_js_runtime.Helper.body else [])
    Loop_js_runtime.helpers

let to_ast (p : Loop_program.t) =
  let nm =
    {
      vars = Hashtbl.create 8;
      temps = Hashtbl.create 8;
      arrays = Hashtbl.create 8;
      buffers = Hashtbl.create 8;
      sites = Loop_js_failure.sites p;
      next_site = 0;
    }
  in
  (* Buffers take their positions in program order, before any use. *)
  let params =
    List.map (fun (b : Loop_buffer.t) -> buffer nm b) p.Loop_program.buffers
  in
  let limits = p.Loop_program.scan_limits in
  let body = block nm ~limits p.Loop_program.body in
  if nm.next_site <> Array.length nm.sites then
    invalid_arg "Loop_js: a failure site was not written";
  let floats, int64s, indices, meter = declarations p in
  (* Index temporaries print as [0] like the float ones; the kind only says how
     the temporary may be assigned. *)
  let temps =
    List.map (fun t -> B.Stmt.let_num (temp nm t) (B.Num.const 0.)) floats
    @ List.map (fun t -> B.Stmt.let_idx (temp nm t) (B.Idx.const 0)) indices
    @ List.map (fun t -> B.Stmt.let_big (temp nm t) (B.Big.const 0L)) int64s
  in
  (* A per-channel quantized buffer's parameters, once, as constant arrays. *)
  let tables =
    List.concat_map
      (fun (b : Loop_buffer.t) ->
        match fmt_of b with
        | "i16" | "i8" -> (
            let q = quant_of b in
            match Quant.channel_count q with
            | None -> []
            | Some n ->
                let params =
                  List.init n (fun c -> Quant.params q ~c:(Dim.index c))
                in
                [
                  B.Stmt.const_arr (quant_scales nm b)
                    (B.Arr.literal
                       (List.map (fun (s, _) -> B.Num.const s) params));
                  B.Stmt.const_arr (quant_zeros nm b)
                    (B.Arr.literal
                       (List.map
                          (fun (_, z) -> B.Num.const (float_of_int z))
                          params));
                ])
        | _ -> [])
      p.Loop_program.buffers
  in
  let meter =
    if meter then
      [
        B.Stmt.let_num scan_remaining
          (exact_number (Expr.Scan_limits.max_updates limits));
        B.Stmt.let_num scan_live (B.Num.const 0.);
      ]
    else []
  in
  let entry =
    {
      Js_ast.Func.name = Js_ident.v function_name;
      params;
      body = temps @ tables @ meter @ body @ [ B.Stmt.return_null ];
    }
  in
  let program = { Js_ast.Program.prelude = prelude entry; entry } in
  (match Js_check.closed program with
  | Ok () -> ()
  | Error faults ->
      invalid_arg
        (Fmt.str "Loop_js.to_ast: the program is not closed: %a"
           Fmt.(list ~sep:(any "; ") Js_check.Fault.pp)
           faults));
  program

let emit p = Js_print.script (to_ast p)

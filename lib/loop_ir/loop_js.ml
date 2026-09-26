let function_name = "loop_kernel"

(* Names by first appearance, per emission, as [Loop_pp] does. *)
type names = {
  vars : (int, int) Hashtbl.t;
  temps : (int, int) Hashtbl.t;
  arrays : (int, int) Hashtbl.t;
  buffers : (int, int) Hashtbl.t;
}

let ordinal table key =
  match Hashtbl.find_opt table key with
  | Some n -> n
  | None ->
      let n = Hashtbl.length table in
      Hashtbl.add table key n;
      n

let var nm v = Fmt.str "i%d" (ordinal nm.vars (Loop_var.to_int v))
let temp nm t = Fmt.str "x%d" (ordinal nm.temps (Loop_temp.to_int t))
let array nm a = Fmt.str "a%d" (ordinal nm.arrays (Loop_array.to_int a))

let buffer nm (b : Loop_buffer.t) =
  Fmt.str "b%d" (ordinal nm.buffers (Tensor_id.to_int b.Loop_buffer.id))

(* A binary64 literal that round-trips: [%.17g], with the specials spelled the
   way JavaScript reads them. *)
let float_literal x =
  if Float.is_nan x then "NaN"
  else if x = Float.infinity then "Infinity"
  else if x = Float.neg_infinity then "(-Infinity)"
  else if x = 0. && 1. /. x < 0. then "(-0)"
  else if x < 0. then Fmt.str "(%.17g)" x
  else Fmt.str "%.17g" x

let rec index nm : Loop_index.t -> string = function
  | Loop_index.Add (a, b) -> Fmt.str "(%s + %s)" (index nm a) (index nm b)
  | Loop_index.Ceil_div_pos (a, d) ->
      Fmt.str "Math.ceil(%s / %d)" (index nm a) d
  | Loop_index.Clamp_low a -> Fmt.str "Math.max(0, %s)" (index nm a)
  | Loop_index.Const n -> if n < 0 then Fmt.str "(%d)" n else string_of_int n
  | Loop_index.Floor_div_pos (a, d) ->
      Fmt.str "Math.floor(%s / %d)" (index nm a) d
  | Loop_index.Max (a, b) ->
      Fmt.str "Math.max(%s, %s)" (index nm a) (index nm b)
  | Loop_index.Min (a, b) ->
      Fmt.str "Math.min(%s, %s)" (index nm a) (index nm b)
  | Loop_index.Scale (k, a) -> Fmt.str "(%d * %s)" k (index nm a)
  | Loop_index.Temp t -> temp nm t
  | Loop_index.Var v -> var nm v

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
      | Some acc -> Some (Fmt.str "(%s * %d + %s)" acc extent i))
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

let unary_js : Expr.Value.unary_op -> string = function
  | Expr.Value.Cos -> "Math.cos"
  | Expr.Value.Erf -> "erf"
  | Expr.Value.Exp -> "Math.exp"
  | Expr.Value.Log -> "Math.log"
  | Expr.Value.Sin -> "Math.sin"
  | Expr.Value.Sqrt -> "Math.sqrt"
  | Expr.Value.Trunc -> "Math.trunc"

(* One clause per checked operation: every [Add] and [Scale] node's value must
   stay in the domain [Loop_range.domain]. Post-order and boolean, so the order
   the clauses join in is irrelevant; each node's own JavaScript value is exact
   in a [Number] until it leaves the domain, and the first node to leave is one of
   these clauses, so a later cancellation cannot hide it. *)
let overflow_clauses nm i =
  let clause n = Fmt.str "(%s < -2147483648 || %s >= 2147483648)" n n in
  let rec go acc (i : Loop_index.t) =
    match i with
    | Loop_index.Add (a, b) -> clause (index nm i) :: go (go acc a) b
    | Loop_index.Scale (_, a) -> clause (index nm i) :: go acc a
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
let quant_scales nm (b : Loop_buffer.t) = buffer nm b ^ "_scale"
let quant_zeros nm (b : Loop_buffer.t) = buffer nm b ^ "_zero"

let quant_of (b : Loop_buffer.t) =
  match b.Loop_buffer.sg.Tensor_sig.quant with
  | Some q -> q
  | None -> invalid_arg "Loop_js: a quantized buffer without parameters"

let load_cell nm (b : Loop_buffer.t) c =
  let cell = Fmt.str "%s[%s]" (buffer nm b) (offset nm b c) in
  match fmt_of b with
  | "bf16" -> Fmt.str "bf16_to_float(%s)" cell
  | "bool" -> Fmt.str "(%s !== 0 ? 1 : 0)" cell
  | "f16" -> Fmt.str "f16_to_float(%s)" cell
  | "f32" | "f64" | "i32" -> cell
  | "i64" -> Fmt.str "Number(%s)" cell
  | "i16" | "i8" -> (
      let q = quant_of b in
      match Quant.channel_count q with
      | None ->
          let scale, zero = Quant.params q ~c:(Dim.index 0) in
          Fmt.str "(%s * (%s - %d))" (float_literal scale) cell zero
      | Some _ ->
          let ch = index nm (Expr.Coord.get c Expr.Axis.C) in
          Fmt.str "(%s[%s] * (%s - %s[%s]))" (quant_scales nm b) ch cell
            (quant_zeros nm b) ch)
  | f -> invalid_arg ("Loop_js: no decode for format " ^ f)

let i64_literal n =
  if Int64.compare n 0L < 0 then Fmt.str "(%Ldn)" n else Fmt.str "%Ldn" n

let rec expr : type a. names -> a Loop_expr.t -> string =
 fun nm -> function
  | Loop_expr.Array_get (a, i) -> Fmt.str "%s[%s]" (array nm a) (index nm i)
  | Loop_expr.Binary (op, a, b) ->
      Fmt.str "(%s %s %s)" (expr nm a) (Expr.Value.binary_sym op) (expr nm b)
  | Loop_expr.Const x -> float_literal x
  | Loop_expr.Float_max (a, b) ->
      Fmt.str "float_max(%s, %s)" (expr nm a) (expr nm b)
  | Loop_expr.Float_to_i64 a -> Fmt.str "BigInt(Math.trunc(%s))" (expr nm a)
  | Loop_expr.I64_binary (op, a, b) -> (
      let a = expr nm a and b = expr nm b in
      match op with
      | Expr.Value.I64_add -> Fmt.str "BigInt.asIntN(64, %s + %s)" a b
      | Expr.Value.I64_mul -> Fmt.str "BigInt.asIntN(64, %s * %s)" a b
      | Expr.Value.I64_sub -> Fmt.str "BigInt.asIntN(64, %s - %s)" a b
      | Expr.Value.I64_div -> Fmt.str "(%s / %s)" a b)
  | Loop_expr.I64_const n -> i64_literal n
  | Loop_expr.I64_of_index i -> Fmt.str "BigInt(%s)" (index nm i)
  | Loop_expr.I64_to_float a -> Fmt.str "Number(%s)" (expr nm a)
  | Loop_expr.Load_i64 (b, c) -> Fmt.str "%s[%s]" (buffer nm b) (offset nm b c)
  | Loop_expr.Load (b, c) -> load_cell nm b c
  | Loop_expr.Round_f32 a -> Fmt.str "Math.fround(%s)" (expr nm a)
  | Loop_expr.Select (p, a, b) ->
      Fmt.str "(%s ? %s : %s)" (pred nm p) (expr nm a) (expr nm b)
  | Loop_expr.Temp (Loop_carrier.Float, t) -> temp nm t
  | Loop_expr.Temp (Loop_carrier.Int64, t) -> temp nm t
  | Loop_expr.Unary (op, a) -> Fmt.str "%s(%s)" (unary_js op) (expr nm a)
  | Loop_expr.Value_of_index i -> index nm i

and pred nm : Loop_expr.pred -> string = function
  | Loop_bool.I64_eq (a, b) -> Fmt.str "(%s === %s)" (expr nm a) (expr nm b)
  | Loop_bool.I64_lt (a, b) -> Fmt.str "(%s < %s)" (expr nm a) (expr nm b)
  | Loop_bool.Index_eq (a, b) -> Fmt.str "(%s === %s)" (index nm a) (index nm b)
  | Loop_bool.Index_lt (a, b) -> Fmt.str "(%s < %s)" (index nm a) (index nm b)
  | Loop_bool.Index_overflows i -> (
      match overflow_clauses nm i with
      | [] -> "false"
      | clauses -> Fmt.str "(%s)" (String.concat " || " clauses))
  | Loop_bool.Not p -> Fmt.str "(!%s)" (pred nm p)
  | Loop_bool.Or (p, q) -> Fmt.str "(%s || %s)" (pred nm p) (pred nm q)
  | Loop_bool.Out_of_range (i, n) ->
      let i = index nm i in
      Fmt.str "(%s < 0 || %s >= %d)" i i n
  | Loop_bool.Pool_better (best, value) ->
      Fmt.str "pool_better(%s, %s)" (expr nm best) (expr nm value)
  | Loop_bool.Value_eq (a, b) -> Fmt.str "(%s === %s)" (expr nm a) (expr nm b)
  | Loop_bool.Value_lt (a, b) -> Fmt.str "(%s < %s)" (expr nm a) (expr nm b)

(* A scan projection's failure record. [cached] says whether a stored trace
   local or an inline scan was read; which trace is not carried, since a local's
   identity has no printed form outside the OCaml row. *)
let scan_failure nm which ~local ~row ~lane ~extent =
  Fmt.str
    "{ kind: \"scan_projection\", which: \"%s\", cached: %b, row: %s, lane: \
     %s, extent: %d }"
    which (Option.is_some local) (index nm row) (index nm lane) extent

(* A failure is a returned record, never a host exception. The fields are the
   ones the interpreter's typed row carries. *)
let failure nm : Loop_failure.t -> string = function
  | Loop_failure.Load_out_of_range { buffer = b; coord = c } ->
      let shape = b.Loop_buffer.sg.Tensor_sig.shape in
      Fmt.str "coord_failure(%d, [%s], [%s])"
        (Tensor_id.to_int b.Loop_buffer.id)
        (String.concat ", "
           (List.map
              (fun a -> string_of_int (Dim.to_int (Vec6.get shape a)))
              Expr.Axis.all))
        (String.concat ", "
           (List.map (fun a -> index nm (Expr.Coord.get c a)) Expr.Axis.all))
  | Loop_failure.Gather_out_of_range { raw; extent } ->
      Fmt.str
        "{ kind: \"gather_index_out_of_range\", raw: (%s).toString(), extent: \
         %d }"
        (expr nm raw) extent
  | Loop_failure.I64_division_by_zero -> "{ kind: \"i64_division_by_zero\" }"
  | Loop_failure.I64_division_overflow -> "{ kind: \"i64_division_overflow\" }"
  | Loop_failure.I64_from_float { value } ->
      Fmt.str "i64_from_float_failure(%s)" (expr nm value)
  | Loop_failure.Index_overflow _ -> "{ kind: \"index_overflow\" }"
  | Loop_failure.Local_out_of_range _ -> "{ kind: \"unbound_local\" }"
  | Loop_failure.Scan_lane_out_of_range { local; row; lane; extent } ->
      scan_failure nm "lane" ~local ~row ~lane ~extent
  | Loop_failure.Scan_row_out_of_range { local; row; lane; extent } ->
      scan_failure nm "row" ~local ~row ~lane ~extent

let stored nm : Loop_stored.t -> string = function
  | Loop_stored.Bool e -> Fmt.str "(%s !== 0 ? 1 : 0)" (expr nm e)
  | Loop_stored.F32 e -> expr nm e
  | Loop_stored.I64 e -> expr nm e

let rec stmt nm ~limits ~indent buf (s : Loop_stmt.t) =
  let line fmt =
    Fmt.kstr (fun l -> Buffer.add_string buf (indent ^ l ^ "\n")) fmt
  in
  let sub = indent ^ "  " in
  match s with
  | Loop_stmt.Alloc (a, n) ->
      line "const %s = new Float64Array(%d);" (array nm a) (n :> int)
  | Loop_stmt.Array_set (a, i, e) ->
      line "%s[%s] = %s;" (array nm a) (index nm i) (expr nm e)
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      line "%s = %s;" (temp nm t) (expr nm e)
  | Loop_stmt.Assign (Loop_carrier.Int64, t, e) ->
      line "%s = %s;" (temp nm t) (expr nm e)
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      line "%s = Number(%s);" (temp nm t) (expr nm e)
  | Loop_stmt.Assign_index (t, i) -> line "%s = %s;" (temp nm t) (index nm i)
  | Loop_stmt.Fail_if (p, f) ->
      line "if (%s) return %s;" (pred nm p) (failure nm f)
  | Loop_stmt.For { var = v; lo; hi; body } ->
      let name = var nm v in
      line "for (let %s = %s; %s < %s; %s++) {" name (index nm lo) name
        (index nm hi) name;
      List.iter (stmt nm ~limits ~indent:sub buf) body;
      line "}"
  | Loop_stmt.If (p, yes, no) ->
      line "if (%s) {" (pred nm p);
      List.iter (stmt nm ~limits ~indent:sub buf) yes;
      if no <> [] then (
        line "} else {";
        List.iter (stmt nm ~limits ~indent:sub buf) no);
      line "}"
  | Loop_stmt.Charge_scan_update ->
      line
        "if (scan_remaining <= 0) return { kind: \"scan_meter\", which: \
         \"updates_exhausted\", limit: %Ld };"
        (Expr.Scan_limits.max_updates limits);
      line "scan_remaining -= 1;"
  | Loop_stmt.Mark _ -> ()
  | Loop_stmt.Release_scan_state width -> line "scan_live -= %d;" (2 * width)
  | Loop_stmt.Reserve_scan_state width ->
      line
        "if (scan_live + %d > %d) return { kind: \"scan_meter\", which: \
         \"state_over_limit\", limit: %d };"
        (2 * width)
        (Expr.Scan_limits.max_state limits)
        (Expr.Scan_limits.max_state limits);
      line "scan_live += %d;" (2 * width)
  | Loop_stmt.Reset_meter ->
      line "scan_remaining = %Ld;" (Expr.Scan_limits.max_updates limits);
      line "scan_live = 0;"
  | Loop_stmt.Store { buffer = b; coord = c; value } ->
      line "%s[%s] = %s;" (buffer nm b) (offset nm b c) (stored nm value)

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

let contains haystack needle =
  let n = String.length needle and h = String.length haystack in
  let rec matches i j =
    j >= n || (haystack.[i + j] = needle.[j] && matches i (j + 1))
  in
  let rec at i = i + n <= h && (matches i 0 || at (i + 1)) in
  at 0

let emit (p : Loop_program.t) =
  let nm =
    {
      vars = Hashtbl.create 8;
      temps = Hashtbl.create 8;
      arrays = Hashtbl.create 8;
      buffers = Hashtbl.create 8;
    }
  in
  (* Buffers take their positions in program order, before any use. *)
  let params =
    List.map (fun (b : Loop_buffer.t) -> buffer nm b) p.Loop_program.buffers
  in
  let body = Buffer.create 1024 in
  List.iter
    (stmt nm ~limits:p.Loop_program.scan_limits ~indent:"  " body)
    p.Loop_program.body;
  let buf = Buffer.create 1024 in
  (* Only the helpers the body calls, in the runtime's own order. *)
  List.iter
    (fun (name, source) ->
      if contains (Buffer.contents body) (name ^ "(") then (
        Buffer.add_string buf source;
        Buffer.add_char buf '\n'))
    Loop_js_runtime.helpers;
  Buffer.add_string buf
    (Fmt.str "function %s(%s) {\n" function_name (String.concat ", " params));
  let floats, int64s, indices, meter = declarations p in
  List.iter
    (fun t -> Buffer.add_string buf (Fmt.str "  let %s = 0;\n" (temp nm t)))
    (floats @ indices);
  List.iter
    (fun t -> Buffer.add_string buf (Fmt.str "  let %s = 0n;\n" (temp nm t)))
    int64s;
  (* A per-channel quantized buffer's parameters, once, as constant arrays. *)
  List.iter
    (fun (b : Loop_buffer.t) ->
      match fmt_of b with
      | "i16" | "i8" -> (
          let q = quant_of b in
          match Quant.channel_count q with
          | None -> ()
          | Some n ->
              let params =
                List.init n (fun c -> Quant.params q ~c:(Dim.index c))
              in
              Buffer.add_string buf
                (Fmt.str "  const %s = [%s];\n  const %s = [%s];\n"
                   (quant_scales nm b)
                   (String.concat ", "
                      (List.map (fun (s, _) -> float_literal s) params))
                   (quant_zeros nm b)
                   (String.concat ", "
                      (List.map (fun (_, z) -> string_of_int z) params))))
      | _ -> ())
    p.Loop_program.buffers;
  if meter then
    Buffer.add_string buf
      (Fmt.str "  let scan_remaining = %Ld;\n  let scan_live = 0;\n"
         (Expr.Scan_limits.max_updates p.Loop_program.scan_limits));
  Buffer.add_buffer buf body;
  Buffer.add_string buf "  return null;\n}\n";
  Buffer.contents buf

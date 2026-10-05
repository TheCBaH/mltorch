module R = Loop_c_runtime
module F = Loop_js_failure

type error =
  [ `Unsupported_format of Tensor_id.t * string
  | `Unsupported_precision of Loop_numerics.Refusal.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Unsupported_format (id, f) ->
      Format.fprintf ppf "t%d: format %s has no C implementation"
        (Tensor_id.to_int id) f
  | `Unsupported_precision r ->
      Format.fprintf ppf "binary32 is not admitted: %a" Loop_numerics.Refusal.pp
        r

type t = {
  source : string;
  prelude : string;
  helpers : R.Name.t list;
  local_doubles : int64;
  precision : Loop_numerics.Precision.t;
  refusal : Loop_numerics.Refusal.t option;
  buffer_types : string list;
}

(* Names by first appearance, per emission, as [Loop_js] does. Every operand is
   an expression string built left to right, so a name never depends on the
   order OCaml happens to evaluate arguments in. *)
type names = {
  vars : (int, int) Hashtbl.t;
  temps : (int, int) Hashtbl.t;
  index_temps : (int, int) Hashtbl.t;
  arrays : (int, int) Hashtbl.t;
  buffers : (int, int) Hashtbl.t;
  f32 : bool;  (** the kernel's working precision is binary32 *)
  sites : Loop_failure.t array;
  mutable next_site : int;
  mutable used : R.Name.t list;
  mutable local_doubles : int64;
}

let ordinal table key =
  match Hashtbl.find_opt table key with
  | Some n -> n
  | None ->
      let n = Hashtbl.length table in
      Hashtbl.add table key n;
      n

let use nm h = if not (List.mem h nm.used) then nm.used <- h :: nm.used

let call nm h args =
  use nm h;
  R.Name.to_string h ^ "(" ^ String.concat ", " args ^ ")"

let var nm v = "i" ^ string_of_int (ordinal nm.vars (Loop_var.to_int v))
let bound nm v = "n" ^ string_of_int (ordinal nm.vars (Loop_var.to_int v))
let temp nm t = "x" ^ string_of_int (ordinal nm.temps (Loop_temp.to_int t))

let index_temp nm t =
  "o" ^ string_of_int (ordinal nm.index_temps (Loop_temp.to_int t))

let array nm a = "a" ^ string_of_int (ordinal nm.arrays (Loop_array.to_int a))

let buffer nm (b : Loop_buffer.t) =
  "b" ^ string_of_int (ordinal nm.buffers (Tensor_id.to_int b.Loop_buffer.id))

let fmt_of (b : Loop_buffer.t) =
  let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
  Payload.fmt_name f

(* The storage cell of a buffer, by format. A quantized format has no C
   implementation: it is refused by [check_formats] before any text is made. *)
let cell_type b =
  match fmt_of b with
  | "bf16" | "f16" -> Some "uint16_t"
  | "bool" -> Some "uint8_t"
  | "f32" -> Some "float"
  | "f64" -> Some "double"
  | "i32" -> Some "int32_t"
  | "i64" -> Some "int64_t"
  | _ -> None

let int_lit n = "((int64_t)" ^ string_of_int n ^ ")"

let i64_lit n =
  if Int64.equal n Int64.min_int then "(-INT64_C(9223372036854775807) - 1)"
  else if Int64.compare n 0L < 0 then
    "(-INT64_C" ^ "(" ^ Int64.to_string (Int64.neg n) ^ "))"
  else "INT64_C(" ^ Int64.to_string n ^ ")"

let float_lit x =
  if Float.is_nan x then "NAN"
  else if x = Float.infinity then "INFINITY"
  else if x = Float.neg_infinity then "(-INFINITY)"
  else "(" ^ Printf.sprintf "%h" x ^ ")"

(* The literal of the kernel's working type: an fp32 kernel rounds constants to
   binary32 and writes them [f]-suffixed so nothing promotes through double. *)
let lit nm x = if nm.f32 then Loop_numerics.f32_literal x else float_lit x
let float_type nm = if nm.f32 then "float" else "double"

let rec index nm : Loop_index.t -> string = function
  | Loop_index.Add (a, b) ->
      let a = index nm a in
      let b = index nm b in
      "(" ^ a ^ " + " ^ b ^ ")"
  | Loop_index.Ceil_div_pos (a, d) ->
      let a = index nm a in
      "(-" ^ call nm R.Name.Floor_div [ "-" ^ a; int_lit d ] ^ ")"
  | Loop_index.Clamp_low a -> call nm R.Name.Idx_clamp_low [ index nm a ]
  | Loop_index.Const n -> int_lit n
  | Loop_index.Floor_div_pos (a, d) ->
      let a = index nm a in
      call nm R.Name.Floor_div [ a; int_lit d ]
  | Loop_index.Max (a, b) ->
      let a = index nm a in
      let b = index nm b in
      call nm R.Name.Idx_max [ a; b ]
  | Loop_index.Min (a, b) ->
      let a = index nm a in
      let b = index nm b in
      call nm R.Name.Idx_min [ a; b ]
  | Loop_index.Scale (k, a) -> "(" ^ int_lit k ^ " * " ^ index nm a ^ ")"
  | Loop_index.Temp t -> index_temp nm t
  | Loop_index.Var v -> var nm v

(* The dense row-major offset of a coordinate in a buffer's shape, folded as it
   is built: a unit extent scales by one and a constant component adds a
   constant, so neither prints. Only constants small enough that the product
   cannot leave the index domain are folded. *)
let small n = n > -0x4000_0000 && n < 0x4000_0000

let scale_i k (a : Loop_index.t) : Loop_index.t =
  match a with
  | _ when k = 1 -> a
  | Loop_index.Const n when small n && small k && small (k * n) ->
      Loop_index.Const (k * n)
  | _ -> Loop_index.Scale (k, a)

let add_i (a : Loop_index.t) (b : Loop_index.t) : Loop_index.t =
  match (a, b) with
  | Loop_index.Const 0, x | x, Loop_index.Const 0 -> x
  | Loop_index.Const x, Loop_index.Const y when small x && small y ->
      Loop_index.Const (x + y)
  | _ -> Loop_index.Add (a, b)

let offset nm (b : Loop_buffer.t) (c : Loop_index.coord) =
  let shape = b.Loop_buffer.sg.Tensor_sig.shape in
  List.fold_left
    (fun acc a ->
      let extent = Dim.to_int (Vec6.get shape a) in
      let i = Expr.Coord.get c a in
      match acc with
      | None -> Some i
      | Some acc -> Some (add_i (scale_i extent acc) i))
    None Expr.Axis.all
  |> Option.get |> index nm

type overflow_node = { op : int; value : string; lhs : string; rhs : string }

(* One node per checked operation, post-order: the first to leave the int32
   domain is the one the interpreter reports, with its operands. *)
let overflow_nodes nm i =
  let rec go acc (i : Loop_index.t) =
    match i with
    | Loop_index.Add (a, b) ->
        let acc = go (go acc a) b in
        let lhs = index nm a in
        let rhs = index nm b in
        { op = 0; value = index nm i; lhs; rhs } :: acc
    | Loop_index.Scale (k, a) ->
        let acc = go acc a in
        let rhs = index nm a in
        { op = 1; value = index nm i; lhs = int_lit k; rhs } :: acc
    | Loop_index.Ceil_div_pos (a, _)
    | Loop_index.Clamp_low a
    | Loop_index.Floor_div_pos (a, _) ->
        go acc a
    | Loop_index.Max (a, b) | Loop_index.Min (a, b) -> go (go acc a) b
    | Loop_index.Const _ | Loop_index.Temp _ | Loop_index.Var _ -> acc
  in
  List.rev (go [] i)

let outside_int32 v =
  "(" ^ v ^ " < -INT64_C(2147483648) || " ^ v ^ " >= INT64_C(2147483648))"

type addr = At of Loop_index.coord | Flat of Loop_index.t

let addr_offset nm b = function At c -> offset nm b c | Flat i -> index nm i

let load_cell nm b addr =
  let cell = buffer nm b ^ "[" ^ addr_offset nm b addr ^ "]" in
  match (fmt_of b, nm.f32) with
  | "bf16", false -> call nm R.Name.Bf16_to_float [ cell ]
  | "bf16", true -> "(float)" ^ call nm R.Name.Bf16_to_float [ cell ]
  | "bool", false -> "(" ^ cell ^ " != 0 ? 1.0 : 0.0)"
  | "bool", true -> "(" ^ cell ^ " != 0 ? 1.0f : 0.0f)"
  | "f16", false -> call nm R.Name.F16_to_float [ cell ]
  | "f16", true -> "(float)" ^ call nm R.Name.F16_to_float [ cell ]
  | "f32", true -> cell
  | ("f64" | "i32" | "i64"), true -> "(float)" ^ cell
  | ("f32" | "f64" | "i32" | "i64"), false -> "(double)" ^ cell
  | f, _ -> invalid_arg ("Loop_c: no decode for format " ^ f)

(* Binary32: the binary64 helper on the widened argument, rounded once. A
   square root and a truncation are exact in [float], so they use the float
   function. *)
let unary_f32 nm (op : Expr.Value.unary_op) a =
  let wide f = "(float)" ^ f ^ "((double)" ^ a ^ ")" in
  match op with
  | Expr.Value.Cos -> wide "cos"
  | Expr.Value.Erf -> call nm R.Name.Erf_f32 [ a ]
  | Expr.Value.Exp -> wide "exp"
  | Expr.Value.Log -> wide "log"
  | Expr.Value.Sin -> wide "sin"
  | Expr.Value.Sqrt -> "sqrtf(" ^ a ^ ")"
  | Expr.Value.Trunc -> "truncf(" ^ a ^ ")"

let unary_c nm (op : Expr.Value.unary_op) a =
  if nm.f32 then unary_f32 nm op a
  else
    match op with
    | Expr.Value.Cos -> "cos(" ^ a ^ ")"
    | Expr.Value.Erf -> call nm R.Name.Erf [ a ]
    | Expr.Value.Exp -> "exp(" ^ a ^ ")"
    | Expr.Value.Log -> "log(" ^ a ^ ")"
    | Expr.Value.Sin -> "sin(" ^ a ^ ")"
    | Expr.Value.Sqrt -> "sqrt(" ^ a ^ ")"
    | Expr.Value.Trunc -> "trunc(" ^ a ^ ")"

let binary_sym : Expr.Value.binary_op -> string = function
  | Expr.Value.Add -> "+"
  | Expr.Value.Div -> "/"
  | Expr.Value.Mul -> "*"
  | Expr.Value.Sub -> "-"

let rec num nm : float Loop_expr.t -> string = function
  | Loop_expr.Array_get (a, i) -> array nm a ^ "[" ^ index nm i ^ "]"
  | Loop_expr.Binary (op, a, b) ->
      let a = num nm a in
      let b = num nm b in
      "(" ^ a ^ " " ^ binary_sym op ^ " " ^ b ^ ")"
  | Loop_expr.Const x -> lit nm x
  | Loop_expr.Fma (a, b, c) ->
      let a = num nm a in
      let b = num nm b in
      let c = num nm c in
      (if nm.f32 then "fmaf(" else "fma(") ^ a ^ ", " ^ b ^ ", " ^ c ^ ")"
  | Loop_expr.Float_max (a, b) ->
      let a = num nm a in
      let b = num nm b in
      call nm
        (if nm.f32 then R.Name.Float_max_f32 else R.Name.Float_max)
        [ a; b ]
  | Loop_expr.I64_to_float a -> "((" ^ float_type nm ^ ")" ^ big nm a ^ ")"
  | Loop_expr.Load (b, c) -> load_cell nm b (At c)
  | Loop_expr.Load_flat (b, i) -> load_cell nm b (Flat i)
  | Loop_expr.Round_f32 a ->
      if nm.f32 then num nm a else "((double)(float)" ^ num nm a ^ ")"
  | Loop_expr.Select (p, a, b) ->
      let p = pred nm p in
      let a = num nm a in
      let b = num nm b in
      "(" ^ p ^ " ? " ^ a ^ " : " ^ b ^ ")"
  | Loop_expr.Temp (Loop_carrier.Float, t) -> temp nm t
  | Loop_expr.Unary (op, a) -> unary_c nm op (num nm a)
  | Loop_expr.Value_of_index i -> "((" ^ float_type nm ^ ")" ^ index nm i ^ ")"

and big nm : int64 Loop_expr.t -> string = function
  | Loop_expr.Float_to_i64 a -> call nm R.Name.I64_from_float [ num nm a ]
  | Loop_expr.I64_binary (op, a, b) -> (
      let a = big nm a in
      let b = big nm b in
      let wrap sym =
        "((int64_t)((uint64_t)" ^ a ^ " " ^ sym ^ " (uint64_t)" ^ b ^ "))"
      in
      match op with
      | Expr.Value.I64_add -> wrap "+"
      | Expr.Value.I64_div -> call nm R.Name.I64_div [ a; b ]
      | Expr.Value.I64_mul -> wrap "*"
      | Expr.Value.I64_sub -> wrap "-")
  | Loop_expr.I64_const n -> i64_lit n
  | Loop_expr.I64_of_index i -> index nm i
  | Loop_expr.Load_i64 (b, c) -> buffer nm b ^ "[" ^ offset nm b c ^ "]"
  | Loop_expr.Load_i64_flat (b, i) -> buffer nm b ^ "[" ^ index nm i ^ "]"
  | Loop_expr.Select (p, a, b) ->
      let p = pred nm p in
      let a = big nm a in
      let b = big nm b in
      "(" ^ p ^ " ? " ^ a ^ " : " ^ b ^ ")"
  | Loop_expr.Temp (Loop_carrier.Int64, t) -> temp nm t

and pred nm : Loop_expr.pred -> string = function
  | Loop_bool.I64_eq (a, b) ->
      let a = big nm a in
      let b = big nm b in
      "(" ^ a ^ " == " ^ b ^ ")"
  | Loop_bool.I64_lt (a, b) ->
      let a = big nm a in
      let b = big nm b in
      "(" ^ a ^ " < " ^ b ^ ")"
  | Loop_bool.Index_eq (a, b) ->
      let a = index nm a in
      let b = index nm b in
      "(" ^ a ^ " == " ^ b ^ ")"
  | Loop_bool.Index_lt (a, b) ->
      let a = index nm a in
      let b = index nm b in
      "(" ^ a ^ " < " ^ b ^ ")"
  | Loop_bool.Index_overflows i -> (
      match overflow_nodes nm i with
      | [] -> "0"
      | nodes ->
          "("
          ^ String.concat " || "
              (List.map (fun n -> outside_int32 n.value) nodes)
          ^ ")")
  | Loop_bool.Not p -> "(!" ^ pred nm p ^ ")"
  | Loop_bool.Or (p, q) ->
      let p = pred nm p in
      let q = pred nm q in
      "(" ^ p ^ " || " ^ q ^ ")"
  | Loop_bool.Out_of_range (i, n) ->
      let i = index nm i in
      "(" ^ i ^ " < 0 || " ^ i ^ " >= " ^ int_lit n ^ ")"
  | Loop_bool.Pool_better (best, value) ->
      let best = num nm best in
      let value = num nm value in
      (* [Max_op.pool_better]: the candidate wins on strict greater-than OR
         on NaN. Both operands are pure, so naming them twice is exact. *)
      "(" ^ value ^ " > " ^ best ^ " || " ^ value ^ " != " ^ value ^ ")"
  | Loop_bool.Value_eq (a, b) ->
      let a = num nm a in
      let b = num nm b in
      "(" ^ a ^ " == " ^ b ^ ")"
  | Loop_bool.Value_lt (a, b) ->
      let a = num nm a in
      let b = num nm b in
      "(" ^ a ^ " < " ^ b ^ ")"

let kind_num k = string_of_int (R.kind_index k)

(* A failure is a returned status, never an exception or a [longjmp]. The
   fields are the ones [Loop_js_failure.fields] names, in its order, written to
   [err->v]; the expressions are evaluated only here, once the check fired. *)
let set_fields fields =
  String.concat " "
    (List.mapi
       (fun i e -> Printf.sprintf "err->v[%d] = (int64_t)(%s);" i e)
       fields)

let fail_record kind fields =
  Printf.sprintf "{ fail_set(err, %s); %s return 1; }" (kind_num kind)
    (set_fields fields)

let scan_failure nm which ~site ~local ~row ~lane ~extent =
  let row = index nm row in
  let lane = index nm lane in
  fail_record F.Kind.Scan_projection
    [
      (match which with F.Projection.Lane -> "0" | F.Projection.Row -> "1");
      (if Option.is_some local then "1" else "0");
      row;
      lane;
      string_of_int extent;
      string_of_int site;
    ]

let failure nm ~site : Loop_failure.t -> string = function
  | Loop_failure.Load_out_of_range { buffer = b; coord = c } ->
      let shape = b.Loop_buffer.sg.Tensor_sig.shape in
      let extents =
        List.map
          (fun a -> int_lit (Dim.to_int (Vec6.get shape a)))
          Expr.Axis.all
      in
      let coord =
        List.rev
          (List.fold_left
             (fun acc a -> index nm (Expr.Coord.get c a) :: acc)
             [] Expr.Axis.all)
      in
      Printf.sprintf
        "{ const int64_t ext[6] = {%s}; const int64_t co[6] = {%s}; return %s; \
         }"
        (String.concat ", " extents)
        (String.concat ", " coord)
        (call nm R.Name.Coord_failure
           [ "err"; int_lit (Tensor_id.to_int b.Loop_buffer.id); "ext"; "co" ])
  | Loop_failure.Gather_out_of_range { raw; extent } ->
      fail_record F.Kind.Gather_index_out_of_range
        [ big nm raw; string_of_int extent ]
  | Loop_failure.I64_division_by_zero ->
      fail_record F.Kind.I64_division_by_zero []
  | Loop_failure.I64_division_overflow ->
      fail_record F.Kind.I64_division_overflow []
  | Loop_failure.I64_from_float { value } ->
      "return "
      ^ call nm R.Name.I64_from_float_failure [ "err"; num nm value ]
      ^ ";"
  | Loop_failure.Index_overflow _ ->
      invalid_arg "Loop_c.failure: an index overflow is written per node"
  | Loop_failure.Local_out_of_range _ ->
      fail_record F.Kind.Unbound_local [ string_of_int site ]
  | Loop_failure.Scan_lane_out_of_range { local; row; lane; extent } ->
      scan_failure nm F.Projection.Lane ~site ~local ~row ~lane ~extent
  | Loop_failure.Scan_row_out_of_range { local; row; lane; extent } ->
      scan_failure nm F.Projection.Row ~site ~local ~row ~lane ~extent

(* The [Fail_if] sites are numbered in the walk [Loop_js_failure.sites] makes,
   and each is checked against that array by physical equality. *)
let next_site nm f =
  let k = nm.next_site in
  if k >= Array.length nm.sites || nm.sites.(k) != f then
    invalid_arg "Loop_c: a failure site drifted from Loop_js_failure.sites";
  nm.next_site <- k + 1;
  k

let meter_failure which limit =
  fail_record F.Kind.Scan_meter
    [
      (match which with
      | F.Meter.State_over_limit -> "0"
      | F.Meter.Updates_exhausted -> "1");
      limit;
    ]

let indent n = String.make (2 * n) ' '

module R = Loop_c_runtime
module F = Loop_js_failure

type error = [ `Unsupported_format of Tensor_id.t * string ]

let pp_error ppf : [< error ] -> unit = function
  | `Unsupported_format (id, f) ->
      Format.fprintf ppf "t%d: format %s has no C implementation"
        (Tensor_id.to_int id) f

type t = {
  source : string;
  helpers : R.Name.t list;
  local_doubles : int64;
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
  match fmt_of b with
  | "bf16" -> call nm R.Name.Bf16_to_float [ cell ]
  | "bool" -> "(" ^ cell ^ " != 0 ? 1.0 : 0.0)"
  | "f16" -> call nm R.Name.F16_to_float [ cell ]
  | "f32" | "f64" | "i32" | "i64" -> "(double)" ^ cell
  | f -> invalid_arg ("Loop_c: no decode for format " ^ f)

let unary_c nm (op : Expr.Value.unary_op) a =
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
  | Loop_expr.Const x -> float_lit x
  | Loop_expr.Float_max (a, b) ->
      let a = num nm a in
      let b = num nm b in
      call nm R.Name.Float_max [ a; b ]
  | Loop_expr.I64_to_float a -> "((double)" ^ big nm a ^ ")"
  | Loop_expr.Load (b, c) -> load_cell nm b (At c)
  | Loop_expr.Load_flat (b, i) -> load_cell nm b (Flat i)
  | Loop_expr.Round_f32 a -> "((double)(float)" ^ num nm a ^ ")"
  | Loop_expr.Select (p, a, b) ->
      let p = pred nm p in
      let a = num nm a in
      let b = num nm b in
      "(" ^ p ^ " ? " ^ a ^ " : " ^ b ^ ")"
  | Loop_expr.Temp (Loop_carrier.Float, t) -> temp nm t
  | Loop_expr.Unary (op, a) -> unary_c nm op (num nm a)
  | Loop_expr.Value_of_index i -> "((double)" ^ index nm i ^ ")"

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

(* ---- vector loops --------------------------------------------------------- *)

module V = Loop_vector

(* A vector expression as C text of type [v4df] (a mask as [v4di]). Vector
   temporaries are named by their id; a splatted scalar by its position among the
   loop's splats. *)
type vstate = { mutable splats : (float Loop_expr.t * string) list }

let lane_offset (a : V.Access.t) k : Loop_index.t =
  if k = 0 then a.V.Access.offset
  else
    Loop_index.Add (a.V.Access.offset, Loop_index.Const (k * a.V.Access.stride))

let vtemp t = "vt" ^ string_of_int (V.Temp.to_int t)

let vload nm (a : V.Access.t) =
  let b = a.V.Access.buffer in
  let scalar k = load_cell nm b (Flat (lane_offset a k)) in
  let lanes () = String.concat ", " (List.init 4 scalar) in
  if a.V.Access.stride = 0 then "vf_splat(" ^ scalar 0 ^ ")"
  else if a.V.Access.stride <> 1 then "((v4df){" ^ lanes () ^ "})"
  else
    let at = "(" ^ buffer nm b ^ " + " ^ index nm a.V.Access.offset ^ ")" in
    match fmt_of b with
    | "f32" -> "vf_load_f32(" ^ at ^ ")"
    | "f64" -> "vf_load_f64(" ^ at ^ ")"
    | "i32" -> "vf_load_i32(" ^ at ^ ")"
    | _ -> "((v4df){" ^ lanes () ^ "})"

let rec vexpr nm vs (e : V.t) : string =
  match e with
  | V.Const x -> "vf_splat(" ^ float_lit x ^ ")"
  | V.Splat s -> List.assq s vs.splats
  | V.Binary (op, a, b) ->
      let a = vexpr nm vs a in
      let b = vexpr nm vs b in
      "(" ^ a ^ " " ^ binary_sym op ^ " " ^ b ^ ")"
  | V.Float_max (a, b) ->
      let a = vexpr nm vs a in
      let b = vexpr nm vs b in
      "vf_max(" ^ a ^ ", " ^ b ^ ")"
  | V.Round_f32 a -> "vf_round_f32(" ^ vexpr nm vs a ^ ")"
  | V.Unary (op, a) ->
      let a = vexpr nm vs a in
      (match op with Expr.Value.Erf -> use nm R.Name.Erf | _ -> ());
      let name =
        match op with
        | Expr.Value.Cos -> "vf_cos"
        | Expr.Value.Erf -> "vf_erf"
        | Expr.Value.Exp -> "vf_exp"
        | Expr.Value.Log -> "vf_log"
        | Expr.Value.Sin -> "vf_sin"
        | Expr.Value.Sqrt -> "vf_sqrt"
        | Expr.Value.Trunc -> "vf_trunc"
      in
      name ^ "(" ^ a ^ ")"
  | V.Select (m, a, b) ->
      let m = vmask nm vs m in
      let a = vexpr nm vs a in
      let b = vexpr nm vs b in
      "vf_sel(" ^ m ^ ", " ^ a ^ ", " ^ b ^ ")"
  | V.Temp t -> vtemp t
  | V.Index_value { base; step } ->
      "((v4df){"
      ^ String.concat ", "
          (List.init 4 (fun k ->
               "(double)"
               ^ index nm (Loop_index.Add (base, Loop_index.Const (k * step)))))
      ^ "})"
  | V.Load a -> vload nm a

and vmask nm vs (m : V.mask) : string =
  match m with
  | V.Not m -> "(~" ^ vmask nm vs m ^ ")"
  | V.Or (a, b) -> "(" ^ vmask nm vs a ^ " | " ^ vmask nm vs b ^ ")"
  | V.Value_eq (a, b) -> "(" ^ vexpr nm vs a ^ " == " ^ vexpr nm vs b ^ ")"
  | V.Value_lt (a, b) -> "(" ^ vexpr nm vs a ^ " < " ^ vexpr nm vs b ^ ")"
  | V.Pool_better (best, value) ->
      (* The candidate wins on strict greater-than or on NaN. Both operands are
         pure, so naming the value twice is exact. *)
      let best = vexpr nm vs best in
      let value = vexpr nm vs value in
      "((" ^ value ^ " > " ^ best ^ ") | (" ^ value ^ " != " ^ value ^ "))"

let vstore nm vs ~ind (a : V.Access.t) (value : V.stored) =
  let b = a.V.Access.buffer in
  let e = match value with V.F32 e | V.Bool e -> e in
  let v = vexpr nm vs e in
  match value with
  | V.F32 _ when a.V.Access.stride = 1 ->
      [
        Printf.sprintf "%svf_store_f32(%s + %s, %s);" ind (buffer nm b)
          (index nm a.V.Access.offset)
          v;
      ]
  | _ ->
      let lane k =
        let cell = buffer nm b ^ "[" ^ index nm (lane_offset a k) ^ "]" in
        match value with
        | V.F32 _ -> Printf.sprintf "%s%s = (float)vs[%d];" ind cell k
        | V.Bool _ -> Printf.sprintf "%s%s = vs[%d] != 0.0 ? 1 : 0;" ind cell k
      in
      [ Printf.sprintf "%s{ const v4df vs = %s;" ind v ]
      @ List.init 4 lane
      @ [ ind ^ "}" ]

let rec collect_splats acc (e : V.t) =
  match e with
  | V.Splat s -> if List.memq s acc then acc else acc @ [ s ]
  | V.Binary (_, a, b) | V.Float_max (a, b) ->
      collect_splats (collect_splats acc a) b
  | V.Round_f32 a | V.Unary (_, a) -> collect_splats acc a
  | V.Select (m, a, b) ->
      collect_splats (collect_splats (mask_splats acc m) a) b
  | V.Const _ | V.Index_value _ | V.Load _ | V.Temp _ -> acc

and mask_splats acc (m : V.mask) =
  match m with
  | V.Not m -> mask_splats acc m
  | V.Or (a, b) -> mask_splats (mask_splats acc a) b
  | V.Pool_better (a, b) | V.Value_eq (a, b) | V.Value_lt (a, b) ->
      collect_splats (collect_splats acc a) b

let vloop nm ~depth ~scalar_stmt (l : V.loop) : string list =
  let ind = indent depth in
  use nm R.Name.Vector_prelude;
  use nm R.Name.Float_max;
  use nm R.Name.Erf;
  let lo, hi =
    match (l.V.lo, l.V.hi) with
    | Loop_index.Const lo, Loop_index.Const hi -> (lo, hi)
    | _ -> invalid_arg "Loop_c: a vector loop without constant bounds"
  in
  let stop = lo + (max 0 (hi - lo) / l.V.lanes * l.V.lanes) in
  let vs = { splats = [] } in
  let exprs =
    List.concat_map
      (fun (s : V.stmt) ->
        match s with
        | V.Assign (_, e) -> [ e ]
        | V.Store { value = V.F32 e | V.Bool e; _ } -> [ e ])
      l.V.body
  in
  let splat_exprs = List.fold_left collect_splats [] exprs in
  let prelude =
    List.mapi
      (fun k e ->
        let name = Printf.sprintf "vs%d" k in
        vs.splats <- vs.splats @ [ (e, name) ];
        Printf.sprintf "%s  const v4df %s = vf_splat(%s);" ind name (num nm e))
      splat_exprs
  in
  let iv = var nm l.V.var in
  let body =
    List.concat_map
      (fun (s : V.stmt) ->
        match s with
        | V.Assign (t, e) ->
            [
              Printf.sprintf "%s    const v4df %s = %s; (void)%s;" ind (vtemp t)
                (vexpr nm vs e) (vtemp t);
            ]
        | V.Store { access; value } ->
            vstore nm vs ~ind:(ind ^ "    ") access value)
      l.V.body
  in
  let remainder =
    match l.V.scalar with
    | Loop_stmt.For f ->
        scalar_stmt (Loop_stmt.For { f with lo = Loop_index.Const stop })
    | s -> scalar_stmt s
  in
  [ ind ^ "{" ]
  @ prelude
  @ [
      Printf.sprintf "%s  for (int64_t %s = %s; %s < %s; %s += %d) {" ind iv
        (int_lit lo) iv (int_lit stop) iv l.V.lanes;
    ]
  @ body
  @ [ ind ^ "  }" ]
  @ remainder
  @ [ ind ^ "}" ]

let rec stmt nm ~limits ~depth (s : Loop_stmt.t) : string list =
  let ind = indent depth in
  let line l = [ ind ^ l ] in
  match s with
  | Loop_stmt.Alloc (a, n) ->
      let n = (n :> int) in
      let off = nm.local_doubles in
      nm.local_doubles <- Int64.add off (Int64.of_int n);
      let a = array nm a in
      line
        (Printf.sprintf
           "double *%s = local + %Ld; memset(%s, 0, %d * sizeof(double));" a off
           a n)
  | Loop_stmt.Array_set (a, i, e) ->
      let i = index nm i in
      let e = num nm e in
      line (Printf.sprintf "%s[%s] = %s;" (array nm a) i e)
  | Loop_stmt.Assign (Loop_carrier.Float, t, e) ->
      line (Printf.sprintf "%s = %s;" (temp nm t) (num nm e))
  | Loop_stmt.Assign (Loop_carrier.Int64, t, e) ->
      line (Printf.sprintf "%s = %s;" (temp nm t) (big nm e))
  | Loop_stmt.Assign_index_of_i64 (t, e) ->
      line (Printf.sprintf "%s = %s;" (index_temp nm t) (big nm e))
  | Loop_stmt.Assign_index (t, i) ->
      line (Printf.sprintf "%s = %s;" (index_temp nm t) (index nm i))
  | Loop_stmt.Fail_if (p, f) -> (
      let site = next_site nm f in
      match (p, f) with
      | Loop_bool.Index_overflows i, Loop_failure.Index_overflow { index = j }
        when i = j ->
          List.map
            (fun n ->
              ind
              ^ Printf.sprintf "if %s %s" (outside_int32 n.value)
                  (fail_record F.Kind.Index_overflow
                     [ string_of_int n.op; n.lhs; n.rhs ]))
            (overflow_nodes nm i)
      | _, Loop_failure.Index_overflow _ ->
          invalid_arg
            "Loop_c: an index overflow failure under a foreign predicate"
      | _ ->
          let p = pred nm p in
          line (Printf.sprintf "if %s %s" p (failure nm ~site f)))
  | Loop_stmt.For { var = v; lo; hi = hi_ix; body } ->
      let name = var nm v in
      let lo = index nm lo in
      let hi = index nm hi_ix in
      (* The interpreter evaluates [hi] once, on entry; a C [for] test runs per
         iteration. A literal or a loop variable cannot change meanwhile, so it
         stays in the test; anything else is bound once before the loop. *)
      let inline, limit =
        match hi_ix with
        | Loop_index.Const _ | Loop_index.Var _ -> ([], hi)
        | _ ->
            let n = bound nm v in
            ([ ind ^ "  const int64_t " ^ n ^ " = " ^ hi ^ ";" ], n)
      in
      [ ind ^ "{" ]
      @ inline
      @ [
          Printf.sprintf "%s  for (int64_t %s = %s; %s < %s; %s++) {" ind name
            lo name limit name;
        ]
      @ block nm ~limits ~depth:(depth + 2) body
      @ [ ind ^ "  }"; ind ^ "}" ]
  | Loop_stmt.If (p, yes, no) ->
      let p = pred nm p in
      let yes = block nm ~limits ~depth:(depth + 1) yes in
      let no = block nm ~limits ~depth:(depth + 1) no in
      [ ind ^ "if " ^ p ^ " {" ]
      @ yes
      @
      if no = [] then [ ind ^ "}" ]
      else [ ind ^ "} else {" ] @ no @ [ ind ^ "}" ]
  | Loop_stmt.Charge_scan_update ->
      let limit = i64_lit (Expr.Scan_limits.max_updates limits) in
      line
        (Printf.sprintf "if (scan_remaining <= 0) %s"
           (meter_failure F.Meter.Updates_exhausted limit))
      @ line "scan_remaining -= 1;"
  | Loop_stmt.Mark _ -> []
  | Loop_stmt.Release_scan_state width ->
      line (Printf.sprintf "scan_live -= %d;" (2 * width))
  | Loop_stmt.Reserve_scan_state width ->
      let live = 2 * width in
      let max_state = Expr.Scan_limits.max_state limits in
      line
        (Printf.sprintf "if (scan_live + %d > %s) %s" live (int_lit max_state)
           (meter_failure F.Meter.State_over_limit (string_of_int max_state)))
      @ line (Printf.sprintf "scan_live += %d;" live)
  | Loop_stmt.Reset_meter ->
      line
        (Printf.sprintf "scan_remaining = %s;"
           (i64_lit (Expr.Scan_limits.max_updates limits)))
      @ line "scan_live = 0;"
  | Loop_stmt.Store { buffer = b; coord = c; value } ->
      line (store nm b (At c) value)
  | Loop_stmt.Store_flat { buffer = b; offset = i; value } ->
      line (store nm b (Flat i) value)

and store nm b addr value =
  let cell () = buffer nm b ^ "[" ^ addr_offset nm b addr ^ "]" in
  match value with
  | Loop_stored.Bool e ->
      (* Evaluation order between the target and the value does not matter:
         both are pure. *)
      let e = num nm e in
      let cell = cell () in
      Printf.sprintf "%s = (%s) != 0.0 ? 1 : 0;" cell e
  | Loop_stored.F32 (Loop_expr.Round_f32 e) | Loop_stored.F32 e ->
      let e = num nm e in
      Printf.sprintf "%s = (float)%s;" (cell ()) e
  | Loop_stored.I64 e ->
      let e = big nm e in
      Printf.sprintf "%s = %s;" (cell ()) e

and block nm ~limits ~depth body = List.concat_map (stmt nm ~limits ~depth) body

and node nm ~limits ~depth (nd : V.node) : string list =
  let ind = indent depth in
  match nd with
  | V.Scalar s -> stmt nm ~limits ~depth s
  | V.If (p, yes, no) ->
      let p = pred nm p in
      let yes = nodes nm ~limits ~depth:(depth + 1) yes in
      let no = nodes nm ~limits ~depth:(depth + 1) no in
      [ ind ^ "if " ^ p ^ " {" ]
      @ yes
      @
      if no = [] then [ ind ^ "}" ]
      else [ ind ^ "} else {" ] @ no @ [ ind ^ "}" ]
  | V.Loop { var = v; lo; hi = hi_ix; body } ->
      let name = var nm v in
      let lo = index nm lo in
      let hi = index nm hi_ix in
      let inline, limit =
        match hi_ix with
        | Loop_index.Const _ | Loop_index.Var _ -> ([], hi)
        | _ ->
            let n = bound nm v in
            ([ ind ^ "  const int64_t " ^ n ^ " = " ^ hi ^ ";" ], n)
      in
      [ ind ^ "{" ]
      @ inline
      @ [
          Printf.sprintf "%s  for (int64_t %s = %s; %s < %s; %s++) {" ind name
            lo name limit name;
        ]
      @ nodes nm ~limits ~depth:(depth + 2) body
      @ [ ind ^ "  }"; ind ^ "}" ]
  | V.Vector l -> vloop nm ~depth ~scalar_stmt:(stmt nm ~limits ~depth) l

and nodes nm ~limits ~depth ns = List.concat_map (node nm ~limits ~depth) ns

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
    | Loop_stmt.Mark _ | Loop_stmt.Store _ | Loop_stmt.Store_flat _ ->
        ()
  in
  List.iter go p.Loop_program.body;
  (List.rev !floats, List.rev !int64s, List.rev !indices, !meter)

let check_formats (p : Loop_program.t) =
  match
    List.find_opt (fun b -> Option.is_none (cell_type b)) p.Loop_program.buffers
  with
  | None -> Ok ()
  | Some b -> Error (`Unsupported_format (b.Loop_buffer.id, fmt_of b))

let param_type (b : Loop_buffer.t) =
  let t = Option.get (cell_type b) in
  match b.Loop_buffer.role with
  | Loop_buffer.Input -> "const " ^ t
  | Loop_buffer.Output | Loop_buffer.Scratch -> t

let kernel ?vector ~name (p : Loop_program.t) : (t, [> error ]) Err.t =
  match check_formats p with
  | Error e -> Err.fail e
  | Ok () ->
      let nm =
        {
          vars = Hashtbl.create 8;
          temps = Hashtbl.create 8;
          index_temps = Hashtbl.create 8;
          arrays = Hashtbl.create 8;
          buffers = Hashtbl.create 8;
          sites = F.sites p;
          next_site = 0;
          used = [];
          local_doubles = 0L;
        }
      in
      (* Buffers take their positions in program order, before any use. *)
      let params =
        List.map
          (fun (b : Loop_buffer.t) ->
            Printf.sprintf "%s *%s" (param_type b) (buffer nm b))
          p.Loop_program.buffers
      in
      let limits = p.Loop_program.scan_limits in
      let body =
        match vector with
        | None -> block nm ~limits ~depth:1 p.Loop_program.body
        | Some target ->
            let vp, _ = Loop_vectorize.program ~target p in
            (match Err.payload (Loop_vector_check.program vp) with
            | Ok () -> ()
            | Error e ->
                invalid_arg
                  (Fmt.str "Loop_c: the vectorizer built an invalid program: %a"
                     Loop_vector_check.pp_error e));
            nodes nm ~limits ~depth:1 vp.Loop_vector.body
      in
      if nm.next_site <> Array.length nm.sites then
        invalid_arg "Loop_c: a failure site was not written";
      let floats, int64s, indices, meter = declarations p in
      let decl ty init ids name_of =
        List.map
          (fun t -> Printf.sprintf "  %s %s = %s;" ty (name_of nm t) init)
          ids
      in
      (* A temporary the program only assigns would trip the compiler's
         set-but-unused warning: name each once. *)
      let named =
        List.map (fun t -> temp nm t) floats
        @ List.map (fun t -> index_temp nm t) indices
        @ List.map (fun t -> temp nm t) int64s
      in
      let temps =
        decl "double" "0.0" floats temp
        @ decl "int64_t" "0" indices index_temp
        @ decl "int64_t" "0" int64s temp
        @ List.map (fun n -> Printf.sprintf "  (void)%s;" n) named
      in
      let meter =
        if meter then
          [
            Printf.sprintf "  int64_t scan_remaining = %s;"
              (i64_lit (Expr.Scan_limits.max_updates limits));
            "  int64_t scan_live = 0;";
            "  (void)scan_live;";
          ]
        else []
      in
      let params_text =
        String.concat ", "
          ("struct model_error *err" :: "double *local" :: params)
      in
      let voids =
        "  (void)err; (void)local;"
        :: List.map
             (fun (b : Loop_buffer.t) ->
               Printf.sprintf "  (void)%s;" (buffer nm b))
             p.Loop_program.buffers
      in
      let source =
        String.concat "\n"
          ([ Printf.sprintf "static int %s(%s) {" name params_text ]
          @ voids @ temps @ meter @ body @ [ "  return 0;"; "}"; "" ])
      in
      let used = nm.used in
      Err.return
        {
          source;
          helpers = List.filter (fun n -> List.mem n used) R.Name.all;
          local_doubles = nm.local_doubles;
          buffer_types =
            List.map (fun b -> Option.get (cell_type b)) p.Loop_program.buffers;
        }

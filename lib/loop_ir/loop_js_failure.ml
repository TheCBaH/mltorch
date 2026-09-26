module Kind = struct
  type t =
    | Coord_out_of_range
    | Defect
    | Gather_index_out_of_range
    | I64_division_by_zero
    | I64_division_overflow
    | I64_from_float_infinite
    | I64_from_float_nan
    | I64_from_float_out_of_range
    | Index_overflow
    | Scan_meter
    | Scan_projection
    | Unbound_local

  let all =
    [
      Coord_out_of_range;
      Defect;
      Gather_index_out_of_range;
      I64_division_by_zero;
      I64_division_overflow;
      I64_from_float_infinite;
      I64_from_float_nan;
      I64_from_float_out_of_range;
      Index_overflow;
      Scan_meter;
      Scan_projection;
      Unbound_local;
    ]

  let to_string = function
    | Coord_out_of_range -> "coord_out_of_range"
    | Defect -> "defect"
    | Gather_index_out_of_range -> "gather_index_out_of_range"
    | I64_division_by_zero -> "i64_division_by_zero"
    | I64_division_overflow -> "i64_division_overflow"
    | I64_from_float_infinite -> "i64_from_float_infinite"
    | I64_from_float_nan -> "i64_from_float_nan"
    | I64_from_float_out_of_range -> "i64_from_float_out_of_range"
    | Index_overflow -> "index_overflow"
    | Scan_meter -> "scan_meter"
    | Scan_projection -> "scan_projection"
    | Unbound_local -> "unbound_local"

  let of_string s = List.find_opt (fun k -> to_string k = s) all
end

module Field = struct
  type t =
    | Axis
    | Buffer
    | Cached
    | Coord
    | Extent
    | Index
    | Lane
    | Lhs
    | Limit
    | Op
    | Raw
    | Rhs
    | Row
    | Site
    | Value
    | Which

  let to_string = function
    | Axis -> "axis"
    | Buffer -> "buffer"
    | Cached -> "cached"
    | Coord -> "coord"
    | Extent -> "extent"
    | Index -> "index"
    | Lane -> "lane"
    | Lhs -> "lhs"
    | Limit -> "limit"
    | Op -> "op"
    | Raw -> "raw"
    | Rhs -> "rhs"
    | Row -> "row"
    | Site -> "site"
    | Value -> "value"
    | Which -> "which"
end

let kind_key = "kind"

let fields : Kind.t -> Field.t list = function
  | Kind.Coord_out_of_range -> [ Field.Buffer; Axis; Index; Coord ]
  | Kind.Defect -> []
  | Kind.Gather_index_out_of_range -> [ Field.Raw; Extent ]
  | Kind.I64_division_by_zero | Kind.I64_division_overflow -> []
  | Kind.I64_from_float_infinite | Kind.I64_from_float_nan -> []
  | Kind.I64_from_float_out_of_range -> [ Field.Value ]
  | Kind.Index_overflow -> [ Field.Op; Lhs; Rhs ]
  | Kind.Scan_meter -> [ Field.Which; Limit ]
  | Kind.Scan_projection -> [ Field.Which; Cached; Row; Lane; Extent; Site ]
  | Kind.Unbound_local -> [ Field.Site ]

let record kind given =
  let expected = fields kind in
  if List.map fst given <> expected then
    invalid_arg
      (Printf.sprintf "Loop_js_failure.record %s: fields must be [%s]"
         (Kind.to_string kind)
         (String.concat "; " (List.map Field.to_string expected)));
  Js_build.record
    ((kind_key, Js_build.string (Kind.to_string kind))
    :: List.map (fun (f, e) -> (Field.to_string f, e)) given)

let of_table table s =
  List.find_map (fun (v, n) -> if n = s then Some v else None) table

module Overflow_op = struct
  type t = Add | Mul

  let table = [ (Add, "add"); (Mul, "mul") ]
  let to_string v = List.assoc v table
  let of_string = of_table table
end

module Projection = struct
  type t = Lane | Row

  let table = [ (Lane, "lane"); (Row, "row") ]
  let to_string v = List.assoc v table
  let of_string = of_table table
end

module Meter = struct
  type t = State_over_limit | Updates_exhausted

  let table =
    [
      (State_over_limit, "state_over_limit");
      (Updates_exhausted, "updates_exhausted");
    ]

  let to_string v = List.assoc v table
  let of_string = of_table table
end

let sites (p : Loop_program.t) =
  let acc = ref [] in
  let rec go (s : Loop_stmt.t) =
    match s with
    | Loop_stmt.Fail_if (_, f) -> acc := f :: !acc
    | Loop_stmt.For { body; _ } -> List.iter go body
    | Loop_stmt.If (_, yes, no) ->
        List.iter go yes;
        List.iter go no
    | Loop_stmt.Alloc _ | Loop_stmt.Array_set _ | Loop_stmt.Assign _
    | Loop_stmt.Assign_index _ | Loop_stmt.Assign_index_of_i64 _
    | Loop_stmt.Charge_scan_update | Loop_stmt.Mark _
    | Loop_stmt.Release_scan_state _ | Loop_stmt.Reserve_scan_state _
    | Loop_stmt.Reset_meter | Loop_stmt.Store _ ->
        ()
  in
  List.iter go p.Loop_program.body;
  Array.of_list (List.rev !acc)

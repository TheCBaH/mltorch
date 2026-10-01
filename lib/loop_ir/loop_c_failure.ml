module F = Loop_js_failure

let ( let* ) = Result.bind

(* [v] words are int64; an index-sized field is bounded before it is narrowed,
   since js_of_ocaml's [int] is 32 bits. *)
let int_of (v : int64) =
  if Int64.compare v (-2147483648L) >= 0 && Int64.compare v 2147483647L <= 0
  then Ok (Int64.to_int v)
  else Error "field is not a 32-bit integer"

let word (v : int64 array) i =
  if i >= 0 && i < Array.length v then Ok v.(i) else Error "missing field"

let int_word v i =
  let* w = word v i in
  int_of w

let site_failure sites v i =
  let* site = int_word v i in
  if site >= 0 && site < Array.length sites then Ok sites.(site)
  else Error "failure site out of range"

let decode ~(sites : Loop_failure.t array) ~kind ~(v : int64 array) :
    (Loop_interp.error, string) result =
  match List.nth_opt F.Kind.all kind with
  | None -> Error (Printf.sprintf "unknown failure kind %d" kind)
  | Some k -> (
      match k with
      | F.Kind.Coord_out_of_range ->
          let* buffer = int_word v 0 in
          let* axis = int_word v 1 in
          let* index = int_word v 2 in
          let* n = int_word v 3 in
          let* t = int_word v 4 in
          let* d = int_word v 5 in
          let* h = int_word v 6 in
          let* w = int_word v 7 in
          let* c = int_word v 8 in
          if axis < 0 || axis >= List.length Expr.Axis.all then
            Error "axis out of range"
          else
            Ok
              (`Coord_out_of_range
                 ( Expr_bridge.source_of_id (Tensor_id.of_int buffer),
                   List.nth Expr.Axis.all axis,
                   index,
                   Expr.Coord.make ~n ~t ~d ~h ~w ~c ))
      | F.Kind.Defect -> Error "coord_failure found no axis out of range"
      | F.Kind.Gather_index_out_of_range ->
          let* raw = word v 0 in
          let* extent = int_word v 1 in
          Ok
            (`Gather_index_out_of_range
               { Expr.Eval.Gather_index_out_of_range.raw; extent })
      | F.Kind.I64_division_by_zero -> Ok `I64_division_by_zero
      | F.Kind.I64_division_overflow -> Ok `I64_division_overflow
      | F.Kind.I64_from_float_infinite -> Ok `I64_from_float_infinite
      | F.Kind.I64_from_float_nan -> Ok `I64_from_float_nan
      | F.Kind.I64_from_float_out_of_range ->
          let* bits = word v 0 in
          Ok (`I64_from_float_out_of_range (Int64.float_of_bits bits))
      | F.Kind.Index_overflow -> (
          let* op = int_word v 0 in
          let* lhs = int_word v 1 in
          let* rhs = int_word v 2 in
          match op with
          | 0 ->
              Ok (`Index_overflow { Expr.Index_overflow.op = `Add; lhs; rhs })
          | 1 ->
              Ok (`Index_overflow { Expr.Index_overflow.op = `Mul; lhs; rhs })
          | _ -> Error "unknown index overflow operator")
      | F.Kind.Scan_meter -> (
          let* which = int_word v 0 in
          match which with
          | 1 ->
              let* limit = word v 1 in
              Ok (`Scan_meter (Expr.Scan_meter.Updates_exhausted { limit }))
          | 0 ->
              let* limit = int_word v 1 in
              Ok (`Scan_meter (Expr.Scan_meter.State_over_limit { limit }))
          | _ -> Error "unknown scan meter kind")
      | F.Kind.Scan_projection -> (
          let* which = int_word v 0 in
          let* row = int_word v 2 in
          let* lane = int_word v 3 in
          let* extent = int_word v 4 in
          let* site = site_failure sites v 5 in
          let bounds local =
            {
              Expr.Eval.Scan_bounds.projection =
                { Expr.Eval.Scan_projection.local; row; lane };
              extent;
            }
          in
          match (which, site) with
          | 0, Loop_failure.Scan_lane_out_of_range { local; _ } ->
              Ok (`Scan_projection (Expr.Eval.Lane_out_of_range (bounds local)))
          | 1, Loop_failure.Scan_row_out_of_range { local; _ } ->
              Ok (`Scan_projection (Expr.Eval.Row_out_of_range (bounds local)))
          | _ -> Error "scan projection record disagrees with its site")
      | F.Kind.Unbound_local -> (
          let* site = site_failure sites v 0 in
          match site with
          | Loop_failure.Local_out_of_range { local; _ } ->
              Ok (`Unbound_local local)
          | _ -> Error "unbound_local record disagrees with its site"))

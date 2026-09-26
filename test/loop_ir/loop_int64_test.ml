open Loop_ir
open Loop_fixtures
open Loop_programs

(* The int64 track: exact values past 2^53, modular arithmetic, checked division
   and float conversion, and the reductions with their own seeds. *)

let floats = [| 2.9; -2.9; 0.; 0. |]

let show name ?(floats = floats) ~cells body =
  let kernel = i64_body_kernel body in
  let plan = Fusion_plan.default kernel in
  let bind = i64_bind ~floats ~cells in
  Fmt.pr "%s: %a" name Loop_check.pp_verdict (Loop_check.run plan ~bind);
  (match Err.payload (Loop_lower.lower plan) with
  | Error _ -> ()
  | Ok program -> (
      match Err.payload (Loop_interp.run program ~bind) with
      | Error e ->
          Fmt.pr "; failed: %s" (Loop_check.kind (e :> Kernel_eval.error))
      | Ok m -> (
          let (Tensor.Tensor t) = Tensor_id.Map.find (tid 1) m in
          match t.Tensor.payload with
          | { Payload.fmt = Payload.I64; data; _ } ->
              Fmt.pr "; %s"
                (String.concat " "
                   (List.init 4 (fun i -> Int64.to_string data.{i})))
          | _ -> ())));
  Fmt.pr "@."

let big = Int64.add (Int64.shift_left 1L 53) 1L

let%expect_test "a value past 2^53 stays exact through a load and an add" =
  show "2^53+1, plus one"
    ~cells:[| big; Int64.max_int; Int64.min_int; 0L |]
    (Expr.Value.i64_add i64_here (Expr.Value.i64_const 1L));
  [%expect
    {| 2^53+1, plus one: agree; 9007199254740994 -9223372036854775808 -9223372036854775807 1 |}]

let%expect_test "add, sub and mul are modular" =
  let cells = [| Int64.max_int; Int64.min_int; 3L; -5L |] in
  show "max_int + 1" ~cells
    (Expr.Value.i64_add i64_here (Expr.Value.i64_const 1L));
  show "min_int - 1" ~cells
    (Expr.Value.i64_sub i64_here (Expr.Value.i64_const 1L));
  show "cell * cell" ~cells (Expr.Value.i64_mul i64_here i64_here);
  [%expect
    {|
    max_int + 1: agree; -9223372036854775808 -9223372036854775807 4 -4
    min_int - 1: agree; 9223372036854775806 9223372036854775807 2 -6
    cell * cell: agree; 1 0 9 25 |}]

let%expect_test
    "division truncates toward zero and fails on zero and min_int / -1" =
  let cells = [| -7L; 7L; -7L; 7L |] in
  show "x / 2" ~cells (Expr.Value.i64_div i64_here (Expr.Value.i64_const 2L));
  show "x / -2" ~cells
    (Expr.Value.i64_div i64_here (Expr.Value.i64_const (-2L)));
  show "x / 0" ~cells (Expr.Value.i64_div i64_here (Expr.Value.i64_const 0L));
  show "min_int / -1"
    ~cells:[| Int64.min_int; 1L; 1L; 1L |]
    (Expr.Value.i64_div i64_here (Expr.Value.i64_const (-1L)));
  show "min_int / 1"
    ~cells:[| Int64.min_int; 1L; 1L; 1L |]
    (Expr.Value.i64_div i64_here (Expr.Value.i64_const 1L));
  [%expect
    {|
    x / 2: agree; -3 3 -3 3
    x / -2: agree; 3 -3 3 -3
    x / 0: agree on failure: i64_division_by_zero; failed: i64_division_by_zero
    min_int / -1: agree on failure: i64_division_overflow; failed: i64_division_overflow
    min_int / 1: agree; -9223372036854775808 1 1 1 |}]

let float_to_i64 =
  Expr.Value.float_to_i64
    (Expr.Value.load (Expr_bridge.source_of_id (tid 0)) out_coord)

let%expect_test
    "float to int64 truncates, and refuses NaN, infinities and out of range" =
  let case name x =
    show name ~floats:[| x; x; x; x |] ~cells:[| 0L; 0L; 0L; 0L |] float_to_i64
  in
  case "2.9" 2.9;
  case "-2.9" (-2.9);
  case "-0" (-0.);
  case "-2^63" (-.Float.pow 2. 63.);
  case "2^63" (Float.pow 2. 63.);
  case "nan" nan;
  case "inf" infinity;
  case "-inf" neg_infinity;
  case "1e30" 1e30;
  [%expect
    {|
    2.9: agree; 2 2 2 2
    -2.9: agree; -2 -2 -2 -2
    -0: agree; 0 0 0 0
    -2^63: agree; -9223372036854775808 -9223372036854775808 -9223372036854775808 -9223372036854775808
    2^63: agree on failure: i64_from_float_out_of_range; failed: i64_from_float_out_of_range
    nan: agree on failure: i64_from_float_nan; failed: i64_from_float_nan
    inf: agree on failure: i64_from_float_infinite; failed: i64_from_float_infinite
    -inf: agree on failure: i64_from_float_infinite; failed: i64_from_float_infinite
    1e30: agree on failure: i64_from_float_out_of_range; failed: i64_from_float_out_of_range |}]

let%expect_test "an int64 read through a float keeps the nearest double" =
  show "float_of_int64 (2^53+1) then back" ~cells:[| big; 3L; -3L; 0L |]
    (Expr.Value.float_to_i64 (Expr.Value.i64_to_float i64_here));
  [%expect
    {| float_of_int64 (2^53+1) then back: agree; 9007199254740992 3 -3 0 |}]

(* ---- reductions ------------------------------------------------------------- *)

let%expect_test "int64 reductions keep their own seeds" =
  let cells = [| 5L; -9L; 5L; 2L |] in
  show "sum" ~cells (reduce_w Expr.Reduction.Sum ());
  show "max" ~cells (reduce_w Expr.Reduction.Max ());
  show "argmax value" ~cells (reduce_w Expr.Reduction.Argmax_value ());
  show "argmax index (first of the ties)" ~cells
    (reduce_w Expr.Reduction.Argmax_index ());
  let empty = (Expr.Index.zero, Expr.Index.const 0) in
  let lo, hi = empty in
  show "empty sum" ~cells (reduce_w Expr.Reduction.Sum ~lo ~hi ());
  show "empty max" ~cells (reduce_w Expr.Reduction.Max ~lo ~hi ());
  show "empty argmax index" ~cells
    (reduce_w Expr.Reduction.Argmax_index ~lo ~hi ());
  show "modular sum"
    ~cells:[| Int64.max_int; 1L; 1L; 1L |]
    (reduce_w Expr.Reduction.Sum ());
  [%expect
    {|
    sum: agree; 3 3 3 3
    max: agree; 5 5 5 5
    argmax value: agree; 5 5 5 5
    argmax index (first of the ties): agree; 0 0 0 0
    empty sum: agree; 0 0 0 0
    empty max: agree; -9223372036854775808 -9223372036854775808 -9223372036854775808 -9223372036854775808
    empty argmax index: agree; 0 0 0 0
    modular sum: agree; -9223372036854775806 -9223372036854775806 -9223372036854775806 -9223372036854775806 |}]

(* ---- gathers ---------------------------------------------------------------- *)

(* out[w] = t0[w := t2[w]]: the index component is the stored value of an int64
   tensor, checked against [-extent, extent - 1] and normalized. *)
let gather name cells =
  let plan = Fusion_plan.default gather_kernel in
  let bind = i64_bind ~floats:[| 10.; 20.; 30.; 40. |] ~cells in
  Fmt.pr "%s: %a@." name Loop_check.pp_verdict (Loop_check.run plan ~bind)

let%expect_test "a gather index is checked in the int64 domain, then normalized"
    =
  gather "in range" [| 0L; 1L; 2L; 3L |];
  gather "negative wraps (-1 is the last)" [| -1L; -4L; -2L; -3L |];
  gather "-extent" [| -4L; -4L; -4L; -4L |];
  gather "extent - 1" [| 3L; 3L; 3L; 3L |];
  gather "extent" [| 4L; 0L; 0L; 0L |];
  gather "-extent - 1" [| -5L; 0L; 0L; 0L |];
  (* Narrowed first, these would wrap into small in-range ints. *)
  gather "near min_int" [| Int64.min_int; 0L; 0L; 0L |];
  gather "near max_int" [| Int64.max_int; 0L; 0L; 0L |];
  gather "2^32 (wraps to 0 in 32 bits)" [| Int64.shift_left 1L 32; 0L; 0L; 0L |];
  [%expect
    {|
    in range: agree
    negative wraps (-1 is the last): agree
    -extent: agree
    extent - 1: agree
    extent: agree on failure: gather_index_out_of_range
    -extent - 1: agree on failure: gather_index_out_of_range
    near min_int: agree on failure: gather_index_out_of_range
    near max_int: agree on failure: gather_index_out_of_range
    2^32 (wraps to 0 in 32 bits): agree on failure: gather_index_out_of_range |}]

(* ---- interleaved with float values ----------------------------------------- *)

let%expect_test
    "an int64 value runs between the float values it reads and is read by" =
  let bind =
    i64_bind ~floats:[| 1.5; 2.5; -1.5; 1e10 |] ~cells:[| 1L; 2L; 3L; 4L |]
  in
  let plan = Fusion_plan.default mixed_kernel in
  Fmt.pr "%a@." Loop_check.pp_verdict (Loop_check.run plan ~bind);
  (match Err.payload (Loop_lower.lower plan) with
  | Ok p ->
      let ids = ref [] in
      List.iter
        (function
          | Loop_stmt.For _ as s ->
              let rec store = function
                | Loop_stmt.Store { buffer; _ } -> Some buffer
                | Loop_stmt.For { body; _ } -> List.find_map store body
                | _ -> None
              in
              Option.iter
                (fun (b : Loop_buffer.t) ->
                  ids := Fmt.str "%a" Tensor_id.pp b.Loop_buffer.id :: !ids)
                (store s)
          | _ -> ())
        p.Loop_program.body;
      Fmt.pr "nest order: %s@." (String.concat " " (List.rev !ids))
  | Error _ -> ());
  [%expect {|
    agree
    nest order: t1 t3 t4 |}]
